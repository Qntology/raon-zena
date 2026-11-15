# Raon-1-nano 추론과 LLM 가속 (`raon-infer`)

`crates/raon-infer`는 KRAFTON Raon-1-nano의 순전파를 **candle**로 구현한 크레이트입니다. LLM 가속은 `Qntology/logis-center`와 같은 방식으로 붙였습니다. 즉 crates.io의 `candle-core`/`candle-nn` 0.11을 **`Qntology/candle-rocm-vulkan` 포크**로 교체(`[patch.crates-io]`)하고, 기능 플래그로 GPU 백엔드를 고릅니다.

## 백엔드 연결 (logis-center와 동일)

`Cargo.toml`(워크스페이스):

```toml
[workspace.dependencies]
candle-core = "0.11.0"
candle-nn = "0.11.0"

[patch.crates-io]
candle-core = { git = "https://github.com/Qntology/candle-rocm-vulkan", branch = "master" }
candle-nn = { git = "https://github.com/Qntology/candle-rocm-vulkan", branch = "master" }
```

`crates/raon-infer`와 `apps/raon-zena`의 기능 플래그:

| 기능 | 내용 | 필요 환경 |
|---|---|---|
| (기본) | CPU (SIMD k-quant 커널, rayon) | — |
| `cuda` | `candle-core/cuda`, `candle-nn/cuda` | CUDA 툴킷(`nvcc`), `CUDA_COMPUTE_CAP` |
| `rocm` | `candle-rocm` + `candle-core/rocm` (HIP 커널 + rocBLAS) | ROCm 6.x–7.2 / HIP SDK, 또는 ROCm Core SDK 10.x |
| `vulkan` | `candle-core/vulkan` (`ash`가 런타임에 로더를 열고, naga로 GLSL 커널 컴파일) | 빌드 시 SDK 불필요 |
| `metal` | `candle-core/metal` | macOS |
| `flash-attn` | 포크의 FlashAttention v1/v2/v3 계층 커널 | `cuda`/`rocm`/`vulkan`과 함께 |

빌드 예:

```bash
cargo build --release -p raon-zena --features cuda,flash-attn
```

```bash
cargo build --release -p raon-zena --features rocm
```

```bash
cargo build --release -p raon-zena --features vulkan
```

장치 선택은 logis-center의 `new_gpu_device`와 같은 순서(CUDA → ROCm → Vulkan → Metal → CPU)이며, `--llm-device auto|cpu|cuda|rocm|vulkan|metal`로 직접 지정할 수도 있습니다.

### 장치별 실행 경로

| 장치 | 양자화 행렬곱 | Raon GGUF 형식 지원 |
|---|---|---|
| CUDA | `QMatMul::forward_exact`: 활성값을 양자화하지 않습니다. 디코드는 블록을 커널 안에서 f32로 풀어 f32로 누산하고, 프리필은 타일 단위 역양자화 후 cuBLAS GEMM을 씁니다. | Q8_0 / Q4_K / Q5_K / Q6_K 모두 GPU |
| ROCm | `forward_exact`(f32 mat-vec 또는 rocBLAS) | 모두 GPU |
| Vulkan | `forward`. **Q8_0만** VRAM GEMV/GEMM 커널을 쓰고, K-quant는 매핑된 블록에서 CPU 커널로 돕니다. | Vulkan에서는 `--tier3 q8_0 --tier4 q8_0` 레시피를 권장 |
| Metal / CPU | `forward`(활성값을 Q8_K로 양자화하는 SIMD k-quant 커널). 이 장치에서 `forward_exact`는 호출마다 가중치 전체를 역양자화하므로 쓰지 않습니다. logis-center와 같은 정책입니다. | 모두 |

## 모델 구현

`model.rs`는 공개된 `modeling_raon1_nano.py`를 연산 단위로 옮긴 것입니다. 활성값은 f32입니다.

- **Raon-mHC (4-stream 잔차).**
  - 선택기 GEMM에 RMS 스케일을 곱합니다.
  - `h_pre = σ(·)`, `h_post = 2σ(·)`
  - `E = Q Z Qᵀ`(Helmert 기저), `ρ = max(0, max(−E/B))`, `Φ = B + E/(1+ρ)`
  - `H_res = (1−γ)I + γΦ`
- **RDA 18층.**
  - q, k, v 각각에 depthwise causal conv를 적용한 뒤 SiLU를 거칩니다.
  - q·k는 L2 정규화하고, q에는 `1/√d`를 곱합니다.
  - 감쇠는 **키 채널별**입니다: `log α = −5·σ(e^{A_log}·(f_b(f_a(x)) + dt_bias))`
  - 지우기 게이트는 `β = σ(b)`이고, 쓰기 게이트 `σ(b+δ)/σ(b)`는 값 스케일로 적용합니다.
  - V-first delta rule로 상태를 갱신합니다: `S = S·diag(α)`, `S += β (v − S k) kᵀ`, `o = S q`
  - 출력은 per-head RMSNorm 후 σ(gate)를 곱합니다.
  - **포크의 `fused_gated_delta_rule`(Qwen3.5 GDN)은 헤드당 스칼라 감쇠만 지원하므로 쓰지 않습니다.**
  - 재귀 계산은 디코드(T=1, GPU)에서는 디바이스 텐서 연산으로, 프리필과 CPU에서는 호스트 fp32 스캔(rayon, 헤드 병렬)으로 합니다.
- **MLA 6층 (NoPE, 게이트).**
  - `q = q_b(RMS(q_a x))`
  - `kv_a`의 출력을 512 latent + 64 공유 키 채널로 나눕니다.
  - `kv_b(RMS(latent))`의 출력을 k_nope와 v로 나눕니다.
  - 192차원 점곱에 `1/√192` 스케일을 쓰고, causal mask와 KV 캐시를 적용합니다.
  - 출력에 `σ(g(x))`를 곱합니다.
- **MLP와 PLE.**
  - MLP: SwiGLU
  - PLE: `p = (RMS(W_mp e) + 16·T[id]) / √2`. 층마다 `RMS(proj(gelu_tanh(gate(mean(stream)))·p))`를 모든 스트림에 더합니다.
- **임베딩 테이블.** 임베딩과 PLE 테이블은 GGUF mmap에서 토큰 행만 읽어 옵니다. 절삭된 PLE는 `.kept`/`.cold`에서 id를 다시 매핑해 읽습니다.
- **출력 head.** `full`(기본, 정확), `kept`(절삭 행만, 근사), `guarded`(cold 클러스터 상한) 중 고릅니다.
- **상태 캐시.** `State`(RDA conv·재귀 상태, MLA KV)는 복제 비용이 낮습니다. 그래서 Zena의 고정 프롬프트(페르소나 + 툴 정의)를 한 번만 프리필하고 요청마다 복제합니다(`Generator::prefix_state`).

## 검증 (공식 PyTorch 구현 대비)

`scripts/reference_logits.py`는 HF 공식 구현을 `trust_remote_code`, float32로 실행합니다. 그 출력과 `raon-chat --dump`의 출력을 `scripts/compare_logits.py`로 비교했습니다. 평가 텍스트는 `scripts/calibration/raon_eval.txt`(278토큰, 한국어/영어/툴콜)이고, 보정 텍스트와 겹치지 않습니다.

| 가중치 | 상주 | hidden 상대오차 | KL(ref‖x) 평균 | top-1 | top-5 |
|---|---|---|---|---|---|
| 원본 safetensors (정확 모드) | — | 2.6e-6 | **2.6e-11** | **100%** | 100% |
| 피라미드: PRD 원안 | 905 MiB | 0.151 | 0.100 | 87.1% | 84.4% |
| + imatrix, PLE Q8_0 | 905 MiB | 0.137 | 0.091 | 87.4% | 86.6% |
| + 초반 6층 MLP Q6_K | 947 MiB | 0.113 | 0.047 | 88.8% | 89.0% |
| **`--preset zena`** (+ MLP Q5_K) | **994 MiB** | — | **0.0235** | **91.4%** | **92.9%** |

결론:
- 정확 모드가 일치하므로 **모델 구현이 정확하다**는 것이 확인됩니다.
- 남는 차이는 모두 양자화에서 옵니다.

### 오차 기여도

원본 가중치에서 해당 그룹만 PRD 원안 GGUF 값으로 교체해 측정했습니다(`RAON_MIX_GGUF`/`RAON_MIX_PATTERN`, 45개 위치).

| 그룹 | 형식 | KL |
|---|---|---|
| MLP 전체 | Q4_K (경계 Q8_0) | **0.236** |
| └ gate/up | Q4_K | 0.187 |
| └ down | Q4_K | 0.071 |
| └ 2–7층 MLP만 | Q4_K | 0.202 |
| └ 16–21층 MLP만 | Q4_K | 0.029 |
| PLE 테이블 | Q4_K | 0.084 (Q8_0이면 0.00017) |
| 어텐션 | Q6_K | 0.0085 |
| PLE 투영·게이트 | Q8_0/Q6_K | 0.0029 |
| embed_tokens | Q8_0 | 0.00016 |

- **오차의 주원인은 초반 층 MLP의 Q4_K입니다.** imatrix는 도움이 되지만 효과가 작습니다(0.236 → 0.206).
- **PLE 테이블은 디스크 gather 대상**이라 Q8_0으로 올려도 VRAM 비용이 없습니다.
- 그래서 `--preset zena`의 구성은 다음과 같습니다.
  - MLP Q5_K
  - 앞쪽 경계 다음 6개 MLP는 Q6_K
  - PLE Q8_0
  - imatrix 사용
- 이 구성은 PRD 원안 대비 상주 메모리가 +89 MiB 늘어나는 대신 KL이 1/4로 줄어듭니다.

### imatrix

`raon-imatrix`는 원본 가중치로 보정 코퍼스(`scripts/calibration/raon_calib.txt`, 1,435토큰)를 순전파하며 모든 Linear 입력 열의 x² 평균을 모읍니다. 출력은 llama.cpp legacy 형식이며 `raon-quant --imatrix`가 바로 읽습니다.

```bash
./target/release/raon-imatrix --safetensors models/KRAFTON/Raon-1-nano --calib scripts/calibration/raon_calib.txt --out out/raon-1-nano.imatrix.dat
```

```bash
./target/release/raon-quant nano --preset zena --imatrix out/raon-1-nano.imatrix.dat --model-dir models/KRAFTON/Raon-1-nano --out out/raon-1-nano.zena.gguf
```

## 실사용 검증: Zena fast path

`cargo run --release -p raon-infer --example zena_fastpath -- out/raon-1-nano.zena.gguf models/KRAFTON/Raon-1-nano/tokenizer.json cpu`

입력은 `zena-core::prompt`의 실제 시스템 프롬프트와 툴 정의입니다(프리픽스 캐시 사용). 탐욕 디코딩 결과:

| 상황 | 출력 (파싱 결과) |
|---|---|
| 보스 돌진, 오른쪽 코앞(20°, 0.12), HP 18% | `control_zena_avatar`: gaze 20°/0.12, `FOCUSED`, `LEAN_FORWARD_POINT`, "조심해! 빠르게 회피하자!" |
| "앱 종료해 줘" | `manage_zena_system(EXIT_APP)` |
| "인벤토리에서 물약 어디 있어?" | 툴콜 없이 일반 문장 → 파이프라인이 대사로 말함 |

## 성능 (이 환경: 4코어 CPU, GPU 없음)

| 항목 | 값 |
|---|---|
| 디코드 순전파 | 약 240 ms/토큰 |
| └ MLP | 105 ms |
| └ RDA | 70 ms |
| └ MLA | 26 ms |
| └ mHC | 20 ms |
| └ PLE | 7 ms |
| └ mix | 6 ms |
| full head (25.9만 행, mmap Q8_0, 4스레드) | 약 140 ms |
| 탐욕 디코드 속도 | **2.0–2.7 tok/s** |
| 프리필 | 180–290 ms/토큰 (CPU k-quant는 행마다 vec_dot이라 토큰 수에 선형) |

**CPU 디코드는 메모리 대역폭에 묶여 있습니다.** 매 토큰 상주 가중치 약 1 GB를 읽어야 합니다. 이것이 GPU 가속이 필요한 이유입니다.
- GPU 처리 범위: 행렬곱, softmax/RMSNorm, 디코드 RDA 갱신은 모두 디바이스에서 돕니다.
- 호스트 왕복: 프리필 RDA 스캔과 head 결과 읽기만 호스트를 거칩니다.
- **실측 필요.** 이 환경에는 GPU가 없어 CUDA/ROCm/Vulkan 실행 속도는 재지 못했습니다. 실기에서 측정해야 합니다.

정확성 가드에 대한 실측 결과입니다.
- `guarded` head는 실제 hidden state에서 cold 행 234,993개 중 234,828개(99.9%)를 다시 계산했습니다. 1536차원에서 Cauchy–Schwarz 클러스터 상한이 사실상 가지치기를 못 하기 때문입니다.
- 그래서 기본 head는 `full`(전체 어휘, 정확)입니다.
- VRAM이 부족하면 `kept`(24,847행, 38 MiB)를 쓸 수 있지만, 근사이므로 출력이 달라질 수 있습니다. 실측으로도 초반부터 다른 문장이 나왔습니다.
- GPU에서 `full` head를 쓰면 `embed_tokens` Q8_0 404 MiB가 VRAM에 더 올라갑니다.

## 빌드 검증 현황

| 대상 | 결과 |
|---|---|
| Linux CPU | 빌드, 테스트, 실가중치 실행 통과 |
| Linux `vulkan`, `rocm` (`CANDLE_ROCM_SKIP_KERNEL_BUILD=1`) | `cargo check` 통과 |
| Windows (`x86_64-pc-windows-gnu`) 기본, `vulkan`, `rocm` | `cargo check` 통과 |
| `cuda` | 이 환경에 `nvcc`가 없어 미검사. `.github/workflows/cross-verify.yml`의 linux-cuda 잡이 apt CUDA 툴킷으로 타입 체크합니다(PR 또는 수동 실행 시). |

## 한계와 다음 단계

1. GPU 실기에서 tok/s와 VRAM을 측정해야 합니다(CUDA / ROCm / Vulkan).
2. **RDA fused 커널.** 포크의 `delta_rule.rs`에 키 채널별 감쇠와 쓰기 게이트를 지원하는 변형을 추가하는 패치가 필요합니다(CUDA/HIP). 지금은 프리필 RDA가 호스트 스캔이라 GPU 프리필에서 왕복이 생깁니다.
3. **MLA 커널.** `flash-attn` 기능의 fused 커널은 아직 MLA(Dqk 192 ≠ Dv 128)에 연결하지 않았습니다. 지금은 matmul + softmax 경로를 씁니다.
4. **MTP 추측 디코딩.** 55M MTP 가중치는 GGUF에 들어 있지만 디코딩 루프는 아직 없습니다. MTP는 SFT 단계 가중치이고, 공개 본체는 그 뒤 RL로 학습되었으므로 수락률을 측정해야 합니다.
5. **보정 코퍼스 확대.** 현재 1.4k 토큰이며, 게임 대화 로그로 늘리면 imatrix 품질이 나아질 수 있습니다.
