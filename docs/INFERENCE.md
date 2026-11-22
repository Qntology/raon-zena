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
| Vulkan | `forward` (`GpuQuant` VRAM GEMV/GEMM 셰이더). `candle@b90b12c` 부터 Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K 모두 장치 VRAM 에서 직접 계산합니다. | Q8_0 / Q4_K / Q5_K / Q6_K 모두 GPU |
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

## 로드 시점 피라미드 양자화 (GGUF 없이)

미리 GGUF로 변환하지 않고, **원본 safetensors를 mmap으로 열면서 그 자리에서 4계층 피라미드 양자화**를 적용할 수도 있습니다. `logis-center`가 SigLIP2 타워를 원본에서 바로 양자화하는 방식과 같은 발상이고, PRD §3.1의 "원본을 mmap으로 직접 읽어 단 1회 양자화"를 그대로 따릅니다.

```bash
./target/release/raon-chat --safetensors models/KRAFTON/Raon-1-nano --quantize zena --imatrix out/raon-1-nano.imatrix.dat --device auto --prompt "..."
```

```bash
cargo run --release -p raon-zena --features cuda -- --model models/KRAFTON/Raon-1-nano --quant-preset zena --imatrix out/raon-1-nano.imatrix.dat
```

동작 방식(`Weights::open_safetensors_quantized`):
1. **계획을 세웁니다.** `raon_quant::pyramid::nano_formats`가 GGUF 변환과 똑같은 티어 정책으로 텐서마다 형식을 정합니다.
   - T1: 원본 그대로
   - T2: Q8_0
   - T3: Q6_K
   - T4: Q4_K 또는 Q5_K
   - 프리셋은 `prd`, `zena`, `q8` 중에서 고릅니다.
2. **텐서마다 한 번 양자화합니다.** `Linear`를 만들 때 원본 BF16 행을 f32로 디코드하고, 가중 최소제곱 인코더(imatrix 반영)로 블록을 만들어 `qtensor_from_ggml`로 바로 디바이스에 올립니다. 중간 파일은 없습니다.
3. **gather 테이블은 양자화하지 않습니다.** `embed_tokens`와 PLE 테이블은 원본 BF16 행을 토큰마다 mmap에서 읽습니다. GGUF 경로(Q8_0)보다 정확하고, 상주 메모리는 0입니다. CPU의 `full` head도 BF16 테이블을 융합 커널로 직접 스캔합니다.
4. **로드 중에 읽은 원본 페이지를 해제합니다.** 로드 후 파일을 다시 매핑해, 이후 gather한 행만 RSS에 남게 합니다.

측정 결과(4코어 CPU, 페이지 캐시가 따뜻한 상태, 278개 위치 평가셋):

| 방식 | 로드 (캐시 있음 / 없음) | RSS (로드 후) | KL(ref‖x) | top-1 / top-5 |
|---|---|---|---|---|
| zena GGUF (미리 변환) | 1.3 s / 29.9 s | 1.09 GiB | 0.0235 | 91.4% / 92.9% |
| safetensors + 로드 시점 `zena`, 캐시 없음(`--no-quant-cache`) | 31.7 s / 약 115 s | 1.14 GiB | 0.0232 | 90.3% / 92.6% |
| safetensors + 로드 시점 `zena`, **블록 캐시 첫 실행** | 34.2 s (생성 32.8 s) | 1.16 GiB | 0.0232 | 90.3% / 92.6% |
| safetensors + 로드 시점 `zena`, **블록 캐시 재사용** | **1.2 s** | **1.09 GiB** | **0.0232** | 90.3% / 92.6% |

"캐시 없음"은 파일이 OS 페이지 캐시에 없어 이 컨테이너 디스크에서 처음 읽는 경우입니다(GGUF 3.0 GB, 원본 7.1 GB). 두 경로 모두 로드가 끝나면 원본 매핑을 다시 열어, 로드 중에 건드린 파일 페이지가 RSS에 남지 않게 했습니다. 이 처리 전에는 GGUF 경로의 RSS가 2.06 GB였습니다.

- **블록 양.** 247개 텐서, 11.2억 파라미터를 913 MB 블록으로 올립니다.
- **로드 시간 구성.** 인코딩 28.4 s, 원본 읽기 2.0 s, 업로드 0.4 s입니다.
- **원본이 페이지 캐시에 없을 때.** 디스크 읽기 시간이 더해집니다.
- **코어 수에 따른 변화.** 인코더는 rayon으로 병렬화되어 있어 코어 수에 거의 비례해 빨라집니다(16코어면 약 8 s로 추정).
- **인코더 개선.** 각 후보 스케일의 가중 오차를 두 번째 패스 대신 첫 패스의 합으로 닫힌 형태로 계산하도록 바꿨습니다(`kquant.rs`의 `[raon-zena]` 표시). 결과는 다음과 같습니다.
  - 속도: 20 → 33 Melem/s
  - 상대오차: 실제 Raon 텐서에서 원래 알고리즘과 소수점 6자리까지 같습니다(Q4_K 0.072400, Q5_K 0.035935, Q6_K 0.017543).
  - 블록 값: 95–99.8%가 비트 단위로 동일합니다.

### 런타임 블록 캐시 (logis-center 방식, 기본값)

`logis-center`가 SigLIP2 타워(`models/siglip2/gguf_quant.rs`)와 Qwen3.5 런타임 GGUF에 쓰는 구조를 그대로 따릅니다. 구현은 `raon_quant::cache`에 있습니다.

```text
<모델 폴더>/runtime-<프리셋>[-hg]-<fnv32(키)>/
    blocks.gguf   계획이 양자화하는 모든 행렬의 블록(원래 텐서 이름)
    report.json   텐서별 바이트와 상대오차
    READY         레시피 키 (마지막에 기록)
```

동작:
1. **첫 실행.** 원본 safetensors에서 블록을 만들어 저장하고, 마지막에 `READY`를 씁니다. 도중에 중단되면 `READY`가 없으므로 다음 실행에서 다시 만듭니다.
2. **다음 실행부터.** `READY`의 키가 맞으면 변환 없이 블록을 mmap해 바로 디바이스에 올립니다. 이 블록은 메모리 양자화 경로와 비트 단위로 같아, KL·top-1이 완전히 동일합니다.
3. **키 구성.** 아래 중 하나라도 바뀌면 키가 달라져 다시 만듭니다.
   - 캐시 형식, 인코더 버전(`kquant-wls-closedform/1`)
   - 프리셋과 전체 설정
   - imatrix 내용(FNV-64)
   - `-hg` 여부 (GPU의 `full` head용 Q8_0 `embed_tokens` 포함 여부)
   - 원본 샤드별 이름, 크기, 수정 시각
4. **정리.** 같은 종류(프리셋, `-hg` 여부)의 이전 캐시는 지웁니다. 그래서 CPU용과 GPU용 캐시는 함께 남습니다.
5. **저장 위치.** 모델 폴더에 쓸 수 없으면 사용자 캐시 폴더를 씁니다(`%LOCALAPPDATA%`, `$XDG_CACHE_HOME`, `~/.cache` 아래 `raon-zena/<모델>`). 여기에도 쓸 수 없으면 경고를 남기고 메모리 양자화로 계속합니다.
6. **포함하지 않는 것.** 정규화, 게이트 같은 T1 텐서와 gather 테이블은 원본에서 읽습니다.
7. **크기.** CPU용 913 MB입니다. GPU용은 `embed_tokens` Q8_0 404 MB가 더해집니다.

옵션:
- `--quant-cache DIR`: 캐시 위치를 직접 지정합니다.
- `--no-quant-cache`: 매번 메모리에서 양자화합니다.

어느 방식을 쓸지:
- **safetensors + 블록 캐시 (권장)**: 배포물은 원본 하나이고, 레시피나 imatrix를 바꾸면 자동으로 다시 만듭니다. 첫 실행 뒤에는 시작이 1초대입니다.
- **미리 변환한 GGUF**: 원본 7.1 GB 대신 3.0 GB만 배포하면 될 때 씁니다.
- **GPU 사용 시**: `-hg` 캐시가 `embed_tokens`의 Q8_0 블록도 갖고 있어, 재사용 때는 추가 인코딩이 없습니다.

### 필요할 때만 올리기 (앱)

앱에서는 Raon-1-nano를 상주시키지 않습니다. 사전이 `ask`로 넘긴 첫 요청 때 올라가고, `--llm-idle`(기본 120초) 동안 요청이
없으면 내려가서 VRAM과 RAM을 돌려줍니다(`LlmWorker::spawn_lazy`). 블록 캐시가 있으면 다시 올리는 데 약 1초가 걸립니다.
올라가는 중이거나 생성 중에 들어온 요청은 캐시된 대사로 바로 반응합니다. `--llm-idle 0`이면 계속 상주합니다.
대부분의 반응은 단어 사전과 화면 감지가 LLM 없이 처리합니다([EXPRESSION.md](EXPRESSION.md)).

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
