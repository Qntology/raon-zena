# RAON ZENA

**KRAFTON Raon Zena(크래프톤 라온 제나)** 는 게임 화면을 실시간으로 보고 시선·표정·짧은 코칭 대사로 반응하는 인게임 AI 버튜버 컴패니언입니다. 이 저장소는 Master PRD v3.0을 바탕으로 만든 Rust 워크스페이스입니다.

> 상태: **기반 구현 단계**입니다. 다음은 실제 가중치나 테스트로 검증했습니다.
> - 양자화기: 원본 KRAFTON 가중치를 실제로 양자화했습니다.
> - **Raon-1-nano 추론 엔진(candle)**: 공식 PyTorch 구현과 로짓이 일치(KL 2.6e-11, top-1 100%)하고, 실제 모델이 Zena 툴콜을 생성합니다.
> - LLM 가속은 `logis-center`와 같은 `candle-rocm-vulkan` 포크로 붙였습니다(CUDA / ROCm / Vulkan / Metal). 단, 이 환경에 GPU가 없어 GPU 실행 속도는 측정하지 못했습니다.
> - 이벤트 센서, NLP-NMS 라우터, 툴콜·아바타 매핑
> - Windows 클라이언트: 타입 체크까지만 했습니다.
>
> TTS는 아직 스텁입니다. 자세한 내용은 [현재 범위와 한계](#현재-범위와-한계)를 보세요.

## 구성

| 경로 | PRD 단계 | 내용 |
|---|---|---|
| `crates/zena-core` | §2.1 · §2.4 · §4 · §5 | 화면 중앙 기준 극좌표 `(r, θ)` 변환과 Inward/Outward 흐름 판정, Lv1–Lv3 반응 라우팅, 툴콜 스키마와 Raon 네이티브 `<tool_call>` 파서, Live2D/VRM 파라미터 매핑(한글 중성 기반 비셈 립싱크), Win32 스타일 계산, 안전 종료 시퀀스, FPS 스로틀 |
| `crates/zena-sensor` | §2.1 · §2.2 | 로그 휘도 차분 **WGSL 컴퓨트 셰이더**(naga로 검증)와 동일한 알고리즘의 CPU 참조 구현, 타일 ROI 클러스터링, 트래커, NaFlex 패치 추출 |
| `crates/nlp-nms-router` | §2.3 | `Qntology/logis-center`에서 이식한 기능: 조사 박리, 스팬 NMS, μ+3σ 적응 중복 임계, snap/twin 박스 NMS, 배타 할당. 여기에 시간축 의미 NMS와 공간-텍스트 NMS를 더했습니다. |
| `crates/raon-quant` | §3 | 원본 safetensors를 mmap으로 읽어 **단일 패스 4계층 피라미드 양자화**를 하고 GGUF v3로 씁니다. 그 밖에 imatrix 반영, `--preset zena`(측정 기반 품질 레시피), `kept_ids` 어휘 절삭, 융합 dequant-dot 커널, 자가 검증이 들어 있습니다. |
| `crates/raon-infer` | §2.4 | **Raon-1-nano 순전파**(RDA 18층 + MLA 6층 + Raon-mHC + PLE)를 candle로 구현했습니다. `cuda`/`rocm`/`vulkan`/`metal`/`flash-attn` 기능 플래그와 `Qntology/candle-rocm-vulkan` 포크를 씁니다. 그 밖에 GGUF 블록 디바이스 상주(`QMatMul`), 프리픽스 상태 캐시, `raon-chat`, `raon-imatrix`가 들어 있습니다. [`docs/INFERENCE.md`](docs/INFERENCE.md) |
| `apps/raon-zena` | §5 | Windows 오버레이(350×450 마이크로 뷰포트). 구성 요소: winit, 단일 wgpu DX12 디바이스(DirectComposition), DXGI Desktop Duplication, Win32 레이어 플래그와 전역 단축키, 절차적 플레이스홀더 아바타 |
| `scripts/` | §0 A안 | 원본 가중치 병렬 다운로드와 SHA-256 검증(`fetch_models.sh`), llama.cpp `gguf-py` 교차 검증(`xcheck_gguf.py`), 공식 PyTorch 참조 로짓과 비교(`reference_logits.py`, `compare_logits.py`), 보정·평가 텍스트(`calibration/`) |
| `.github/workflows/cross-verify.yml` | — | CPU 테스트와 `vulkan`/`rocm`/`cuda`(apt CUDA 툴킷) 빌드 검사, Linux·Windows. PR 또는 수동 실행 시 동작합니다. |

## 빠른 시작

```bash
scripts/fetch_models.sh models
```

```bash
cargo build --release
```

```bash
./target/release/raon-quant vision --model-dir models/KRAFTON/Raon-VisionEncoder --out out/raon-ve-vision.pyramid.gguf
```

```bash
./target/release/raon-imatrix --safetensors models/KRAFTON/Raon-1-nano --calib scripts/calibration/raon_calib.txt --out out/raon-1-nano.imatrix.dat
```

```bash
./target/release/raon-quant nano --preset zena --imatrix out/raon-1-nano.imatrix.dat --model-dir models/KRAFTON/Raon-1-nano --out out/raon-1-nano.zena.gguf
```

```bash
cp models/KRAFTON/Raon-1-nano/tokenizer.json out/
```

```bash
./target/release/raon-chat --gguf out/raon-1-nano.zena.gguf --device auto --prompt "보스가 내려찍기 할 때 어떻게 피해?"
```

```bash
cargo test --workspace --release
```

Windows 클라이언트는 아래 명령으로 실행합니다. `--features`로 GPU 백엔드를 고르며, 실행 중 stdin에 문장을 입력하면 제나에게 말을 걸 수 있습니다.

```bash
cargo run --release -p raon-zena --features cuda,flash-attn -- --model out/raon-1-nano.zena.gguf --llm-device auto --vision out/raon-ve-vision.pyramid.gguf
```

단축키:
- `Ctrl+Shift+Q`: 작별 인사 후 종료
- `Ctrl+Shift+H`: 숨기기 / 보이기
- `Ctrl+Shift+L`: 클릭 투과 모드와 드래그 모드 전환

## 실측 결과 (원본 가중치, 이 저장소의 도구로 생성)

| 모델 | 원본 | 산출 GGUF | 상주(Device) | 디스크 gather | 시간(4코어 CPU) |
|---|---|---|---|---|---|
| Raon-1-nano + MTP, PRD 원안 | 7.06 GB BF16 (3.13B + 55M) | 2,173 MiB | **905 MiB** (949 MB) | 1,261 MiB (`embed_tokens`, PLE 테이블) | 185 s |
| Raon-1-nano + MTP, `--preset zena` + imatrix | 같음 | 3,018 MiB | **994 MiB** | 2,022 MiB (PLE Q8_0) | 약 190 s |
| Raon-VisionEncoder (비전 타워만) | 4.54 GB F32 (1.14B) | 320 MiB | **320 MiB** | — (텍스트 타워 0.708B 제외) | 26 s |

계층별 가중 상대오차 ‖x̂−x‖/‖x‖:

| 계층 | 형식 | 상대오차 |
|---|---|---|
| T1 | F32 / BF16 원본 | 0 |
| T2 | Q8_0 | 0.51% |
| T3 | Q6_K | 1.76% |
| T4 | Q4_K | 7.2% |

공식 PyTorch 구현 대비 품질(278개 위치, KL 평균 / top-1):

| 레시피 | KL | top-1 |
|---|---|---|
| 원본 가중치 (정확 모드, 구현 검증) | 2.6e-11 | 100% |
| PRD 원안 | 0.100 | 87.1% |
| `--preset zena` + imatrix | **0.0235** | **91.4%** |

자세한 내용은 [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md), [`docs/INFERENCE.md`](docs/INFERENCE.md), [`docs/reports/`](docs/reports)에 있습니다.

검증 방법:
- 다운로드한 원본 파일 5개의 SHA-256이 HF 공개값과 일치합니다.
- 독립 구현인 llama.cpp `gguf-py`로 읽어 dequant해도 같은 오차가 재현됩니다. 이로써 ggml 비트 호환성을 확인했습니다.
- 커널 자가 검증 결과 최대 오차는 6.2e-7입니다(기준 1e-3).
- 실제 Nano 가중치에서 절삭 head(24,847행)와 가드로 구한 argmax가 전체 259,840행 argmax와 일치합니다.
- 공식 HF 구현(`trust_remote_code`, float32)과 우리 candle 구현의 로짓이 278개 위치 모두에서 일치합니다.

## 현재 범위와 한계

PRD의 가정 중 실제 모델·코드로 확인한 결과 다르거나 추가 작업이 필요한 항목입니다. 상세는 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#prd-가정-검증-결과)에 있습니다.

1. **Raon-OpenTTS-0.3B를 그대로 쓸 수 없습니다.**
   - **영어 전용**이고 라이선스가 **CC-BY-NC-4.0(비상업)** 입니다.
   - 가중치가 safetensors가 아니라 5.4 GB 학습 체크포인트(`.pt`)입니다.
   - F5-TTS 계열 flow-matching이라 다단계 샘플링과 별도 보코더가 필요합니다.
   - 따라서 한국어 대사와 45 ms 목표에는 별도 TTS가 필요합니다.
2. **`mm_projector`가 존재하지 않습니다.** Raon-1-nano는 텍스트 전용입니다(chat template: "supports text content only"). 1152→1536 비전-언어 병합기는 공개되지 않았으므로 학습이 필요합니다. 현재는 비전·센서 결과를 텍스트 상황 설명으로 바꿔 프롬프트에 넣습니다.
3. **`kept_ids` 절삭 효과가 PRD 가정과 다릅니다.**
   - Raon 토크나이저는 원래 **96%가 한국어/영어**라서, 문자 영역 필터만으로는 4%만 줄어듭니다.
   - 그래서 빈도 상위 N개 상주(`--keep-top`, 기본 24,576)와 cold 행 클러스터 상한 검사를 넣어 **결과가 정확하도록** 했습니다.
   - 다만 실제 hidden state에서 가드가 cold 행의 **99.9%** 를 다시 계산했습니다. 1536차원에서는 상한이 거의 가지치기를 못 한다는 뜻입니다.
   - 그래서 추론 기본값은 전체 어휘 head(`--head full`, 정확)입니다. 절삭 head(`kept`)는 근사 옵션으로 남겼습니다.
4. **VRAM 예산이 목표를 넘습니다.**
   - Nano 상주는 PRD 원안 905 MiB, 품질 레시피 994 MiB로, 목표 890 MB보다 큽니다.
   - PRD 원안(MLP Q4_K)은 KL 0.10으로 품질 손실이 크고, 주원인은 초반 MLP층입니다. [`docs/INFERENCE.md`](docs/INFERENCE.md#오차-기여도) 참고.
   - GPU에서 정확한 `full` head를 쓰면 `embed_tokens` 404 MiB가 VRAM에 더 올라갑니다.
   - 비전은 320 MiB로, 목표 240 MB보다 큽니다. SO400M의 폭 1152와 4304가 256의 배수가 아니라 K-quant 패딩이 약 11% 생기기 때문입니다. 전부 Q4_K로 내려도 하한이 약 267 MB입니다.
5. **NLP-NMS 임베딩 모델이 Raon이 아닙니다.** logis-center는 granite-embedding-97m을 씁니다. 이 저장소는 "100% Raon" 원칙에 따라 의존성 없는 해시 n-gram 임베더를 기본으로 두었고, `TextEmbedder` trait으로 교체할 수 있습니다.
6. **추론 엔진은 구현했지만 GPU 실측이 남았습니다.**
   - Raon-1-nano는 llama.cpp가 지원하지 않는 커스텀 아키텍처라 candle로 직접 구현했고, 공식 구현과 일치합니다.
   - CPU(이 환경, 4코어)에서는 2–2.7 tok/s로 메모리 대역폭에 묶입니다.
   - CUDA/ROCm/Vulkan 경로는 컴파일과 타입 체크만 했고, 실기 속도와 VRAM은 측정해야 합니다.
   - RDA 프리필은 호스트 스캔입니다. 포크의 fused GDN은 RDA 수식(키 채널별 감쇠, 쓰기 게이트)과 맞지 않습니다.
   - MTP 추측 디코딩 루프는 아직 없습니다.
   - 음성 입력(ASR) 모델도 스택에 없습니다.
7. **KPI는 검증하지 않았습니다.** 15 ms / 45 ms 지연과 RAM 25 MB는 Windows와 GPU 실기에서 측정해야 합니다. 이 환경(Linux 컨테이너, GPU 없음)에서는 Windows 타깃 `cargo check`까지만 했습니다.

## 라이선스

이 저장소는 Apache-2.0입니다. 이식한 코드의 출처와 라이선스는 [`NOTICE`](NOTICE)에 있습니다. 모델 가중치는 저장소에 포함하지 않으며 각 모델의 라이선스를 따릅니다.
