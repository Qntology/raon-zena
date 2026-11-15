# 아키텍처와 PRD 대응표

## 데이터 흐름

```
DXGI Desktop Duplication ──► zena-sensor ─────────────► nlp-nms-router ─────────► zena-core::reaction
 (capture.rs, 1/4 다운샘플)    log|ΔI| ≥ C 타일 집계       조사 박리 → 스팬 NMS        Lv1 시선만
                               ROI 클러스터 → 트래커       → 의도 점수                 Lv2 캐시 큐
                               (r, θ), Inward/Outward     시간축 의미 NMS             Lv3 Raon-1-nano
                                                          공간-텍스트 NMS                 │
stdin / (향후 ASR) ──── 발화 ─────────────────────────────────┘                          ▼
                                                                    prompt.rs (non-thinking 프리필)
                                                                    → LlmBackend → tools.rs 파서
                                                                                          │
winit 350×450 창 ◄── overlay.rs ◄── AvatarCmd (EventLoopProxy로 즉시 깨움) ◄─────────────┘
  └ wgpu DX12(DirectComposition) · avatar.wgsl · Win32 WS_EX_* · RegisterHotKey · 종료 시퀀스
```

## PRD 항목별 구현 위치

| PRD | 구현 | 상태 |
|---|---|---|
| §0 A안 (도메인 승인, 원본 직결 다운로드) | 개인 클라우드 환경 `raon-zena`(HF 및 CDN 호스트 허용), `scripts/fetch_models.sh` | 완료. SHA-256 검증 포함 |
| §0 참조 코드 클론 | `Qntology/logis-center` (공개 저장소, git 프록시로 읽기) | 분석 후 이식 |
| §2.1 극좌표, Inward/Outward | `zena-core/geometry.rs`, `zena-sensor` | 테스트 완료 |
| §2.1 로그 휘도 차분 셰이더 0.5 ms | `zena-sensor/src/event_diff.wgsl` (naga 검증), CPU 참조 구현 | 셰이더 검증만 함. GPU 디스패치 연결과 지연 측정은 하지 않음 |
| §2.2 NaFlex 패치 | `zena-sensor/naflex.rs` (공식 프로세서와 같은 이진 탐색과 패치 순서) | 테스트 완료 |
| §2.2 1152-d 특징을 `mm_projector`로 투영 | — | **불가**. 공개된 병합기가 없음(아래 참조) |
| §2.3 NLP + 임베딩 NMS | `nlp-nms-router` | 테스트 완료. 해시 임베더 사용 |
| §2.4 Lv1–Lv3, non-thinking fast path | `zena-core/reaction.rs`, `prompt.rs` (`<think></think>` 프리필) | 테스트 완료 |
| §2.4 Raon-1-nano 추론과 툴콜 | `raon-infer` (candle). 앱의 `RaonNanoBackend`가 전용 스레드에서 실행하고 프리픽스 상태를 캐시합니다 | 공식 구현과 로짓 일치. 실제 툴콜 생성 확인. GPU 실측은 남음 |
| §2.4 55M MTP 추측 디코딩 | MTP 가중치를 T2로 양자화해 GGUF에 포함 | 디코딩 루프는 미구현 |
| (LLM 가속) | `Qntology/candle-rocm-vulkan` 포크, 기능 플래그 `cuda`/`rocm`/`vulkan`/`metal`/`flash-attn` (logis-center와 동일) | Vulkan·ROCm·Windows는 `cargo check` 통과. CUDA는 CI(apt 툴킷)에서 검사 |
| §3 피라미드 양자화 | `raon-quant` | 실가중치로 완료([QUANTIZATION.md](QUANTIZATION.md)) |
| §3.5 Fused dequant-matmul, F32 누산, verify | `raon-quant/kernels.rs` (CPU); GPU에서는 포크의 `QMatMul::forward_exact` (CUDA/ROCm, 활성값 비양자화) | CPU 검증 완료. GPU 실측은 남음 |
| §4 툴콜 스키마, 2D/3D 매핑 | `zena-core/tools.rs`, `avatar.rs` | 테스트 완료. Live2D/VRM 렌더러는 미연결 |
| §5.1 마이크로 뷰포트, 단일 wgpu, 동적 FPS | `apps/raon-zena/overlay.rs` | Windows 타깃 `cargo check` 통과. 실기 미검증. LLM은 candle 백엔드(CUDA/ROCm/Vulkan)의 별도 GPU 컨텍스트에서 돌므로, 단일 wgpu 디바이스 공유는 센서·렌더에만 해당합니다 |
| §5.1 egui 말풍선, tray-icon | — | 미구현 (말풍선 대신 stderr 로그) |
| §5.2 Topmost, 클릭 투과, 호버 30% | `zena-core/window.rs`, `apps/raon-zena/win32.rs` | 플래그 계산은 테스트 완료. 적용은 실기 미검증 |
| §5.2 `Alt+드래그` | — | 클릭 투과 창은 마우스 입력을 받지 않으므로 저수준 마우스 훅(WH_MOUSE_LL)이 필요. 현재는 `Ctrl+Shift+L`만 지원 |
| §5.3 EXIT, TOPMOST, HIDE와 안전 종료 | `zena-core/window.rs`, `overlay.rs` | 시퀀스 테스트 완료 |

## PRD 가정 검증 결과

실제 모델 저장소(HF API, `config.json`, safetensors 헤더, 모델링 코드)로 확인한 사실입니다.

### Raon-1-nano

- **총 파라미터는 3.13B입니다.**
  - 본체 1.13B: 어텐션 0.42B, MLP 0.68B, 기타 0.03B
  - 임베딩 0.40B
  - **PLE 테이블 1.60B** (259,840 × 24층 × 256)
- PRD의 "1.5B Dense + 1.5B PLE"는 본체와 임베딩을 합한 1.53B로 보면 일치합니다.
- **층 구성.** 24층 = RDA(Raon Delta Attention, 선형 어텐션) 18층 + NoPE gated MLA 6층(4, 8, …, 24번째)이고, 모든 층은 Raon-mHC 4-stream 잔차로 연결됩니다. PRD의 "선형 18 + MLA 6"과 일치합니다.
- **MTP.** 54.8M이고 `mtp/model.safetensors`에 별도 파일로 있습니다.
  - SFT 단계에서 학습했고, 공개된 본체는 이후 RL로 학습했습니다.
  - 공식 vLLM 통합도 "실험적"으로 표시되어 있습니다.
- **입력 형식.** 텍스트 전용입니다. chat template에 `raon-1-nano supports text content only`라고 되어 있습니다.
  - 따라서 PRD §2.2의 `mm_projector`(1152→1536)에 해당하는 공개 가중치가 **없습니다**.
  - 비전 특징을 LLM에 직접 넣으려면 projector를 학습해야 합니다.
  - 현재 설계는 센서·비전 결과를 텍스트 상황 설명(`prompt::Situation`)으로 바꿔 넣습니다.
- **툴콜 형식.** `<tool_call><function=…><parameter=…>` XML 형식입니다. `zena-core::tools::parse_model_output`이 이 형식과 PRD의 JSON 형식을 모두 파싱합니다.
- **사고 과정 생략.** `enable_thinking=false`이면 `<think></think>`를 프리필합니다. Lv3 fast path가 이 방식을 씁니다.

### Raon-VisionEncoder

- **구성.** 1.14B = 비전 0.428B + 텍스트 0.708B(텐서 330개)이고 F32입니다. LocCa ViT-SO400M-16-SigLIP2이며 NaFlex를 지원하고 기본 패치 수는 256입니다. PRD와 일치합니다.
- **텍스트 타워를 빼면 런타임에 텍스트 임베딩을 만들 수 없습니다.** 자연어 요청과 화면 영역을 매칭하려면 두 방법 중 하나가 필요합니다.
  - 라벨 텍스트 앵커를 오프라인에서 미리 계산해 두기
  - 라우터 임베더를 같은 공간으로 정렬하기
- **로짓 파라미터.** `logit_scale`과 `logit_bias`는 T1로 보존해, 사전 계산한 앵커로 zero-shot 분류를 할 수 있게 했습니다.

### Raon-OpenTTS-0.3B

- **언어.** 영어 전용입니다. 학습 데이터 61.5만 시간이 모두 영어 데이터셋 11종입니다.
- **라이선스.** **CC-BY-NC-4.0(비상업)** 입니다.
- **가중치 형식.** `model_225000.pt`(5.4 GB) 학습 체크포인트이고 safetensors가 아닙니다.
- **구조.** F5-TTS DiT(336M) flow matching입니다. 다단계 ODE 샘플링과 16 kHz 보코더(`sbhifigan16k`, 저장소에 포함되지 않음)가 필요합니다.
- **결론.** PRD의 한국어 코칭 대사와 "45 ms 음성 코칭"에는 맞지 않습니다. 한국어 TTS를 따로 골라야 합니다.

### Qntology/logis-center

- **NMS의 실제 형태.** 이 저장소에서 "NLP 임베딩 NMS"에 해당하는 것은 다음과 같습니다.
  - 질의 단어 구간 스팬 NMS: 겹치면 억제, 점수→길이 순
  - 증거 중복 제거: 적응 임계 `clamp(μ+3σ, 0.90, 0.995)`
  - SigLIP2 크롭 박스 NMS: snap 0.85 / twin 0.90 / merge 0.25
  - 배타 할당 `exclusive_assign_by_score`
  - 이들을 이식했고, 시간 창과 사용자 발화 우선 규칙은 새로 추가했습니다.
- **임베딩 모델.** granite-embedding-97m(384-d)을 쓰며 KRAFTON 모델이 아닙니다. 임계값도 그 모델의 보정된 코사인 척도 기준입니다. 이 저장소에서는 해시 n-gram 임베더 기준으로 다시 맞췄습니다.
- **기존 측정 결과.** lm_head 절삭과 SigLIP2 피라미드 플랜은 이전에 측정 후 기각되었습니다. 절삭은 이 저장소에서 정확성 가드로 보완했고, 피라미드는 게임 화면으로 재평가가 필요합니다.

### KPI

- **VRAM 1.4 GB 이하.** 양자화한 가중치 기준으로 Nano 905 MiB + 비전 320 MiB = 1,225 MiB(1.28 GB)입니다. 여기에 KV/RDA 상태, 활성값, 렌더러, TTS가 더해져야 하므로 **TTS를 포함하면 1.4 GB를 넘을 가능성이 높습니다**.
- **RAM 25 MB / 52 MB.** PRD 본문(§1)과 총괄표(§6)의 수치가 서로 다릅니다. mmap 페이지 캐시를 RSS에 넣을지에 대한 기준도 정해야 합니다.
- **15 ms / 42.5 ms.** Windows와 GPU 실기에서 측정해야 합니다.

## 다음 단계 (우선순위)

1. GPU 실기(CUDA / ROCm / Vulkan)에서 tok/s와 VRAM을 측정합니다. CI(`cross-verify`)로 CUDA 빌드를 확인합니다.
2. 포크에 RDA용 fused 커널(키 채널별 감쇠, 쓰기 게이트)을 추가하고, MLA에 flash-attn을 연결하고, MTP 추측 디코딩을 구현합니다.
3. 보정 코퍼스를 게임 대화 로그로 확대하고 imatrix를 다시 만듭니다.
4. 센서·렌더 GPU 경로를 연결합니다. 순서는 다음과 같습니다.
   - `event_diff.wgsl` 디스패치
   - DXGI 텍스처를 DX12와 공유
   - 비전 타워 WGSL 융합 커널
5. 한국어 TTS를 선정하고(라이선스와 지연 시간 기준) ASR을 선정합니다.
6. Live2D Cubism Native SDK 또는 VRM(glTF) 렌더러 어댑터를 구현합니다(`zena-core::avatar` 매핑은 준비됨).
7. egui 말풍선과 tray-icon, WH_MOUSE_LL 기반 Alt+드래그를 구현합니다.
