# 범용 엔진 통합 · 런타임 판 맞추기 (2026-11-17)

"범용 AI/센서/창 엔진(`crates/`) 대통합 및 런타임 버전 동기화 마스터 기획서"를 반영한 기록입니다. 그 앞의 이름 표준화 · 1차 분리는 [`MASTER_PLAN_CRATES_UNIFICATION.md`](MASTER_PLAN_CRATES_UNIFICATION.md), premium-discount 쪽 기록은 그 저장소 README 23절과 `crates/README.md` 입니다.

## 도메인 경계

| | premium-discount | raon-zena |
| --- | --- | --- |
| "도메인" | 사용자가 머문 장소 · 대상 · 화면 영역 (`cc` · `to` · `Region`) | 제나의 표현 사전 (`<group>.<name>`, `zena_lexicon`) |
| 집합 | 열린 집합 (URL · 실행 파일 · 화면 영역이 즉석에서 정규화 · 해시) | 닫힌 · 큐레이션 집합 (`--learn` 으로 세 번 본 구절만 더함) |
| 이번에 둔 곳 | 수학 · 모델 · 센서 · 형식 코어 (`premium-discount-*`) | 사전, 제나의 동작(`Action`) · 호출어 · 말투, Raon 모델 전용 계획, 접착부 |

코어는 캐릭터 타입을 모릅니다. 그래서 다음처럼 받습니다.
- `DomainBank<M, E>`: 항목 내용 `M` 은 부르는 쪽 타입입니다. raon-zena 에서는 `zena_lexicon::Action` 입니다.
- `DomainSpec` · `VisionDomainSpec`: 사전에서 만드는 중립 명세입니다.
- `set_wake_words`: 호출어는 부르는 쪽이 정합니다(제나야 …).

## 옮긴 것 · 만든 것

| 이전 (raon-zena) | 지금 | raon-zena 에 남은 것 |
| --- | --- | --- |
| `crates/onnxruntime(-sys)` | premium-discount `onnxruntime(-sys)` | 없음 (지움) |
| `crates/zena-nlp` | `premium-discount-nlp` | 없음 (지움) |
| `crates/nlp-nms-router` 의 korean · normalize · analyze · embed · domain · span · semantic · spatial | `premium-discount-nms` | `nlp-nms-router`: `domain`(사전 → 은행, `LexiconBank`, `WAKE_WORDS`) · `router`, 예전 모듈 경로는 다시 내보냄 |
| `crates/zena-sds` | `premium-discount-sds` | `zena-sds`: `pub use premium_discount_sds::*` (장부 recipe 그대로) |
| `crates/zena-sensor`, `zena-core::geometry`, `zena_lexicon::VisionKind` | `premium-discount-sensor` | `zena_core::geometry` · `zena_lexicon::VisionKind` 는 다시 내보냄 |
| `crates/raon-vision` | `premium-discount-vision` (사전 의존 없음) | `raon-vision`: 다시 내보내기 + `vision_specs` · `LexiconAnchors::build` · 사전 틀의 `text::compute_anchors` |
| `crates/raon-quant` 의 safetensors · gguf · imatrix · kquant · kernels · vocab · lmhead | `premium-discount-quant` (+ 새 `trim`) | `raon-quant`: 피라미드 계획 · Raon-1-nano 블록 캐시 · `raon-quant` 명령, 예전 경로는 다시 내보냄 |
| (없음) | `premium-discount-relay` | 아직 부르는 곳 없음 |
| `zena-core::window` 의 스타일 비트 · 뷰포트 · 알파 · 프레임 간격 | `premium-discount-window` | `zena_core::window` 가 이 crate 로 계산 (제나의 모드 · 반응 단계를 옮겨 넘김) |

## 런타임 판

- candle 포크
  - raon-zena 의 `[patch]` 를 `branch = "master"` 에서 `rev = ebaf93d` 로 바꿨습니다.
  - raon-infer 의 `candle-rocm` 도 같은 rev 로 고정했습니다. 하나만 branch 로 두면 cargo 가 ROCm 쪽 candle 을 고르지 못합니다.
- premium-discount 앱: lancedb 0.41 · tokenizers 0.22 로 맞췄습니다.
- 결과: 네 잠금 파일(raon-zena, premium-discount `app/` · `app/src-tauri/` · `crates/`)이 같은 판을 고릅니다.

| 의존 | 판 |
| --- | --- |
| candle | `rev=ebaf93d` |
| lancedb · lance | 0.41.0 · 14.0.2 |
| tokenizers | 0.22.2 하나 |
| datafusion | 54.1 |

## 기획서와 다른 점

- **단일 Cargo.lock 은 없습니다.** 저장소가 둘입니다. 대신 네 잠금 파일이 같은 판을 고르게 맞췄습니다.
- **premium-discount-app 은 여전히 프로세스로 둡니다**(Tauri · tokio · 앱 상태). 판 충돌은 이제 그 이유가 아닙니다.
- **raon-zena 에 얇은 crate 가 남습니다.** 사전과 Raon 모델에 기대는 부분이라 옮기지 않았습니다.
  - `raon-vision`
  - `raon-quant`
  - `zena-sds` (경로 유지)
- **호출어는 프로세스 전체 설정입니다**(`set_wake_words`). raon-zena 는 은행을 만들 때 넣습니다. 한 프로세스에서 다른 호출어를 쓰는 두 앱이 함께 돌면 안 됩니다.
- **사전이 필요한 시험은 raon-zena 에 남았습니다.** 코어에는 같은 수의 합성 시험을 넣었습니다.
  - 구간 NMS "한 발화 두 도메인"
  - 도메인 은행의 정확 · 접두 일치
- **Relay**
  - 기획서는 키 묶기를 "SDS 상한 띠(`relay.cos`)"로 적었지만, 실측해 보니 키 몇 개로 낸 띠(0.881)는 같은 이름의 한 · 영 짝도 막았습니다.
  - granite 실측값:

    | 짝 | 코사인 |
    | --- | --- |
    | 말레니아 ↔ malenia | 0.833 |
    | 라다곤 ↔ radagon | 0.848 |
    | 다른 이름끼리 | 0.739 ~ 0.771 |

  - 그래서 "서로 가장 가까운 짝 + 바닥 0.80 + 차이 0.03" 으로 묶습니다. 문장 맞추기는 띠(평균 + 2σ, 바닥 0.75)를 그대로 씁니다.
  - 포함률 하한(0.5)은 고정값입니다.
  - 한국어 Stanza 원형(`말레니아+의`)은 첫 형태소만 씁니다.
- **어휘 잘라 내기**
  - logis-center 규칙을 `premium-discount-quant::trim` 으로 옮기고 `ColdGuard` 와 합쳤습니다.
  - Raon-1-nano 경로(`raon-infer`)는 지금 규칙(`vocab::kept_ids_from_tokenizer`)을 그대로 씁니다. 어휘의 약 96 % 가 이미 한국어 · 영어라 언어 규칙의 이득이 작습니다.
- **창 계산**
  - 누르기와 끌기(5 px) · 제자리 돌아가기 · 점 아래 창 찾기는 crate 에 있습니다. raon-zena 오버레이는 아직 누르면 바로 끌기입니다.
  - 고치는 것은 overlay.rs 의 다음 일입니다.

## 검증

| 묶음 | 결과 |
| --- | --- |
| raon-zena `cargo check --workspace --all-targets` | 통과. 예전에 실패하던 onnxruntime 시험 대상이 옮기면서 고쳐져 `--exclude` 없이 통과 |
| raon-zena `cargo test --workspace` (premium-discount `9925011` 고정) | 전부 통과: nlp-nms-router 11 + eval_ko 3, raon-infer 3, raon-quant 1, raon-vision 1, 앱 17, zena-avatar-mcp 32 + 3, zena-core 30, zena-lexicon 8, zena-verdict 7 |
| `eval_ko` (93문장, `--release --features stanza`, 실제 granite · Stanza) | 옮기기 전과 같음: v1 64/93 · 오탐 9, v1 + 규칙 67/93 · 오탐 6, v2 74/93(79.6 %) · 오탐 1, 항목별(정확 33/34 · 바꿔 말하기 21/38 · 다중 3/3 · 부정 8/8 · 무관 9/10)까지 |
| google SigLIP2 (`premium-discount-vision` 예제) | 옮기기 전과 같음: 이미지 0.999999 ~ 1.000001, 텍스트 0.999999, 토큰 6/6 |
| premium-discount 쪽 | crates 전부(실제 모델 시험 포함), 앱 237, UI 71, lance 8 ↔ 14 호환, 실제 E2E, src-tauri 확인 (README 23절) |
