# Raon Zena × premium-discount 5대 과업 반영 (2026-11-17)

> 같은 날 뒤이은 작업에서 `pd-*` crate 는 `premium-discount-*` 로 이름이 바뀌었고, `premium-discount-mcp` 는 premium-discount 저장소로 옮겨 갔습니다([`MASTER_PLAN_CRATES_UNIFICATION.md`](MASTER_PLAN_CRATES_UNIFICATION.md)). 아래는 그 전의 기록입니다.

"KRAFTON Raon Zena × premium-discount — 5대 핵심 과업 통합 마스터 기획서"를 두 저장소(raon-zena · premium-discount, 브랜치 `claude/master-plan-5`)에 반영한 기록입니다. 각 절은 한 일, 기획서와 코드가 달랐던 점, 검증 순서로 적습니다.

## 1. 공유 라이브러리 단일화

| 항목 | 한 일 |
| --- | --- |
| 의존 연결 | 작업 공간 `Cargo.toml` 의 `[workspace.dependencies]` 에 `pd-verdict` · `pd-identity` · `pd-nav` 를 `git = "https://github.com/Qntology/premium-discount", rev = "540311b"` 로 고정했습니다. CI 는 raon-zena 만 받으므로 `path` 대신 git 리비전을 씁니다. 옆 체크아웃으로 빌드하려면 `[patch."https://github.com/Qntology/premium-discount"]` 에 `path` 를 적습니다. 잠금 파일에 새로 들어온 크레이트는 hmac · pbkdf2 · ecdsa · rfc6979 · signature 다섯 개이고, 겹치는 크레이트(k256 0.13.4, sha2 0.10.9, serde 1.0.229 …)는 판이 같아 충돌이 없습니다. |
| `zena-verdict` | 판정 수식(375줄)을 지우고 `pub use pd_verdict::*` 로 다시 내보냅니다. `feedback`(대화 피드백 → 관측) · `speak`(말하는 방식)만 남았고, 예전 이름 `RECIPE` 는 `pd_verdict::RECIPE_V1` 입니다. 계약 사본(`contracts/verdict_v1.json`)은 고정한 리비전을 이 저장소에서 다시 시험하려고 남겼습니다. |
| `premium-discount-mcp/address.rs` | `k256` · `tiny-keccak` 직접 의존을 없애고 `pd_identity::hash_id` · `hash_message` 를 부릅니다. `Option` → `anyhow::Result` 변환만 둡니다. `compute_address` 를 직접 부르던 시험 하나(`invalid_private_keys_fail`)는 대응 함수가 pd-identity 에 없어서 지웠습니다. |
| `pd-nav` 연결 | raon-zena 에는 아직 접근성 내비게이션 화면이 없어서, MCP 도구 `pd_nav_number` 로 연결했습니다. 앱 프로세스 · 모델 없이 `merge` → `number` → `sensitive_screen` 을 부르고, 개인정보 화면이면 `record: false` 를 돌려줍니다(문서 `PREMIUM_DISCOUNT_MCP.md`). |
| `find_pd_app` | `$PD_APP` → 실행 파일 옆 → `<home>/pd-app.json` → `%LOCALAPPDATA%\premium.discount\` · `%ProgramFiles%\premium.discount\` · macOS `/Applications` → `PATH`. 설치 폴더 이름은 Tauri NSIS 설치기의 기본값(`$LOCALAPPDATA\${PRODUCTNAME}`)이고, 제품 이름은 `tauri.conf.json` 의 `premium.discount` 입니다. |

**기획서와 다른 점**
- 익명 지갑 서명을 "MCP 도구에서 그대로 활용"하는 부분은 넣지 않았습니다. 지갑을 쓰려면 시드 구문을 어딘가에 보관해야 하는데, 보관 방식(DPAPI · Keystore · QR 옮기기)은 아직 결정 전입니다. MCP 도구로 시드 구문을 주고받으면 모델 대화 문맥에 남습니다.
- 설치 폴더 탐색이 실제로 앱을 찾으려면, premium-discount 데스크톱 설치본이 헤드리스 코어 `premium-discount-app` 을 사이드카(`bundle.externalBin`)로 함께 넣어야 합니다. 지금의 Tauri 번들은 창 실행 파일(`premium-discount-desktop`)만 넣습니다.

## 2. 캐릭터 에셋 카탈로그와 셰이더

### 2.1 카탈로그 (`zena-avatar-mcp::catalog`, 도구 `list_avatar_models` · `fetch_avatar_model`)

- 항목마다 고정 URL(GitHub 커밋, Arweave 트랜잭션) · 바이트 수 · SHA-256 을 둡니다. 내려받기는 `<파일>.part` 에 쓰고 크기 · 해시가 맞을 때만 이름을 바꿉니다. 이미 있는 파일도 해시를 다시 봅니다.
- 위치는 `%LOCALAPPDATA%\RaonZena\assets\catalog\<id>\` 입니다. 사용자 항목은 `<assets>\catalog.json` 에 둡니다. 같은 id 는 바꿔치고, `default` 를 붙이면 기본값을 가져갑니다. `..` · 절대 경로 · http(로컬 시험 서버 제외)는 거부합니다.
- `AssetService` 의 `ServiceConfig::catalog` 가 있으면:
  - `Subculture3d` 에 `vrm_path` 가 없을 때 카탈로그 기본 `.vrm` 을 받아 씁니다. 전에는 오류를 내고 절차적 얼굴로 돌아갔습니다.
  - `Subculture2d` 에 키트가 없을 때 카탈로그 기본 키트를 받아 씁니다. 받을 파일이 있는 키트일 때만이고, 키트가 바뀌면 캐시 이름(`body_2d_<id>.png`, `stamps_8f_<id>`)도 바뀝니다.
- 앱은 기본으로 켭니다. `--no-avatar-download` 로 끕니다.

| id | 모델 | 라이선스 (VRM 메타에서 확인) |
| --- | --- | --- |
| `vrm1-twist-sample` (3D 기본) | pixiv 의 VRM 1.0 샘플 (three-vrm · vrm-specification) | VRM Public License 1.0: 아바타 사용 누구나, 법인 상업 이용 가능, 재배포 · 수정 허용, 표기 불필요 |
| `seed-san` | VirtualCast 의 VRM 1.0 샘플 | VRM Public License 1.0, 표기 필요 |
| `100avatars-rose` · `100avatars-rabbit` | Polygonal Mind 100Avatars (로우폴리, VRM 0.x) | CC0 |
| `builtin-chibi` (2D 기본) | 코드로 그리는 내장 키트 | Apache-2.0 |

**기획서와 다른 점**
- "CC0 고퀄리티 기본 VRM" 은 찾지 못했습니다. VRoid 공식 CC0 샘플은 VRoid Hub 로그인 뒤에만 받을 수 있어 URL · 해시를 고정할 수 없고, 바로 받을 수 있는 CC0 VRM(100Avatars)은 로우폴리입니다. 그래서 3D 기본값은 VRM Public License 1.0 인 pixiv 샘플로 했습니다. 이 샘플의 VRM 메타가 사용 누구나 · 상업 이용 · 재배포를 허용합니다. CC0 만 쓰려면 `catalog.json` 에서 `100avatars-rose` 에 `default` 를 붙입니다.
- 2D 는 `kit.json` 레이어 구조(hair_back · body · face · eyes · mouth · hair_front · hands · fx)에 맞는 공개 일러스트 키트를 찾지 못했습니다. PNGTuber · Inochi2D 형식은 레이어 구조가 다릅니다. 그래서 받을 수 있는 키트를 넣는 길(사용자 `catalog.json`)만 만들었고, 기본값은 지금처럼 내장 키트입니다.

### 2.2 `mesh.wgsl` (MToon-lite)

- **음영 색**: 어두운 면을 회색으로 낮추지 않고, 바탕색에 보라빛 웜톤 `(0.80, 0.66, 0.90)` 을 곱합니다. 밝기는 예전 회색 바닥(앰비언트 + 나머지의 62 %)에 맞췄습니다.
- **하프 램버트**: `N·L × 0.5 + 0.5` 를 `smoothstep(0.30, 0.65)` 으로 넘깁니다. 경계는 N·L 약 −0.4 ~ +0.3 에 걸칩니다(전: −0.05 ~ 0.12).
- **프레넬 림**: 실루엣을 따라 라일락빛 흰색을 더하고, 어두운 면 쪽을 더 세게 둡니다. 카메라가 −z 방향 직교 투영이라 시선 벡터를 +z 로 둡니다.
- Track A 로봇(forge · SolidCraft `.glb`)도 같은 셰이더를 씁니다.
- 확인: `raon-zena --snapshot DIR --vrm model.vrm`(새 장면 `6_trackB2_vrm_joy` · `7_trackB2_vrm_side`)으로 전 · 후를 같은 GPU(RX 6600, DX12)에서 그려 비교했습니다. 4개 카탈로그 모델 모두 실제 URL 에서 받아 이 크레이트의 파서로 읽힙니다(VRM 1.0: 정점 24,951 · 34,059 / 표정 18, VRM 0.x: 4,091 · 2,145 / 표정 6).
- **남은 것**: 모델 자신의 MToon 재질값(`VRMC_materials_mtoon` 의 shadeColorFactor 등)은 아직 읽지 않습니다. 모든 재질에 같은 음영 색을 씁니다. 장면 틀은 여전히 전신 T 포즈라 얼굴이 작게 보입니다.

## 3. premium-discount-app 3단계 오버레이

premium-discount README 20절에 적었습니다. 요약하면 다음과 같습니다.
- 창: 360×56 → 360×260 → 400×580, 투명 · 테두리 없음 · 항상 위, 오른쪽 아래 도킹. 화면이 `ui/stage.ts`(순수 함수)로 크기 · 위치를 바꿉니다.
- `app/src/*` 는 바꾸지 않았습니다.
- 대조하다가 찾은 결함: `cmd_vote` · `cmd_verdicts` 가 코어와 화면에는 있었지만 Tauri 창에 등록되어 있지 않아, 창에서 투표하면 실패했습니다. 등록했습니다.
- 앱에는 지갑 명령과 피드 명령이 없습니다. 그래서 피드는 내 평가(Dexie 행)이고, 지갑 줄은 세션 주소와 "지갑 명령은 아직 앱에 없음"만 보여 줍니다.

## 4. embeddinggemma-2 · granite 역할 분담

새로 만든 코드는 없습니다. 기획서의 표는 premium-discount README 16 ~ 17절에 이미 구현 · 실측되어 있습니다.
- Q8_0 텍스트 층 520.1 → 138.1 MB, 코사인 0.999906
- 임베딩 표 mmap 상주 0
- 타워 지연 상주 · 교차 스왑(`residency.rs`)
- 비전 층 0 · 오디오 층 8 F32 유지

기획서와 숫자가 다른 곳은 하나입니다. "짧은 문장 단건 120 ~ 160 ms" 는 F32 기준(약 0.15초)이고, 기본값인 Q8_0 Dequant 경로는 약 0.28초입니다(README 16-5). 역할 분담은 이미 그렇게 되어 있습니다.
- raon-zena 의 `zena-nlp` · `nlp-nms-router` 와 `pd-nav` 의 라벨 공간 `granite-97m-r2-cls-v1` 은 granite 를 씁니다.
- premium-discount-app 의 평가 · 검색은 embeddinggemma-2 를 씁니다.

## 5. 비전 인코더: google/siglip2-so400m-patch16-naflex

`raon-vision` 이 Raon-VisionEncoder(timm 이름)와 google 체크포인트(HF 이름)를 모두 읽습니다. 헤더에 `vision_model.embeddings.patch_embedding.weight` 가 있으면 HF 로 봅니다.

| 수정 | 내용 |
| --- | --- |
| `layout.rs` (새 파일) | timm 이름으로 요청하면 HF 텐서로 답합니다. q · k · v 행을 이어 fused `qkv` · `in_proj` 를 만들고, 헤드의 packed `in_proj` 를 latent query(앞 1152행)와 kv(뒤 2304행)로 나눕니다. **패치 투영의 열 순서를 `(p_h, p_w, c)` → `(c, p_h, p_w)` 로 바꿉니다.** |
| `tower.rs` | HF 는 위치표를 줄일 때 antialias 를 씁니다(transformers `resize_positional_embeddings(antialias=True)`). timm 은 쓰지 않습니다. 격자 한 변이 16 보다 작을 때(1080p · 256 패치는 12 × 21) 값이 달라집니다. |
| `zena-sensor::naflex` | `Preprocess { mean, std, resample }`. Raon-VE 는 CLIP 평균 · 표준편차 + bicubic, google 은 0.5 / 0.5 + bilinear(PIL 방식 커널)입니다. `raon_vision::preprocess_for` 가 `preprocessor_config.json` 을 먼저 보고, 없으면 레이아웃으로 정합니다. |
| `text.rs` | google `tokenizer.json` 은 파일 안에 64 고정 패딩 · 자르기 설정이 있습니다. 그대로 쓰면 EOS 가 패딩 뒤로 밀려서 끕니다(시험에서 0/6 → 6/6). |
| `raon-quant` 피라미드 | `.layers.` 층 번호, `text_model.` 제외, `position_embedding` 은 spine, `.self_attn.` 은 T3 로 둡니다. |
| 앵커 지문 | `fingerprint` 를 `raon-ve:<크기>` · `siglip2-hf:<크기>` 로 바꿨습니다. 화면 감시기는 지문이 다른 앵커를 쓰지 않습니다. 사전 서비스는 앵커가 없거나 지문이 다르면, 모델 옆 `tokenizer.json`(또는 `--screen-tokenizer`)으로 텍스트 타워를 CPU Q8_0 으로 한 번 올려 다시 계산하고 내립니다. |

**기획서와 다른 점**
- "코드 변경 없이 그대로 유지"는 틀렸습니다. 패치 투영의 열 순서, 위치표 antialias, 리샘플 필터, 토크나이저 패딩 설정 네 곳이 조용히 틀린 값을 냅니다. 로드 단계에서는 오류가 나지 않습니다.
- "기존 지문 비교 로직이 자동 감지"도 틀렸습니다. 원래 지문은 파일 크기뿐이었고, 실행 중에 비교하는 곳이 없어서 다른 체크포인트의 앵커를 그대로 썼습니다. 이번에 비교와 재계산을 새로 넣었습니다.
- 앵커는 `zena_lexicon.json` 이 아니라 LanceDB `anchors` 표에 있습니다. JSON 에는 프롬프트와 틀만 있습니다.
- 피라미드 GGUF 는 q · k · v 를 따로 씁니다. 런타임(`VisionTower`)은 GGUF 가 아니라 safetensors 를 읽고 로드할 때 잇습니다.

**검증** (CPU, google 체크포인트 4.54 GB, `scripts/siglip2_reference.py` → `ve_xcheck`)

참조는 transformers 5.9 의 `Siglip2Model` 입니다. 체크포인트를 strict 로 읽어 빠진 이름 · 남는 이름이 없음을 확인했고, 전처리는 `Siglip2ImageProcessor` 입니다.

| 묶음 | F32 | Q8_0 |
| --- | --- | --- |
| 토크나이저 | 6/6 (한국어 1개 포함) | 6/6 |
| 텍스트 임베딩 최소 코사인 | 0.999999 | 0.999902 |
| 이미지 임베딩 (공식 패치 8건) | 0.999999 ~ 1.000001 | 0.99797 ~ 0.99930 |
| 이미지 임베딩 (우리 전처리 8건) | 0.999464 ~ 1.000000 | 0.99683 ~ 0.99933 |

피라미드 변환(실제 파일)은 450개 텐서를 320 MiB 로 바꿨습니다(T3 92 = 가운데 23층 × q · k · v · out, T4 46, T2 28, spine 284).

Raon-VisionEncoder 가중치는 이 PC 에 없어서 timm 경로를 다시 대조하지는 못했습니다. 그 경로는 이름이 그대로 넘어가고 위치표도 예전 함수를 쓰므로, 바뀐 값이 없습니다.

## 시험

| 묶음 | 결과 |
| --- | --- |
| `cargo test -p zena-verdict -p premium-discount-mcp` | zena-verdict 7, MCP 단위 18 + 서버 7 (`pd_nav_number` 포함) + E2E 1 |
| `cargo test -p zena-avatar-mcp` | 단위 34 (카탈로그 4, 카탈로그로 VRM · 키트를 채우는 서비스 시험 1) + 통합 3, 실제 내려받기 시험은 `-- --ignored` 로 따로 돌려 통과 |
| `cargo test -p raon-vision -p zena-sensor --lib` | 10 + 9 |
| premium-discount `app/ui` `npm test` | 70 (새 `stage.test.ts` 18) |
| premium-discount `app/src-tauri` `cargo check` | 통과 |
| `cargo test -p raon-zena --no-default-features` | 17 |
| `cargo check --workspace --all-targets` (LanceDB 기능 포함) | `onnxruntime` · `onnxruntime-sys` 를 빼면 통과. 그 둘의 시험 대상(`test_env_log` 가져오기 실패, bindgen 배치 시험의 null 역참조 오류)은 이번 변경 전 트리에서도 실패합니다 |
