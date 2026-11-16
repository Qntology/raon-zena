# premium-discount-mcp: premium.discount 동기화 · 추천/비추천

이 문서는 세 가지를 담습니다.

- "KRAFTON Raon Zena × premium-discount — MCP 기반 서버 동기화 개선 마스터 기획서"의 Phase 1 구현
- "추천/비추천 역할 분담 · 스키마 정의서"의 P0 · P2 구현
- 기획서를 두 저장소의 실제 코드(raon-zena `c84672a`, premium-discount `e91f752`)와 대조한 결과

**역할 분담.** premium-discount 가 맡는 것:
- 동기화
- 전역 추천/비추천 판정
- 스키마(평가 · 투표 · 판정 필드 이름과 판정 규칙 `contracts/verdict_v1.json`)

raon-zena 가 맡는 것:
- 캐릭터 챗
- 같은 판정 규칙을 기기 안에서 돌리는 로컬 판정(`crates/zena-verdict`)

이 MCP 서버의 이름은 기획서의 `zena-pd-sync-mcp` 에서 `premium-discount-mcp` 로 바꿨습니다. 도구의 인자와 결과는 premium-discount 의 필드 이름을 그대로 씁니다.

## 구조

```text
raon-zena ──McpClient (stdio JSON-RPC 2.0)──▶ premium-discount-mcp ──serve_lines (stdio JSON 한 줄씩)──▶ premium-discount-app
  SyncLoop (병합 · 남은 일이 있으면 10초)        pd_* 도구 8개, cc · hashId,           LanceDB 보관함(원문) · 투표 · 판정 캐시,
  zena-verdict (로컬 판정)                       GET /graph*                           embeddinggemma-2(768), QR 로그인
                                                        │                                    │
                                                        └──────────── HTTPS ─────────────────┴──▶ premium.discount
```

premium-discount-app 을 **라이브러리로 링크하지 않고 프로세스로 둡니다.** 그 크레이트는 LanceDB 0.31 · lance 8 · candle 0.9 · tokenizers 0.21 에 고정되어 있고, 이 작업 공간은 LanceDB 0.41 · lance 14 · candle 0.11 포크 · tokenizers 0.22 입니다. 링크하면 LanceDB(+ datafusion)와 candle 트리가 두 벌 빌드됩니다. 앱에는 이미 모든 명령을 여는 줄 단위 프로토콜(`serve_lines`)이 있으므로, 기획서 2.1절 그대로 그것을 `zena_avatar_mcp::protocol` 의 MCP 틀로 감쌌습니다. `protocol.rs` 는 한 줄도 바꾸지 않았습니다.

### 자식 프로세스 (`bridge.rs`)

- **처음 필요할 때 띄웁니다.** 앱은 시작하면서 embeddinggemma-2 를 올립니다. 그래서 쉬고 있는 제나는 모델도 LanceDB 도 메모리에 두지 않습니다.
- **죽으면 다음 명령에서 다시 띄웁니다.** `service.rs` 의 Track B 처럼 한 번 죽으면 끝나지 않습니다.
- **`--idle-secs`(기본 300) 동안 명령이 없으면 내립니다.** 단, 처리 중인 요청이 있거나(시간 초과로 포기한 요청도 셉니다) 모델 내려받기가 진행 중이면 내리지 않습니다. 보관함은 디스크에 있고 앱은 켤 때 `queued` · `pending` 행으로 큐를 다시 만들기 때문에 잃는 것은 없습니다.
- 기본 설정은 `PD_MODALITIES=text`(제나는 글만 보냄)와 `PD_QUANT=q8_0` 입니다. Q8_0 은 상주 약 300 MB 로 F32(약 660 MB)의 절반이고, 참조 벡터와의 코사인은 0.9999 입니다(premium-discount README 16절). 모델은 한 번에 하나만 올린다는 원칙에 맞춘 값입니다.

## 도구

| 도구 | premium-discount-app 명령 | 이 층이 더하는 것 |
|---|---|---|
| `pd_enqueue_node` | `cmd_submit_evaluation` | `cc` 변환, `to_key` → `to`, `bcc_key`(기본: `text`) → `bcc`(정규화 후 hashId), `parent_local` → `local_parent` |
| `pd_flush_and_sync` | `cmd_flush_outbox`(평가와 투표) + `cmd_sync_cache` | `next_since_ms`, 상태별 개수, `has_work`(못 보낸 투표 포함) |
| `pd_search_merged` | `cmd_search_merged` | `cc` 변환, `to_key`, `tier` 필터 검사. 결과마다 서버 판정(`verdict`)과 내 표(`my_vote`) |
| `pd_vote` | `cmd_vote` | 값(1 · −1 · 0)과 종류(`vote` · `vote_outcome`) 검사 |
| `pd_verdicts` | `cmd_verdicts` | `ids` 검사 (캐시 10분, 오프라인이면 `stale`) |
| `pd_graph_inspect` | (없음) 서버 `GET /graph?cc=`, `/graph/local?id=`, `/graph/similar?id=&topK=` | 노드 · 간선 종류별 개수, 작성자 수, 스레드(root) 크기, 서버 등급 분포 |
| `pd_auth_session` | `cmd_session` · `cmd_login_start` · `cmd_login_status` · `cmd_logout` | — |
| `pd_model_manage` | `cmd_model_status` · `cmd_model_download` · `cmd_model_download_cancel` | 자식 프로세스 상태(`running`, `starts`, `idle_ms`) |

도구 오류는 `<kind>: <message>` 형식입니다. `kind` 는 premium-discount-app 의 `CommandError.kind`(`offline`, `not_ready`, `invalid_input`, `rejected`, `queue_full` …)를 그대로 씁니다. 앱을 띄우거나 닿을 수 없으면 `unavailable` 입니다.

### `cc` · `to` · `bcc`

서버는 `cc` 로 **등록 가능한 도메인만** 받습니다(`^([a-z0-9-]{1,63}\.)+[a-z]{2,63}$`, 64자 이하). 기획서의 `app://code.exe/coding` 은 `:` 와 `/` 때문에 400 으로 거부됩니다. 그래서 다음처럼 바꿉니다.

| 입력 | `cc` | 비고 |
|---|---|---|
| `store.steampowered.com` | `store.steampowered.com` | |
| `https://www.example.com:8443/a?b` | `example.com` | 호스트만, 앞의 `www.` 를 뗌 |
| `app://Code.exe/coding` | `code.exe` | `scene: "coding"` 은 따로 돌려줌 |
| `app://Code - Insiders.exe` | `code-insiders.exe` | 라벨은 `[a-z0-9-]` 로 접음 |
| `app://firefox` | `firefox.exe` | 글자만으로 된 마지막 라벨이 없으면 `.exe` 를 붙임 |
| `app://카카오톡.exe` | `x<sha256 8자리>.exe` | ASCII 가 남지 않으면 해시 |

장면(scene)을 `cc` 에 넣지 않은 이유는 서버가 `cc` 를 정확히 일치로 비교하기 때문입니다. 장면을 넣으면 한 프로그램이 장면 수만큼 다른 도메인으로 갈라져, 같은 프로그램을 쓰는 다른 사용자의 스레드와 만나지 못합니다.

`to` · `bcc` 는 웹 클라이언트의 `hashId(text)` = `ethers.computeAddress(ethers.hashMessage(text)).toLowerCase()` 로 만듭니다. EIP-191 해시를 secp256k1 개인 키로 써서 그 공개 키의 주소를 구하는 방식입니다. 서버는 `ethers.isAddress` 만 검사하므로 아무 20바이트나 통과하지만, 같은 방식을 써야 다른 클라이언트의 값과 비교할 수 있습니다. 서버 저장소에 들어 있는 ethers 6.9.0 번들로 뽑은 8개 값과 바이트 단위로 같습니다(`address.rs` 테스트). `bcc` 는 NFC 정규화 · 공백 접기를 한 뒤 해시하므로, 다른 창에서 읽은 같은 오류 문구는 같은 주소가 됩니다.

### 추천/비추천: 전역 판정은 서버, 로컬 판정은 `zena-verdict`

전역 판정은 premium-discount 서버가 만듭니다.
- 앱의 `cmd_vote` 가 `POST /evaluations/votes` 로 표를 보내고, 서버 cron 이 1분마다 다시 판정합니다.
- 결과는 `GET /evaluations/verdicts` 와 검색 응답의 `verdict` 로 내려옵니다.
- 세는 표는 로그인 확인된 남의 글 표뿐입니다. 게스트 표와 자기 글 표는 저장되지만 가중치가 0 입니다.

로컬 판정은 같은 규칙을 기기 안의 관측(결과 · 대화 피드백 · 수동 보정)에 돌린 값이고, 서버로 올리지 않습니다.

판정 규칙(PD-Verdict v1)은 premium-discount 의 `contracts/verdict_v1.json` 이 기준입니다. 이 파일은 그 저장소의 `contracts/gen_verdict_v1.py`(참조 구현)가 만든 매개변수와 사례 19개입니다. 세 구현이 같은 사례를 통과합니다.
- 서버 cron 의 TypeScript(`cron/src/verdict.ts`)
- 이 저장소의 `zena-verdict`(Rust)
- 앱 E2E 가짜 서버의 파이썬(참조 구현을 그대로 import)

`zena-verdict` 는 계약 파일의 사본을 들고 있습니다. 시험은 두 가지를 확인합니다.
- 사본이 기록된 SHA-256 과 같은지
- 옆에 `premium-discount` 저장소가 있으면 원본과 같은지

처음 Phase 1 에서 둔 기기 안 등급 라벨(`sds_tier` · `zena_tiers.json`)은 없앴습니다. 서버에 판정이 생겼고, 로컬 판정의 원천은 raon-zena 의 궤적 저장소가 될 것이기 때문입니다.

## 앱 쪽: `PdSyncClient` · `SyncLoop`

`SyncLoop` 은 premium-discount `app/ui/coalesce.ts` + `main.ts` 를 옮긴 것입니다.

- 쉬고 있을 때 들어온 요청은 바로 실행합니다.
- 실행 중에 들어온 요청은 버리지 않고, 끝난 뒤 **딱 한 번** 다시 실행합니다. 그중 하나라도 flush 를 원했으면 다시 실행할 때 flush 합니다.
- `queued` · `pending` 행이 남아 있으면 10초마다 flush 를 실행합니다. 이 flush 는 못 보낸 투표도 함께 보냅니다.
- 행은 `request_id` 로 모아 둡니다(웹 UI 의 Dexie 캐시와 같은 역할). 오프라인에서도 로컬 행과 검색은 그대로 동작합니다.

UI 의 `hasWork()` 는 최신 50행만 보지만, 이쪽은 모아 둔 행 전체를 봅니다.

## 기획서 대조

✔ 맞음 · △ 일부 · ✘ 틀림

| 절 | 기획서 주장 | 실제 코드 |
|---|---|---|
| 1.1 | `zena-nlp` · `zena-sds` 가 구축되어 있음 | △ Qntology upstream main(PR #3, `c84672a`)에만 있습니다. sungu-che 포크와 로컬 main 은 PR #2 에서 멈춰 있었습니다. |
| 1.1 · 2.1 | `McpClient` 의 ping 헬스체크와 실패 시 폴백 | ✘ 클라이언트에는 `ping` 메서드가 없고, 서버가 보낸 ping 에 답하기만 합니다. 폴백은 `service.rs` 에 있으며 다시 띄우지 않습니다. 또한 `[profile.release] panic = "abort"` 이므로 `serve` 의 `catch_unwind` 는 릴리스 빌드에서 동작하지 않습니다(도구가 패닉하면 프로세스가 끝남). |
| 1.1 | `StreamToAvatar` 가 zena-core 에 있음 | △ `apps/raon-zena/src/pipeline.rs` 에 있습니다. zena-core 에 있는 것은 `ToolCallStream` 입니다. |
| 1.1 | 서버가 익명 authorKey(HMAC 12자리)를 남김 | ✘ 저장하지 않습니다. `GET /graph` 를 읽을 때 `from` 으로 계산합니다. |
| 1.2 | 텍스트 전용 574MB Q8_0 | ✘ 574 MB 는 내려받는 BF16 크기입니다. Q8_0 은 로드할 때 양자화하며 상주 메모리는 약 300 MB 입니다. |
| 1.2 · 4.1 | SDS 등급(premium/discount)이 확정되면 자동 적재 | ✘ 대조 당시 `zena-sds` 에도 서버에도 등급이 없었습니다. 지금은 서버 투표 · 판정과 `zena-verdict` 를 더했습니다. 로컬 판정은 서버로 올리지 않습니다. |
| 2 | 드래그 앤 드롭 조사, Auto-Dock(82%), 3단계 클릭 UI | ✘ 코드에 없습니다. 지금 있는 것은 AlwaysOnTop, 고스트 모드의 마우스를 올렸을 때 30% 투명, Ctrl+Shift+T 채팅창뿐입니다. 드래그는 누르는 즉시 `drag_window` 로 넘어가 클릭과 구분되지 않습니다. |
| 2 | `VisionKind::Privacy` 면 픽셀을 지움 | △ 정의는 `zena-lexicon` 에 있습니다. 픽셀은 프라이버시 화면만이 아니라 **모든 프레임**에서 인코딩 전에 지웁니다. 프라이버시 화면이면 60초 쉽니다. |
| 3 | `command_contract.json` 의 13개 명령 | ✘ 14개입니다(`cmd_search_local` · `cmd_search_remote` · `cmd_app_status` 포함). |
| 3 | `parent_local` 인자 | △ 명령 인자 이름은 `local_parent` 이고, `parent_local` 은 보관함 열 이름입니다. 도구는 `parent_local` 을 받아 바꿔 넘깁니다. |
| 3 | `pd_enqueue_node` 는 `queued` 행만 기록 | △ `submit_outbox` 는 행을 쓴 **직후 바로 전송**합니다. 온라인이면 `saved` / `accepted`, 오프라인이거나 앞선 항목을 기다리면 `queued` 입니다. |
| 3 | `POST /search` | ✘ `POST /evaluations/search` 입니다(공개용은 `POST /public/evaluations/search`). |
| 3 | `GET /graph/node`, Premium/Discount 분포 | ✘ `/graph/node` 는 없고 `/graph/local` · `/graph/similar` 가 있습니다. 서버에 등급이 없어서 분포도 없습니다. |
| 3 | `pd_model_manage` 가 Raon-1-nano 가중치도 관리 | ✘ premium-discount 의 내려받기는 embeddinggemma-2 만 받습니다. `modalities` 는 내려받기 인자가 아니라 로드 설정이고, 내려받기 인자는 `variant: text | full` 입니다. |
| 4.1 | `cc = app://<exe>/<scene>` | ✘ 서버 정규식이 거부합니다. 위 표처럼 바꿉니다. |
| 4.1 | 창 exe · 제목 · URL 을 읽고 OCR 번역 | ✘ `WindowFromPoint` 같은 호출도 OCR 도 없습니다. 캡처는 주 모니터 전체를 DXGI 로 받는 것뿐입니다. |
| 4.1 | 승률 ≥ 0.65 | ✔ `TIE_WIN_RATE = 0.65` |
| 5 | 로컬 원문 BM25/벡터 검색 | △ premium-discount 의 로컬 검색은 벡터만 씁니다. BM25 는 zena-lexicon 에만 있습니다. |
| 5.1 | coalesce 의미, 10초 타이머 | ✔ (UI 는 최신 50행만 봅니다) |
| 6 | `index.ts` 의 "768차원 벡터화 … inf/ref 를 합쳐 Premium/Discount 분류" | ✘ 그런 문장은 없습니다. 실제 메모는 "rust로 스케줄링하기 / Qwen3.5 0.8B / embeddinggemma-300m 768차원 / 5단계 호가 / 참조값 · 유사값" 입니다. `inf` 는 옛 `items` 표에만 있고, 지금 API 의 `ref` 는 **부모 평가의 UUID** 이지 외부 유사값이 아닙니다. |
| 6 | chromiumoxide 듀얼 브라우저 | ✘ 코드에 없습니다. |
| 7 | Phase 1 왕복 테스트에 FakeD1 / FakeVectorize 사용 | △ 이 둘은 TypeScript(`node:sqlite`) 가짜라서 Rust 에서 쓸 수 없습니다. 대신 각본 백엔드(`tests/server.rs`)와 premium-discount 의 `mock_server.py`(`tests/e2e_pd_app.rs`)를 썼습니다. |

## 시험

```bash
cargo test -p premium-discount-mcp -p zena-verdict
```

시험은 세 묶음입니다.

- **`premium-discount-mcp`**: 단위 18개 + 통합 6개입니다. 통합 시험은 세 가지를 봅니다.
    - 각본 백엔드로 도구가 명령에 어떻게 대응하는지(투표 · 판정 · `tier` 필터 포함)
    - 기본값과 거부
    - 실제 바이너리를 `McpClient` 로 띄웠을 때 앱 없이 `unavailable` 오류가 나는지와 `/graph` 읽기
- **`zena-verdict`**: 5개입니다. 계약 사례 19개 전부, 매개변수가 같은지, 사본 해시, 옆 저장소 원본과의 대조를 확인합니다.
- **premium-discount 쪽**: 앱 단위 240개, UI 50개, 계약 대조(TypeScript) 3개입니다.

실제 premium-discount-app 과의 왕복은 직접 켜야 돌아갑니다.

```bash
python -X utf8 premium-discount/app/tests/e2e/mock_server.py 8787 requests.jsonl &
PD_E2E_APP=…/premium-discount-app.exe PD_E2E_MODEL_DIR=…/model PD_E2E_SERVER=http://127.0.0.1:8787 \
PD_E2E_LOG=requests.jsonl cargo test -p premium-discount-mcp --test e2e_pd_app -- --nocapture
```

시나리오는 다음 순서입니다.

1. 오프라인에서 부모 · 자식(`parent_local`)을 `queued` 로 쌓습니다.
2. 로컬만 검색해 `remote_error` 를 확인합니다.
3. 가짜 서버를 검수 대기 상태로 두고 연결을 되살립니다. `SyncLoop` 에 겹친 요청은 한 번의 flush 로 합쳐지고, 자식의 `ref` 가 부모의 서버 id 로 바뀝니다.
4. 타이머가 스스로 `pending` 을 다시 확인하다가, 검수가 풀리면 `stored` 로 끝내고 멈춥니다.
5. 게스트로 표를 보냅니다(`counted=false`). 판정은 근거 0, `unrated`, `low_evidence` 입니다.
6. QR 로그인 뒤 다른 글에 비추천합니다(`counted=true`). 판정은 근거 1.0, `down` 1 이고, 최소 근거 2 에 못 미쳐 아직 `unrated` 입니다. 다시 물으면 캐시가 답합니다.
7. 검색 결과에 판정이 붙고, `tier: rated` 필터는 아직 판정 난 글이 없어 비어 있습니다.

2026-11-16 실측입니다(Windows, premium-discount-app 릴리스, 텍스트 전용 Q8_0):

- **모델 내려받기**: `pd_model_manage download` 로 574.3 MB 를 105초에 받았습니다. SHA-256 검사를 통과했고 `installed=text` 입니다.
- **첫 `pd_enqueue_node`**: 앱 시작과 모델 로드를 포함해 3.3 ~ 4.0초입니다.
- **동기화 실행 기록** `(runs, queued, pending, stored)`: `(1,0,2,0) (2,0,2,0) (3,0,2,0) (4,0,0,2)`. 이후 타이머 실행은 없습니다.
- **가짜 서버가 받은 요청**: 768차원, 노름 1.0000003, 자식의 `ref` 는 부모의 id 입니다.
- **전체 시나리오**: 14.2초입니다. 투표 · 판정 단계를 더한 뒤에는 15.0초입니다.

## 남은 일 (기획서 Phase 2 ~ 4 의 전제)

- **Phase 2**
  - 클릭과 드래그 분리(지금은 누르면 바로 창 이동).
  - 드롭한 곳의 창 정보: `WindowFromPoint` → exe · 제목, 브라우저 URL.
  - 영역 OCR. raon-zena 에는 OCR 엔진이 없고 indra-ocr 은 라이선스 문제로 쓸 수 없습니다.
  - Auto-Dock 과 82% 투명.
  - 드롭 처리기에서 `PdSyncClient::enqueue` 를 부르고 `SyncLoop::request(true)` 로 이어 주기.
- **Phase 3**: Stage 0 ~ 2 UI. `SyncSnapshot::status_dot` 과 `rows` 가 동기화 점과 히스토리의 데이터 원천입니다.
- **Phase 4**
  - `zena-sds` 위의 등급 엔진(`relative-value-sds.md` Phase 2).
  - 서버에 등급을 올리려면 premium-discount 서버 계약부터 바꿔야 합니다.
  - chromiumoxide 크레이트.
