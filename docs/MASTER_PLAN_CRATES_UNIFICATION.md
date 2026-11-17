# 공용 라이브러리 통합 · 이름 표준화 (2026-11-17)

"공용 라이브러리(`crates/`) 통합 및 네이밍 표준화 마스터 기획서"를 반영한 기록입니다. premium-discount 쪽 기록은 그 저장소 README 22절과 `crates/README.md` 입니다.

## raon-zena 에서 바뀐 것

| 항목 | 내용 |
| --- | --- |
| 의존 | 작업 공간 `Cargo.toml` 이 `premium-discount-*` crate 9개를 `git = "https://github.com/Qntology/premium-discount"` · `rev = "8f2a796"` 으로 고정합니다(아래 "fork 커밋"). 옆 체크아웃으로 빌드하려면 파일 끝의 `[patch]` 주석을 풉니다 |
| `crates/premium-discount-mcp` | 지웠습니다. premium-discount 의 `crates/premium-discount-mcp` 로 옮겨 갔고, 설계 문서도 그 crate 의 README 입니다(`docs/PREMIUM_DISCOUNT_MCP.md` 는 안내만 남김) |
| `zena-avatar-mcp::protocol` | 파일을 지우고 `pub use premium_discount_mcp_protocol as protocol;` 로 바꿨습니다. Track A · B 코드는 그대로입니다. MCP `initialize` 의 `clientInfo.name` 은 고정값 `raon-zena` 대신 실행 파일 이름입니다(raon-zena 앱이면 같은 값) |
| `zena-verdict` | `premium_discount_verdict::*` 를 다시 내보냅니다 |
| 아바타 카탈로그 | `.part` · 크기 · SHA-256 확인 · 이름 바꾸기 · 안전한 경로를 `premium-discount-manifest` 로 합니다(앱의 모델 내려받기와 같은 규칙). `sha2` 직접 의존을 없앴습니다 |
| `zena-sds::Welford` | 필드 · 직렬화(`n` · `mean` · `m2` · `ring`)는 그대로 두고, 더하기 · 합치기 · 분산을 `premium_discount_sds::Moments` 로 계산합니다(수식이 같아 값도 같음) |

## 기획서와 다른 점

- **`premium-discount-mcp` 를 링크해 쓰는 raon-zena crate 는 아직 없습니다.** 작업 공간 의존 목록에 올려 두었을 뿐입니다. 앱에 드롭 처리기(`PdSyncClient::enqueue` → `SyncLoop::request(true)`)가 없기 때문입니다. 서버 실행 파일은 `premium_discount_mcp::find_mcp_server()` 가 `$PD_MCP` → 이 실행 파일 옆 → premium.discount 데스크톱 설치 폴더 순서로 찾습니다.
- **필터 판정**은 기획서의 `Pass` · `Block { reason }` 이 아니라 기존 그대로 `Clean` · `Profane` 입니다. 앱의 `ProfanityFilter` 트레이트는 오류 타입이 앱의 것이라 앱에 남았습니다.
- **명령 수**는 16이 아니라 17입니다.
- **패키지 별칭**(`pd_verdict = { package = … }`)은 두지 않고 한 번에 바꿨습니다. 두 저장소를 같이 고쳤기 때문입니다.

## fork 커밋

`8f2a796` 은 아직 sungu-che/premium-discount 의 `claude/master-plan-5` 브랜치에만 있습니다. GitHub 는 fork 의 커밋도 원 저장소 URL 로 가져오게 해 주므로, 기획서대로 `Qntology/premium-discount` URL 을 썼고 cargo 가 실제로 받아 왔습니다. 그 브랜치가 병합되기 전에 지워지면 이 커밋을 받을 수 없게 될 수 있습니다.

## 검증

| 묶음 | 결과 |
| --- | --- |
| `cargo test -p zena-verdict -p zena-avatar-mcp -p zena-sds` (git 리비전으로) | 7 · 32 + 3 · 25 + 6 통과 (`zena-avatar-mcp` 단위는 34 → 32: MCP 틀 시험 2개가 `premium-discount-mcp-protocol` 로 감) |
| 아바타 카탈로그 실제 내려받기 (`-- --ignored`, 새 폴더) | 4개 모델 통과, `.part` 안 남음 |
| `cargo check --workspace --all-targets --exclude onnxruntime-sys --exclude onnxruntime` | 통과 (두 crate 의 시험 대상은 이전부터 실패) |
| `cargo test -p raon-zena --no-default-features` | 17 통과 |
| premium-discount 쪽 | crates 전부, 실제 앱 E2E, 앱 237, UI 70 (README 22절) |
