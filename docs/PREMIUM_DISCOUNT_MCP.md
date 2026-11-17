# premium-discount-mcp (옮겨 감)

`premium-discount-mcp` 는 2026-11-17 premium-discount 저장소의 `crates/premium-discount-mcp` 로 옮겼습니다. 설계 · 도구 · 시험 · 기획서 대조 기록은 그 crate 의 `README.md` 에 있습니다.

raon-zena 쪽에 남은 것:

- **링크**: 작업 공간 `Cargo.toml` 의 `[workspace.dependencies]` 에 `premium-discount-*` crate 9개를 git 리비전으로 고정합니다. 옆에 둔 premium-discount 체크아웃으로 빌드하려면 같은 파일 끝의 `[patch]` 주석을 풉니다.
- **로컬 판정** `crates/zena-verdict`: `premium_discount_verdict::*` 를 다시 내보내고, 대화 피드백 → 관측(`feedback`)과 제나의 말하는 방식(`speak`)만 둡니다. 계약 사본(`contracts/verdict_v1.json`)으로 고정 리비전을 다시 시험합니다.
- **MCP 틀** `zena_avatar_mcp::protocol`: `premium-discount-mcp-protocol` 을 다시 내보냅니다(Track A · B 코드는 그대로).
- **서버 실행 파일**: raon-zena 작업 공간은 `premium-discount-mcp` 바이너리를 만들지 않습니다. `premium_discount_mcp::find_mcp_server()` 가 `$PD_MCP` → raon-zena 실행 파일 옆 → premium.discount 데스크톱 설치 폴더(사이드카) 순서로 찾습니다. 직접 만들려면 `cargo install --git https://github.com/Qntology/premium-discount premium-discount-mcp` 입니다.
