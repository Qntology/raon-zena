# 캐릭터 챗 · 캐릭터 전용 이모티콘 · 투트랙 헤드리스 MCP

`raon_zena.md` 마스터 기획안의 세 과제(캐릭터 챗 고도화, 캐릭터 전용 2D/3D 이모티콘, CAD/에디터 UI 없는 투트랙 MCP 에셋 파이프라인)를 이 저장소에 구현한 내용입니다. 기획안의 Phase 1–4가 모두 들어 있고, 검증하지 못한 부분은 [한계](#한계)에 따로 적었습니다.

## 1. 캐릭터 챗 (Phase 4)

### 멀티턴 기억과 세션 요약 — `zena-core::prompt`, `zena-core::session`

- `Dialogue`: 최근 N턴(기본 4, `--history N`)의 `플레이어: … / 제나: …` 슬라이딩 윈도우입니다. 한 턴은 60자로 자릅니다.
- `SessionContext`: 화면 장면과 지속 시간, 연속 패배·승리, 친밀도(칭찬·감사·대화로 상승), 긴장도(전투·패배로 상승, 반감기 20초)를 한 줄로 요약하고 말투 힌트(`[말투] 계속 지고 있다. 다그치지 말고 먼저 위로…`)를 붙입니다.
- 둘 다 **user 턴**에 직렬화합니다. `RaonNanoBackend`는 시스템 프롬프트(페르소나 + `<tools>`) 상태를 한 번 프리필해 매 요청 복제하므로, 시스템 프리픽스가 바이트 단위로 같아야 캐시가 맞습니다. `prompt::tests::history_and_session_go_into_the_user_turn_so_the_cached_prefix_never_changes`가 이를 검사합니다.
- 파이프라인은 LLM 스레드가 돌려주는 `Reply`(질문, 실제로 말한 문장)와 사전 Lv2 대사를 대화 기억에 넣습니다. 화면 이벤트에 대한 자동 코칭("근접 위협 조심해!")은 대화가 아니므로 기억하지 않습니다. 실행해 보니 이런 경고가 4턴 창을 밀어내 실제 대화가 빠졌기 때문입니다.

### 스트리밍 `<tool_call>` 파싱 — `zena-core::stream`

`ToolCallStream`은 생성 중인 토큰 델타를 받아 다음 이벤트를 즉시 냅니다.

| 시점 | 이벤트 | 오버레이 동작 |
|---|---|---|
| `emotion_state` / `body_motion` / `character_emote`가 닫힘 (시선만 있으면 감정이 올 때까지 대기) | `Expression` | 표정·시선·동작·이모티콘을 먼저 표시 (`AvatarCmd::Expression`) |
| `speech_text` 안에서 문장 끝(`.!?~…` + 공백, 20자 넘은 쉼표) | `Speech(문장)` | 비셈 타임라인 뒤에 이어 붙이고 말풍선에 추가 (`AvatarCmd::Speak`) |
| `manage_zena_system` / `manage_avatar_asset`의 `</tool_call>` | `Call` | 창 제어 / 에셋 전환 |

- 부분 태그(`</param`처럼 덜 도착한 태그)는 다음 델타까지 소비하지 않습니다. 토큰 한계로 잘리면 남은 문장을 `finish()`에서 흘려보냅니다.
- 문장 전체는 90자(`MAX_SPEECH_CHARS`)를 넘지 않고, 넘치는 뒤 문장은 통째로 버립니다.
- 툴콜 없는 평문 답은 끝에서 `NOD` 표정과 함께 말합니다.
- 스트림 결과는 일괄 파서 `parse_model_output`와 같은 호출을 냅니다(`the_stream_agrees_with_the_batch_parser`).
- `RaonNanoBackend`는 `Generator::generate_from`의 텍스트 델타를 그대로 넘기고, 로그에 첫 텍스트까지의 시간(`first text … ms`)을 찍습니다.

### 능동적 말걸기 — `ProactiveTimer`

사용자 발화, 제나의 발화, LLM 답이 모두 없는 상태가 `--proactive`초(기본 90초) 동안 이어지면 `proactive.<장면>` 사전 대사를 말합니다(LLM 없음). 장면은 화면 감지가 링크한 `react.*` 노드 이름입니다(`react.coding` → `proactive.coding`). 같은 장면이 50분 넘게 이어지면 한 번 `proactive.stretch`(스트레칭 권유)를 말합니다. `privacy` 화면을 감지한 동안에는 말하지 않습니다. 대사는 `data/zena_lexicon.json`의 `proactive` 그룹에 있고, 문구가 없어 사용자 발화로는 잡히지 않습니다.

### 오버레이 말풍선과 채팅창

- **말풍선**(Pass 3): 대사를 시스템 한글 폰트(맑은 고딕; `--font`로 변경)로 CPU에서 줄바꿈·래스터화하고 텍스처 하나로 올립니다. 꼬리가 얼굴을 향하고, 스트리밍 문장은 같은 말풍선에 이어 붙습니다. 말이 끝나고 2.5초 뒤 사라집니다.
- **채팅창**: `Ctrl+Shift+T`로 엽니다. 창이 `WS_EX_NOACTIVATE`/`WS_EX_TRANSPARENT`를 잠시 풀고 포커스와 IME를 받습니다(`WindowState::typing`). 조합 중인 글자(preedit)는 밑줄로 표시합니다. Enter는 전송 후 닫고 게임으로 포커스를 돌려줍니다. Esc를 누르거나 포커스를 잃으면 닫힙니다. 전송한 문장은 stdin과 같은 발화 채널로 들어갑니다.
- 기존 버그도 고쳤습니다. 화면이 멈춰 DXGI가 새 프레임을 주지 않으면(`AcquireNextFrame` 타임아웃) 루프가 발화를 처리하지 않았습니다. 이제 센서만 건너뜁니다(`chat_works_on_a_static_screen_and_remembers_the_previous_exchange`).

## 2. 캐릭터 전용 이모티콘 (Phase 1)

유니코드 이모지 대신 제나 자신의 표현을 `CharacterEmote` 14종으로 정의했습니다(`zena-core::tools`). 그리는 방식(`EmoteForm`)은 세 가지입니다.

| 형태 | 이모티콘 | 그리는 방법 |
|---|---|---|
| `Prop` (3D 입체 파츠, Track A) | `COMIC_ANGER_3D` 💢, `SPARKLE_IDEA` 💡, `EXCLAIM_3D` ❗, `THREAT_ARROW_3D` | 머리 옆 3D 메쉬. 스프링 팝인과 흔들림이 있고, 코칭 화살표는 시선 방향(위협 위치)을 가리키며 15° 단위로 생성합니다. |
| `Morph` (만화적 표정, Track B-2) | `STAR_EYES`, `HEART_EYES`, `DIZZY_SPIRAL`, `TEARS` | VRM에 해당 커스텀 표정이 있으면 블렌드셰이프로, 없으면 스탬프로 표시합니다. |
| `Stamp` (SD 치비 스탬프, Track B-1) | `CHIBI_CHEER`, `CHIBI_HEART`, `SWEAT_PANIC`, `QUESTION`, `RELIEF_SIGH`, `MUSIC_NOTE` | 제나 파츠(보라 머리·보라 눈·의상)로 합성한 8프레임 스트립 |

- 모든 이모티콘에 SD 스탬프가 있어, 3D 프롭이 아직 없을 때의 대체로도 씁니다.
- `ZenaAvatarArgs.character_emote`는 LLM 툴콜과 사전 `express.emote` 양쪽에서 지정할 수 있습니다. 지정하지 않으면 `CharacterEmote::for_expression(감정, 동작)`이 고릅니다(예: `ANGRY` → 💢, `SURPRISED_URGENT`+`LEAN_FORWARD_POINT` → 코칭 화살표).
- 모르는 이름은 호출 전체를 버리지 않고 이모티콘만 버립니다.

## 3. 투트랙 헤드리스 MCP (Phase 2–3)

```text
Overlay ─ Scene.missing() ─▶ AssetService (zena-avatar-mcp, 백그라운드 스레드)
                                │  Track A  ─▶ SolidCraft MCP (solidcraft-cli mcp --allow-path CACHE)
                                │              new_design → batch{commands} → export{format:"glb"}
                                │              (없거나 실패하면 내장 forge)
                                │  Track B  ─▶ zena-subculture-mcp (같은 빌드의 바이너리, stdio MCP)
                                │              compose_chibi_emote / forge_prop / robot_body / vrm_expressions
                                │              (없으면 같은 코드를 프로세스 안에서)
                                └─ 파일 캐시(%LOCALAPPDATA%\RaonZena\assets) → 서비스 스레드에서 디코드 → UI 스레드 업로드
```

- **트랙 전환과 장착**: `manage_avatar_asset` 툴(`SWITCH_TRACK`/`EQUIP_PART`/`REMOVE_PART`/`RESET_PARTS`)이나 사전의 `avatar` 그룹으로 바꿉니다. "로봇 모드로 변신해 줘" → `SOLIDCRAFT`, "안경이랑 헤드셋 써 봐" → 안경+헤드셋, "안경 벗어" → 정리. 같은 문장에 착용과 해제가 함께 있으면 해제를 따릅니다.
- 명령(시스템·에셋)을 낸 문장은 벡터로만 잡힌 질문을 함께 내지 않습니다. 실행 중에 "써 봐"의 "봐"가 `screen.what_is_this`에 벡터 0.81로 잡혀 LLM 질문이 한 번 더 나가는 문제를 발견했습니다. 그래서 라우터 규칙을 추가하고, "봐", "봐 줘", "해 봐"를 `function.request`(요청 어미) 뱅크에 넣었습니다.
- 서비스는 캐시 경로를 절대 경로로 바꿔 MCP 서버에 넘깁니다. 상대 경로를 넘기면 서버가 자기 허용 폴더 기준으로 다시 풀어 `out\assets\out\assets`처럼 중복된 경로가 생겼습니다.
- **Track A**: 내장 forge가 로봇 제나 본체를 만듭니다. 둥근 상자 머리, 바이저, 빛나는 눈, 안테나로 이뤄지고, VRM 1.0 이름의 모프(`blink`, `happy`, `angry`, `sad`, `surprised`, `relaxed`, `aa`…`oh`)가 있습니다. 감정·립싱크·눈 깜빡임이 VRM과 같은 경로로 동작하고, 시선은 모델 전체의 yaw/pitch로 따라갑니다. 액세서리(헤드셋·안경·고양이 귀·리본)와 3D 이모티콘 프롭도 같은 forge가 만듭니다.
- **Track B-1**: `--chibi-kit 폴더`의 `kit.json`(슬롯별 PNG 레이어와 오프셋, 이모티콘별 눈·입·손·효과 조합)으로 스탬프와 2D 본체를 합성합니다. 키트가 없으면 내장 절차 키트를 씁니다.
- **Track B-2**: `--vrm 파일.vrm`. VRM 0.x `blendShapeMaster`와 1.0 `VRMC_vrm.expressions`(프리셋과 커스텀)를 읽어, 감정(`happy`/`joy`…), 비셈(`aa`/`a`…), 깜빡임, 만화 표정을 모프 가중치로 바꿉니다. 휴머노이드 `neck`/`head` 본을 시선 방향으로 돌려 GPU 스키닝합니다. VRM 0.x는 -Z를 바라보므로 180° 돌립니다.

## 4. 3패스 렌더러 — `apps/raon-zena/src/renderers`

모두 하나의 wgpu 디바이스에서 그리고, 깊이 버퍼(Depth32Float)를 공유합니다.

1. **본체**: 트랙별로 `procedural.rs`(기존 셰이더), `mesh.rs`("MToon-lite": 2단 램프, 림 라이트, 이미시브, 알파 마스크/블렌드, CPU 모프 블렌딩, 스토리지 버퍼 조인트로 GPU 스키닝), `sprite.rs`(2D 본체) 중 하나로 그립니다. 같은 패스에서 머리 프레임에 액세서리를 얹습니다.
2. **이모티콘·프롭**: 깊이를 비우고 3D 프롭이나 SD 스탬프(프레임 스트립)를 그립니다.
3. **말풍선**: 말풍선과 채팅창 스프라이트를 그립니다.

본체 에셋이 오기 전에는 절차적 얼굴이, 프롭이 오기 전에는 스탬프가 대신 나옵니다. `--snapshot DIR`은 창 없이 오프스크린 타깃에 장면 다섯 개를 그려 PNG로 저장합니다. 실제 `AssetService`를 거치므로 에셋 경로까지 함께 검사합니다.

## 검증 (Windows 11, RTX 3050 Laptop 4 GB + AMD Radeon iGPU, CUDA 13.3)

- **단위·통합 테스트**: `zena-core` 32, `zena-lexicon`(LanceDB 저장소 포함) 8, `nlp-nms-router` 21, `zena-avatar-mcp` 32(MCP 서버 왕복 포함), `raon-zena` 17개가 통과합니다. 스트리밍 순서(표정이 대사보다 먼저 나옴), 프리픽스 캐시 불변, 멈춘 화면에서의 채팅, 침묵 후 능동 대사, 말로 트랙 전환, 인프로세스 에셋 서비스를 검사합니다.
- **헤드리스 렌더**: `raon-zena --snapshot out/snapshots`로 다섯 장면을 같은 3패스 경로로 그렸습니다. DX12에서는 RTX 3050, Vulkan에서는 AMD iGPU로 그렸고 결과는 같았습니다. Track B 스탬프는 `zena-subculture-mcp` 서버를 거쳤고, Track A는 SolidCraft가 없어 내장 forge로 만들었습니다.
- **실제 오버레이 창**: `--synthetic`으로 띄워 stdin에 대본을 넣었습니다. 인사(사전 Lv2), "로봇 모드로 변신해 줘"(트랙 전환과 로봇 본체 로드), "안경이랑 헤드셋 써 봐"(장착), 코칭 질문(문장 단위 스트리밍 발화), "종료해 줘"(작별 인사 후 정상 종료)를 차례로 확인했습니다.
- **CUDA**: `raon-infer --features cuda`가 이 머신에서 빌드됩니다. 이때 `NVCC_APPEND_FLAGS=-Xcompiler=/Zc:preprocessor`가 필요합니다(README 참고). 새 장치 테스트에서 GPU의 밀집 matmul+softmax와 Q8_0 양자화 matmul이 CPU 결과와 일치했습니다.

## 한계

- **SolidCraft 값 형식은 미검증입니다.** 이 머신에 SolidCraft가 설치돼 있지 않아 실제 서버와 통신해 보지 못했습니다. 문서에 있는 도구와 명령(`new_design`, `batch`, `export`, `sketch.*`, `solid.*`, `appearance.assign`)만 쓰지만, 단위·벡터·평면 이름 형식은 문서에 없어 추정했습니다. 연결할 때 `list_commands`로 없는 명령을 기록하고, 실패하면 내장 forge로 넘어갑니다. 템플릿 JSON으로 명령을 덮어쓸 수도 있습니다.
- **inox2d(Inochi2D)는 연결하지 않았습니다.** Track B-1은 레이어 PNG 합성으로 구현했습니다. `to_live2d()` 파라미터를 inox2d 퍼펫에 연결하는 일은 남아 있습니다.
- **VRM 렌더러는 실제 VRM 파일로 확인하지 못했습니다.** 저장소에 VRM이 없어서, 단위 테스트는 합성한 VRM 0.x/1.0 GLB로만 했습니다. 스프링 본(머리카락 물리)과 MToon의 외곽선·쉐이드 텍스처는 아직 없습니다.
- **TTS는 여전히 스텁입니다.** `Speak` 문장은 로그와 말풍선, 비셈 타임라인으로만 표현합니다.
- Raon-1-nano 가중치가 이 머신에 없어 스트리밍 체감 지연(첫 표정까지의 시간)은 템플릿 백엔드로만 확인했습니다.
