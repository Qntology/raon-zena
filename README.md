# RAON ZENA

**KRAFTON Raon Zena(크래프톤 라온 제나)** 는 게임 화면을 실시간으로 보고 시선·표정·짧은 코칭 대사로 반응하는 인게임 AI 버튜버 컴패니언입니다. 이 저장소는 Master PRD v3.0을 바탕으로 만든 Rust 워크스페이스입니다.

> 상태: **기반 구현 단계**입니다. 다음은 실제 가중치나 테스트로 검증했습니다.
> - 양자화기: 원본 KRAFTON 가중치를 실제로 양자화했습니다.
> - **Raon-1-nano 추론 엔진(candle)**: 공식 PyTorch 구현과 로짓이 일치(KL 2.6e-11, top-1 100%)하고, 실제 모델이 Zena 툴콜을 생성합니다.
> - LLM 가속은 `logis-center`와 같은 `candle-rocm-vulkan` 포크로 붙였습니다(CUDA / ROCm / Vulkan / Metal). 단, 처음 개발한 환경에는 GPU가 없어 GPU 실행 속도는 측정하지 못했습니다(아래 Windows 클라이언트 항목 참고).
> - 이벤트 센서, NLP-NMS 라우터, 툴콜·아바타 매핑
> - **단어 사전 기반 캐릭터 표현**: logis-center `bias.json` 형식의 표현 사전으로 사용자의 말과 PC 화면을 같은 표현 도메인으로 잡아내고, 사전에 적힌 표정·동작·대사로 반응합니다(LLM 없이). 사전은 **LanceDB**에 있으며 실행 중에 고치거나 사용 중에 배우고, 되돌릴 수도 있습니다.
> - **주기적 화면 감지(Raon-VisionEncoder, candle)**: 공식 PyTorch 출력과 코사인 1.000000 일치. noto-sandbox의 체커·fovea·지뢰찾기 패스를 이식했고, 프레임은 저장하지 않습니다. LLM은 필요할 때만 올립니다. [`docs/EXPRESSION.md`](docs/EXPRESSION.md)
> - Windows 클라이언트: RTX 3050 Laptop + Windows 11에서 실제 오버레이 창을 띄워 스크립트 대화(사전 반응, 트랙 전환, 액세서리, 스트리밍 코칭, 작별 후 종료)를 확인했습니다. `--snapshot`으로 3패스 렌더 결과를 PNG로 확인했습니다. 이 머신에서 `--features cuda` 빌드와 GPU 커널(밀집·Q8_0 matmul)이 CPU 결과와 일치하는 것을 확인했습니다. 단, 가중치가 없어 Raon-1-nano의 GPU 생성 속도는 아직 측정하지 못했습니다.
> - **캐릭터 챗·전용 이모티콘·투트랙 헤드리스 MCP**: 멀티턴 기억, 스트리밍 툴콜(표정 선행), 능동적 말걸기, 말풍선·채팅창, 14종 캐릭터 전용 이모티콘(SD 스탬프, 3D 프롭, 만화 표정), SolidCraft/서브컬처 MCP 에셋 파이프라인. [`docs/CHARACTER.md`](docs/CHARACTER.md)
>
> TTS는 아직 스텁입니다. 자세한 내용은 [현재 범위와 한계](#현재-범위와-한계)를 보세요.

## 구성

| 경로 | PRD 단계 | 내용 |
|---|---|---|
| `crates/zena-core` | §2.1 · §2.4 · §4 · §5 | 화면 중앙 기준 극좌표 `(r, θ)` 변환과 Inward/Outward 흐름 판정, Lv1–Lv3 반응 라우팅, 툴콜 스키마와 Raon 네이티브 `<tool_call>` 파서, Live2D/VRM 파라미터 매핑(한글 중성 기반 비셈 립싱크), Win32 스타일 계산, 안전 종료 시퀀스, FPS 스로틀 |
| `crates/zena-sensor` | §2.1 · §2.2 | 로그 휘도 차분 **WGSL 컴퓨트 셰이더**(naga로 검증)와 동일한 알고리즘의 CPU 참조 구현, 타일 ROI 클러스터링, 트래커, NaFlex 패치 추출 |
| `crates/nlp-nms-router` | §2.3 | `Qntology/logis-center`에서 이식한 기능: 조사 박리, **사전 exact·교착어 접두 일치 → bias/prejudice 점수**, 스팬 NMS, μ+3σ 적응 중복 임계, snap/twin 박스 NMS, 배타 할당. 여기에 시간축 의미 NMS와 공간-텍스트 NMS를 더했습니다. 의도 표는 코드가 아니라 단어 사전에서 옵니다. **NLP-NMS v2**: Stanza 역할 분석(내용어 · 보조 용언 · 부정과 그 범위), granite 임베딩 + 해시 혼합 점수, 이중 센터링 후 마진 게이트 1:1 배정, SDS 적응 마진. 부정된 명령("앱 끄지 마")은 실행하지 않습니다. [`docs/plans/relative-value-sds.md`](docs/plans/relative-value-sds.md) §11 |
| `crates/raon-quant` | §3 | 원본 safetensors를 mmap으로 읽어 **단일 패스 4계층 피라미드 양자화**를 하고 GGUF v3로 씁니다. 그 밖에 imatrix 반영, `--preset zena`(측정 기반 품질 레시피), `kept_ids` 어휘 절삭, 융합 dequant-dot 커널, 자가 검증이 들어 있습니다. |
| `crates/zena-lexicon` | §2.3 · §4 | **표현 사전**: logis-center `bias.json` 노드 형식(`semantic`/`bias`/`prejudice`/`exact_match`)에 `express`(표정·동작·대사), `command`, `ask`, noto식 `vision` 노드를 더한 `data/zena_lexicon.json`. 기능 `lancedb`로 **LanceDB 저장소**(공식 Rust SDK): 리비전, 핫 리로드, n-gram BM25, 1152-d 앵커 벡터 검색, 학습 후보, Lance 타임트래블 롤백 |
| `crates/zena-nlp` | NLP-NMS v2 | **Stanza**(stanza-onnx-export format 2, 12개 언어: 토큰 → MWT → 품사 → 원형 → 의존 구문)와 언어 판정, **granite-embedding-97m-multilingual-r2**(ModernBERT, 384차원, CLS; HF 대비 코사인 ≥ 0.9999999). logis-center 에서 이식했습니다. |
| `crates/zena-sds` | NLP-NMS v2 | logis-center **Score Dynamics** 이식: 스코프 백오프 적응 기준선(Welford + 최근 링), 감쇠 형상, 혼동 사전, 전이 사전 확률, 이중 센터링과 적응 배타 배정, 원장 + 실행별 run 파일 |
| `crates/onnxruntime(-sys)` | — | logis-center 의 onnxruntime-rs 벤더 판. ONNX Runtime 을 **실행 시 로드**하므로 없으면 Stanza 만 빠지고 앱은 정상 실행됩니다. |
| `crates/zena-avatar-mcp` | 투트랙 에셋 | CAD/에디터 UI 없는 **헤드리스 MCP** 에셋 파이프라인. MCP(JSON-RPC 2.0, stdio) 클라이언트·서버, Track A **SolidCraft 브리지**(`solidcraft-cli mcp`의 `batch`/`export glb`)와 SolidCraft가 없을 때 같은 결과를 내는 내장 파라메트릭 생성기(3D 감정 프롭·코칭 화살표·액세서리·VRM 이름 모프를 가진 로봇 제나), Track B-1 **SD(치비) 스탬프 합성기**(레이어 PNG 키트 또는 내장 절차 키트), Track B-2 **VRM 0.x/1.0** 표정·휴머노이드 메타데이터, glTF 리더/라이터와 스키닝 수학, 백그라운드 `AssetService`. `zena-subculture-mcp` 바이너리가 Track B를 MCP 도구로 노출합니다. [`docs/CHARACTER.md`](docs/CHARACTER.md) |
| `crates/premium-discount-mcp` | premium.discount 동기화 · 추천/비추천 | **premium-discount-app**(LanceDB 원본 보관함 · 투표 보관함 · 판정 캐시 · embeddinggemma-2 768차원 · QR 로그인)을 자식 프로세스로 두고 그 줄 단위 명령(`serve_lines`)을 MCP 도구 8개(`pd_enqueue_node` · `pd_flush_and_sync` · `pd_search_merged` · `pd_vote` · `pd_verdicts` · `pd_graph_inspect` · `pd_auth_session` · `pd_model_manage`)로 감싼 헤드리스 서버. 스키마는 premium-discount 기준이다(필드 이름 그대로). `app://exe/scene` · URL → 서버가 받는 `cc`, 웹 클라이언트와 같은 `hashId`로 `to`/`bcc`, `GET /graph*`. 앱 쪽은 `PdSyncClient`와 `coalesce.ts`를 옮긴 `SyncLoop`(진행 중 요청은 한 번만 재실행 · flush OR · 남은 일이 있으면 10초 주기). [`docs/PREMIUM_DISCOUNT_MCP.md`](docs/PREMIUM_DISCOUNT_MCP.md) |
| `crates/zena-verdict` | 로컬 추천/비추천 | premium-discount 가 소유한 판정 규칙 **PD-Verdict v1**(축소 평균 → SDS 백오프 기준선 → tanh 상대 순위 → 히스테리시스 등급 + 절대 하한 → 5단계 grade · 이유 코드)의 Rust 구현. premium-discount `contracts/verdict_v1.json` 사본의 사례 19개를 서버 TypeScript · 파이썬 참조 구현과 같은 값으로 통과한다. 로컬 관측(결과 · 대화 피드백 · 수동 보정)에 돌리고 서버로 올리지 않는다. |
| `crates/raon-vision` | §2.2 | **Raon-VisionEncoder**(SigLIP2 So400m/16, NaFlex) 비전·텍스트 타워와 토크나이저를 candle로 구현. noto-sandbox의 sparse pass(`forward_sparse`/`extend_sparse`/`merge_sparse`), 체커·fovea 첫 패스, 지뢰찾기 확장, 사전 앵커 기반 제로샷 화면 도메인 점수 |
| `apps/zena-lexicon` | — | 사전 CLI: `import`·`export`·`add`·`remove`·`search`·`show`·`candidates`·`history`·`rollback`·`anchors`·`nearest`·`stats` |
| `crates/raon-infer` | §2.4 | **Raon-1-nano 순전파**(RDA 18층 + MLA 6층 + Raon-mHC + PLE)를 candle로 구현했습니다. `cuda`/`rocm`/`vulkan`/`metal`/`flash-attn` 기능 플래그와 `Qntology/candle-rocm-vulkan` 포크를 씁니다. 그 밖에 GGUF 블록 디바이스 상주(`QMatMul`), **safetensors 로드 시점 피라미드 양자화**(GGUF 불필요), 프리픽스 상태 캐시, `raon-chat`, `raon-imatrix`가 들어 있습니다. [`docs/INFERENCE.md`](docs/INFERENCE.md) |
| `apps/raon-zena` | §5 | Windows 오버레이(350×450 마이크로 뷰포트). 구성 요소: winit, 단일 wgpu DX12 디바이스(DirectComposition), DXGI Desktop Duplication, Win32 레이어 플래그와 전역 단축키. **3패스 렌더러**(본체: 절차적 얼굴 / Track A `.glb` / Track B-1 2D / Track B-2 `.vrm` 툰 셰이딩·모프·스키닝 → 캐릭터 전용 이모티콘·3D 프롭 → 말풍선), **캐릭터 챗**(스트리밍 툴콜로 표정 선행·문장 단위 발화, 멀티턴 기억, 세션 요약, 능동적 말걸기, 오버레이 채팅창+IME), `--snapshot` 헤드리스 렌더 검사 |
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

GGUF로 변환하지 않고, 원본 safetensors를 로드하면서 바로 피라미드 양자화할 수도 있습니다. 처음 한 번만 블록을 만들어 `runtime-<프리셋>-<해시>/`에 저장하고(4코어 약 33초), 다음부터는 1초대에 올라갑니다. `logis-center`와 같은 런타임 캐시 방식이며 품질은 같습니다.

```bash
./target/release/raon-chat --safetensors models/KRAFTON/Raon-1-nano --quantize zena --imatrix out/raon-1-nano.imatrix.dat --device auto --prompt "보스가 내려찍기 할 때 어떻게 피해?"
```

```bash
cargo test --workspace --release
```

Windows 클라이언트는 아래 명령으로 실행합니다. `--features`로 GPU 백엔드를 고르며, 실행 중 stdin에 문장을 입력하면 제나에게 말을 걸 수 있습니다.
LanceDB 사전 저장소를 빌드하려면 `protoc`가 필요합니다(Ubuntu `apt install protobuf-compiler libprotobuf-dev`, Windows `winget install protobuf`).
`--no-default-features`로 빌드하면 DB 없이 JSON 사전만 씁니다.

Windows에서 CUDA 13.x로 `--features cuda`를 빌드하려면 VS 개발자 셸(`vcvars64.bat`, nvcc가 `cl.exe`를 찾도록)에서 다음 환경 변수를 먼저 설정해야 합니다. CUDA 13의 CCCL 헤더가 MSVC의 전통 전처리기를 거부하기 때문입니다. RTX 3050 Laptop + CUDA 13.3 + MSVC 14.51에서 이 설정으로 `raon-chat`이 빌드되는 것을 확인했습니다.

```bash
set NVCC_APPEND_FLAGS=-Xcompiler=/Zc:preprocessor
```

```bash
cargo run --release -p raon-zena --features cuda,flash-attn -- --model models/KRAFTON/Raon-1-nano --quant-preset zena --llm-device auto --llm-idle 120 --screen models/KRAFTON/Raon-VisionEncoder --learn
```

첫 실행 때 사용자별 LanceDB 사전(`%LOCALAPPDATA%\RaonZena\lexicon.lance`)이 `data/zena_lexicon.json`으로 만들어집니다. 화면 도메인을 쓰려면 앵커를 한 번 계산해야 합니다. 이 단계에서만 텍스트 타워(0.7B)를 올립니다.

```bash
cargo run --release -p zena-lexicon-cli -- anchors --model models/KRAFTON/Raon-VisionEncoder --tokenizer models/timm/ViT-SO400M-16-SigLIP2-256/tokenizer.json
```

실행 중에도 사전을 고칠 수 있습니다. Zena가 3초 안에 다시 읽습니다.

```bash
cargo run --release -p zena-lexicon-cli -- add emotion.celebrate bias "드디어 깼다"
```

단축키:
- `Ctrl+Shift+Q`: 작별 인사 후 종료
- `Ctrl+Shift+H`: 숨기기 / 보이기
- `Ctrl+Shift+L`: 클릭 투과 모드와 드래그 모드 전환
- `Ctrl+Shift+T`: 오버레이 채팅창 열기 (한글 IME 지원, Enter 전송, Esc 닫기)

아바타 트랙과 액세서리는 말로 바꿀 수 있습니다("로봇 모드로 변신해 줘", "안경이랑 헤드셋 써 봐", "원래대로"). 시작 트랙은 `--track procedural|solidcraft|subculture-2d|subculture-3d`(3D는 `--vrm 파일.vrm` 필요), 2D 파츠 키트는 `--chibi-kit 폴더`로 지정합니다. SolidCraft(`solidcraft-cli`)가 PATH나 `C:\Program Files\SolidCraft`에 있으면 Track A 에셋을 그 MCP 서버로 만들고, 없으면 내장 생성기로 만듭니다. 침묵 후 먼저 말 걸기는 `--proactive 초`(0이면 끔), 대화 기억 길이는 `--history N`입니다.

렌더러를 창 없이 확인하려면 다음 명령으로 장면 몇 개를 PNG로 그립니다.

```bash
cargo run --release -p raon-zena -- --snapshot out/snapshots
```

NLP-NMS v2(기본 `--nlp auto`)는 granite 모델(`%LOCALAPPDATA%\RaonZena\models\granite-embedding-97m-multilingual-r2`)과 Stanza format 2 모델(`%LOCALAPPDATA%\RaonZena\models\stanza\ko`, `huggingface.co/Qntology/stanza-ko` 의 `format2` 폴더)이 있으면 씁니다. logis-center 가 이미 받은 폴더도 찾습니다. ONNX Runtime 은 실행 파일 옆, `ORT_DYLIB_PATH`, 또는 개발 빌드가 받아 둔 공식 1.28.0 에서 실행 시 불러옵니다. 오프라인 빌드는 `ORT_STRATEGY=skip` 을 쓰세요. 평가표는 다음 명령으로 봅니다.

```bash
cargo test --release -p nlp-nms-router --features stanza --test eval_ko -- --nocapture
```

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
5. **NLP-NMS 텍스트 모델은 Raon이 아닙니다.** NLP-NMS v2는 logis-center처럼 Stanza(형태·구문)와 granite-embedding-97m-multilingual-r2(임베딩)를 씁니다(둘 다 Apache-2.0). 모델이 없으면 의존성 없는 해시 n-gram 임베더와 규칙 분석기로 돌아갑니다. 93문장 한국어 평가 세트에서 v2는 74/93(79.6%), 명령 오탐 1회이고, v1은 64/93, 명령 오탐 9회입니다. 다만 어휘가 겹치지 않는 바꿔 말하기는 아직 38개 중 21개만 맞습니다.
6. **추론 엔진은 구현했지만 GPU 실측이 남았습니다.**
   - Raon-1-nano는 llama.cpp가 지원하지 않는 커스텀 아키텍처라 candle로 직접 구현했고, 공식 구현과 일치합니다.
   - CPU(이 환경, 4코어)에서는 2–2.7 tok/s로 메모리 대역폭에 묶입니다.
   - CUDA/ROCm/Vulkan 경로는 컴파일과 타입 체크만 했고, 실기 속도와 VRAM은 측정해야 합니다.
   - RDA 프리필은 호스트 스캔입니다. 포크의 fused GDN은 RDA 수식(키 채널별 감쇠, 쓰기 게이트)과 맞지 않습니다.
   - MTP 추측 디코딩 루프는 아직 없습니다.
   - 음성 입력(ASR) 모델도 스택에 없습니다.
7. **KPI는 검증하지 않았습니다.** 15 ms / 45 ms 지연과 RAM 25 MB는 Windows와 GPU 실기에서 측정해야 합니다. 이 환경(Linux 컨테이너, GPU 없음)에서는 Windows 타깃 `cargo check`까지만 했습니다.
8. **화면 감지는 CPU에서 느립니다.** 이 4코어 환경에서 비전 타워 한 번(F32)이 256패치 약 4.7초, 144패치 약 2.8초입니다. GPU에서는 전체 패스를 씁니다(실측은 남음). 화면 도메인 점수는 PIL로 그린 합성 화면 4종으로만 확인했습니다. 실제 게임·차트 화면에서 임계값(`min_score`)을 조정해야 합니다.

## 라이선스

이 저장소는 Apache-2.0입니다. 이식한 코드의 출처와 라이선스는 [`NOTICE`](NOTICE)에 있습니다. 모델 가중치는 저장소에 포함하지 않으며 각 모델의 라이선스를 따릅니다.
