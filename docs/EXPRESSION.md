# 단어 사전으로 움직이는 캐릭터 표현

Zena의 표현은 이제 **단어 사전**이 결정합니다. 사전은 `Qntology/logis-center`의 `bias.json`과 같은 노드 형식
(`semantic`, `bias`, `prejudice`, `exact_match`)을 그대로 씁니다. 각 노드는 캐릭터가 어떻게 반응할지를 함께 적습니다.
사용자의 말과 PC 화면을 같은 사전의 **표현 도메인**으로 잡아내고, 사전에 적힌 표정·동작·대사로 반응합니다.
LLM(Raon-1-nano)은 사전이 `ask`로 넘긴 열린 질문에만 쓰고, 그때만 메모리에 올립니다.

```text
사용자 발화 ──► 정규화(호출어·조사 박리) ──► 스팬 NMS ─┬─ exact / 접두 일치 (벡터보다 먼저)        ┐
                                                      └─ bias·prejudice 코사인 (logis-center 점수) ┘
                                                           │ 의미 NMS(시간축 중복 제거)
                                                           ▼
                         ┌──────────── 표현 도메인 (사전 노드) ────────────┐
                         │ express → 표정·동작·대사 (LLM 없음, Lv1/Lv2)   │
                         │ command → manage_zena_system                    │
                         │ ask     → Raon-1-nano (Lv3, 필요할 때만 로드)  │
                         └──────────────────────▲──────────────────────────┘
PC 화면 ── 3초마다 1장 ── 변화 없으면 건너뜀 ── Raon-VE 비전 타워 ── 앵커 점수 ──┘ (vision 노드 → link)
                          (메모리에서만, 저장 안 함)   (GPU 전체 / CPU fovea)  (LanceDB에 저장된 SigLIP2 앵커)
```

## 1. 사전 형식 — `data/zena_lexicon.json`

| 그룹 | 쓰임 | 예 |
|---|---|---|
| `function` | logis-center의 기능어 뱅크(요청 어미·군말). 이 단어들로만 된 스팬은 도메인이 되지 못합니다. | `해 줘, 좀, 진짜, please` |
| `system` | `command`(+`window_layer_mode`) → 시스템 명령. `express`가 있으면 명령 전에 먼저 연기합니다(`show_avatar`의 "짠, 나 왔어!"). 종료는 오버레이의 작별 시퀀스가 맡습니다. | `exit_app`, `ghost_mode` |
| `emotion`, `market` | `express` → 사전에 적힌 표정·동작·대사로 바로 반응(Lv2, LLM 없음) | `celebrate`, `frustrated`, `stock_down` |
| `coach`, `screen`, `chitchat` | `ask` → Raon-1-nano에 넘김. 같이 적힌 `express`(생각하는 자세)를 먼저 보여 줍니다. | `game_tip`, `what_is_this` |
| `react` | 화면 전용 표현(텍스트로는 잡히지 않음) | `combat`, `market`, `privacy` |
| `vision_prompting` | SigLIP2 프롬프트 앙상블 템플릿과 배경 문구(noto-sandbox `world_lexicon.json` 방식) | `"a screenshot of {}."` |
| `vision` | 화면 도메인. 영어 프롬프트(bias/prejudice), `kind`(scene·event·privacy), `link`(연기할 표현 노드) | `stock_chart → react.market` |

```json
"celebrate": {
  "semantic": "The user won, cleared, succeeded or got a rare drop and is happy about it.",
  "bias": "이겼다, 이겼어, 클리어했다, 보스 잡았어, 득템했다, ..., we won, victory",
  "prejudice": "졌다, 실패했어, 또 죽었어, 망했다, 주식 올랐어",
  "exact_match": ["이겼다", "이겼어", "클리어", "해냈다", "득템", "레벨업", "대박", "we won"],
  "express": { "emotion": "EXCITED", "motion": "CHEER", "lines": ["해냈다! 최고야!", "우와, 클리어 축하해!", "나이스! 완벽했어!"] }
}
```

검증 규칙(`Lexicon::validate`)은 다음과 같습니다.
- 두 노드가 같은 `exact_match` 단어를 가질 수 없습니다(공백 무시).
- `vision.link`는 `express`가 있는 노드를 가리켜야 합니다.
- 감정·동작·명령 이름은 `zena-core` 열거형에 있는 이름만 받습니다(오타가 NEUTRAL로 바뀌지 않게).
- 대사가 있는 표현은 Lv2, 없는 표현은 Lv1입니다.

## 2. 자연어 → 표현 도메인 (NLP · NMS)

`crates/nlp-nms-router`의 하드코딩 의도 표(`DEFAULT_BANK`)를 지우고, 사전의 텍스트 노드 전부를 도메인 뱅크로 씁니다(`domain.rs`).

1. **벡터보다 먼저 exact match.** logis-center `exact_match_filter_key` 방식입니다. 스팬이 `exact_match` 단어와 같으면
   0.95점입니다. 비교는 표면형, 조사 뗀 형태(lemma), 요청 어미 뗀 형태(stem) 중 하나로 하고 공백은 무시합니다.
2. **교착어 접두 일치.** logis-center `prefix_match_filter_stem` 방식입니다. 한글 단어가 사전 단어로 시작하고 남는
   음절이 두 개 이하이면 0.90점입니다(`클리어했다`→`클리어`, `떡락했어`→`떡락`). 남는 음절이 셋 이상이면 다른
   단어로 봅니다(`하이라이트`는 `하이`가 아님). 영어에는 적용하지 않습니다.
3. **bias / prejudice 벡터 점수.** logis-center 필드 점수와 같습니다. `bias`는 bias 문구 코사인의 최댓값×가중치,
   `prej`는 prejudice 문구 코사인의 최댓값입니다. 점수는 `max(bias − prej·w, 0)`이고, w는 2단어 이하일 때 0.3,
   그보다 길면 0.7입니다.
4. **스팬 NMS**: 점수가 높고 긴 스팬부터 받고, 단어가 겹치는 스팬은 버립니다. 그래서 한 발화에서 여러 도메인이
   나올 수 있습니다(`맨 앞으로 나와서 보스 패턴 어떻게 피해` → `system.topmost_on` + `coach.game_tip`).
5. **시간축 의미 NMS**: 같은 말을 연달아 해도 한 번만 반응합니다(`고마워`를 두 번 → 한 번).

`--learn`을 켜면 사전 단어 없이 벡터로만 잡힌 스팬(점수 ≥ 0.75)을 세어 둡니다. 세 번 나오면 그 도메인의 bias 문구가
되고(source `learned`), 이 기록은 LanceDB에 남습니다.

## 3. 사전 DB — LanceDB

사전은 LanceDB(공식 Rust SDK `lancedb` 0.41, Lance 14 / Arrow 58)에 저장합니다(`crates/zena-lexicon/src/store.rs`, 기능 `lancedb`).

| 테이블 | 열 | 쓰임 |
|---|---|---|
| `meta` | revision, meta, prompting, anchor_model, anchor_signature, tables, note | 논리 리비전. 리비전마다 각 테이블의 Lance 버전을 기록합니다. |
| `domains` | key, group, name, record | 노드의 bias.json 원형(JSON). 진실의 원천입니다. |
| `phrases` | domain, role, text, weight, source | 모든 bias / prejudice / exact 문구와 화면 프롬프트. `text`에 BM25(2~3글자 n-gram) 인덱스가 있습니다. |
| `vision` | key, name, kind, link, record | 화면 도메인 |
| `anchors` | text, model, signature, vector[1152] | 프롬프트 앙상블 SigLIP2 앵커. 코사인 벡터 검색을 합니다. |
| `candidates` | domain, text, hits, last_ms | 사용 중에 배우고 있는 문구 |

- **갱신**: `zena-lexicon add emotion.celebrate bias "개꿀"` 같은 편집은 해당 도메인의 레코드와 문구 행, FTS 인덱스를 다시 쓰고
  리비전을 올립니다. 편집 결과가 검증에 실패하면(예: 다른 노드의 exact 단어) 아무것도 쓰지 않습니다.
- **핫 리로드**: 실행 중인 Zena는 3초마다 리비전을 보고, 바뀌면 라우터와 화면 감시기에 새 사전을 넘깁니다.
- **재시드**: `import`로 JSON을 다시 넣어도 손으로 넣은 문구(`user`)와 배운 문구(`learned`)는 해당 도메인에 다시 합칩니다.
- **되돌리기**: Lance는 모든 버전을 보관합니다. `zena-lexicon history`로 리비전 목록을 보고, `rollback N`으로 그때의
  각 테이블 버전을 체크아웃해 최신으로 복원합니다(학습이 잘못 들어갔을 때).
- **정리**: Lance는 편집할 때마다 새 파일과 버전을 남깁니다. `zena-lexicon compact`는 작은 파일을 합치고,
  `--keep-days N`을 주면 N일보다 오래된 버전을 지웁니다(그 리비전으로는 더 이상 되돌릴 수 없습니다).
- **검색**: 한국어는 조사가 붙어 단어 단위 토큰으로는 `물약을`에서 `물약`을 못 찾습니다. 그래서 FTS는 n-gram
  토크나이저를 씁니다. 오타(`candlestik`)도 찾습니다.
- **앵커**: `zena-lexicon anchors --model <Raon-VisionEncoder> --tokenizer <tokenizer.json>`이 텍스트 타워(0.7B)를 한 번 올려
  앵커가 없는 프롬프트만 계산하고 저장합니다. 템플릿이 바뀌면 이전 앵커는 무효가 됩니다(signature). 실행 중에는
  비전 타워만 올라가고, 앵커는 DB에서 읽습니다.

> LatticeDB로 먼저 만들었다가 요청에 따라 LanceDB로 바꿨습니다. LatticeDB는 Rust 바인딩이 없어 FFI가 필요했지만,
> LanceDB는 공식 Rust SDK가 있어 동적 라이브러리 없이 워크스페이스에서 그대로 빌드됩니다. 대신 빌드에 `protoc`가
> 필요합니다(Lance의 protobuf 정의). Ubuntu는 `apt install protobuf-compiler libprotobuf-dev`, Windows는
> `winget install protobuf`로 설치합니다. `--no-default-features`로 빌드하면 DB 없이 JSON 사전만 씁니다.

## 4. 주기적 화면 감지 (개인정보를 남기지 않는 방식)

`apps/raon-zena/src/watcher.rs`가 화면을 봅니다. 이벤트 카메라처럼 계속 돌지 않습니다.

1. 파이프라인이 `--screen-every`(기본 3초)마다 축소된 프레임 1장(RGB)을 감시 스레드에 넘깁니다. 감시기가 아직 바쁘면 그 프레임은 버립니다.
2. 감시기는 32×18 휘도 썸네일만 보관합니다. 이전 판독 때와 평균 차이가 3/255 미만이면 인코더를 돌리지 않습니다.
3. 프레임은 메모리에만 있습니다. 디스크에 쓰지 않고 로그에도 남기지 않으며, 해제할 때 0으로 덮습니다(`ScreenFrame::drop`).
   패치도 인코딩이 끝나면 바로 버립니다. 감시 스레드 밖으로 나가는 것은 도메인 키, 점수, 시선을 둘 칸 하나뿐입니다.
4. `kind: privacy` 화면(로그인·비밀번호, 은행, 결제, 개인 메신저)을 보면 아무 말도 하지 않고 시선만 피합니다
   (`react.privacy`: HIDE_CROUCH, Lv1). 그 뒤 60초 동안 화면을 보지 않습니다.
5. scene은 바뀔 때만, 그리고 같은 장면은 10분에 한 번만 반응합니다. event(승리·게임오버·로딩)는 20초 쿨다운입니다.

## 5. SigLIP2 최적화 — noto-sandbox의 체커 패턴과 지뢰찾기

`crates/raon-vision`은 Raon-VisionEncoder(LocCa ViT-SO400M-16-SigLIP2, NaFlex)를 candle로 구현합니다.
이식한 noto-sandbox 구성 요소는 다음과 같습니다.

- **sparse pass** (`noto-ml` `VisionTower::forward_sparse / extend_sparse / merge_sparse`): 고른 칸만 인코더에 넣습니다.
  구멍은 검게 칠하는 것이 아니라 아예 빠지고, 각 칸은 자기 위치 임베딩을 씁니다. 나중에 여는 칸은 앞 패스가
  남긴 키/값에 어텐션합니다. 모든 칸을 순서대로 넣으면 전체 패스와 같습니다.
- **첫 패스 패턴** (`noto-vision` `perforate.rs`): 체커보드(절반), 또는 화면용 fovea(가운데 2/3는 촘촘하게, 가장자리는
  체커, 맨 위·아래 줄은 1/4).
- **지뢰찾기 확장**: 첫 패스에서 상위 도메인의 열지도(MAP 헤드 value 경로 = 칸별 특징 × 앵커)를 구합니다. 3×3 지역
  최대(≥ seed_min)를 씨앗으로 잡아 주변 8칸을 열고, 다음 라운드에서 grow_min 이상인 칸 둘레를 반경 2까지 엽니다.
  열리지 않은 구멍은 평가된 이웃의 최솟값으로 채워, 구멍에서 봉우리가 생기지 않게 합니다.

noto의 측정 결과를 그대로 따릅니다. 구멍이 있으면 남은 토큰의 특징도 바뀌므로 천공 패스는 전체 패스와 같지 않습니다.
CPU에서는 시간 절약이 크지만 GPU에서는 작습니다. 그래서 기본값은 **GPU에서는 전체 패스, CPU에서는 체커**입니다
(`--screen-pass auto`). 이 저장소에서 공식 PyTorch 출력과 비교한 결과는 아래에 있습니다.

### 공식 구현 대비 정확도 (`scripts/ve_reference.py` → `cargo run -p raon-vision --example ve_xcheck`)

PIL로 그린 480×270 합성 화면 4종(캔들 차트, 코드 편집기, 승리 배너, 그라디언트)을 256패치(12×21)와
144패치(9×16)로 넣어, 공식 `RaonVEModel`(timm, float32)의 출력과 비교했습니다.

| 항목 | 결과 |
|---|---|
| 토크나이저(SigLIP2 Gemma, EOS·패딩·canonicalize) | 5/5 문장이 공식 id와 같음 |
| 텍스트 타워 F32 | 코사인 ≥ 0.999999 |
| 비전 타워 F32 (공식 패치 입력) | 8/8 코사인 1.000000 |
| 우리 패치 추출(PIL bicubic 재현) | 7/8은 패치 차이 0, 1개(그라디언트 256)는 1 LSB → 코사인 0.99963 |
| 비전·텍스트 타워 Q8_0 | 코사인 ≥ 0.9989 / 0.9997. 단, candle CPU에서는 F32보다 4~5배 느림 |
| 공식 모델 제로샷 | 차트→"stock price candlestick chart" p=0.82, 승리→"victory screen" p=0.98 (logit scale 100, bias −13.3) |

### 체커 · fovea · 지뢰찾기 (CPU 4스레드, F32, 화면 8가지)

전체 패스 시간은 256패치 약 4.6~4.8초, 144패치 약 2.7~2.9초입니다. "코사인"은 전체 패스 임베딩과의 코사인이고,
top-1은 참조 문장 5개 중 1위가 전체 패스와 같은 화면의 수입니다.

| 패스 | 토큰 | 시간 | 코사인 (최저 – 최고) | top-1 |
|---|---|---|---|---|
| all (sparse 경로로 전체) | 100% | +0~8% | 1.0000 | 8/8 |
| checker | 50% | **−44~48%** | 0.62 – 0.99 (글자 많은 화면에서 크게 떨어짐) | 7/8 |
| fovea (화면용) | 68~71% | **−23~28%** | 0.92 – 0.997 | **8/8** |
| checker + 지뢰찾기 | 70~78% (씨앗 5~6, 40~57칸 열림) | −7~+22% | 0.65 – 0.99 | 8/8 |
| fovea + 지뢰찾기 | 74~82% (10~31칸 열림) | +1~+26% | 0.92 – 0.997 | 8/8 |

noto-sandbox의 측정(CPU 체커 −41%, fovea −29%, fovea+확장 +14%)과 같은 경향입니다.
- 지뢰찾기 확장은 정확도를 일부 되찾지만, 여는 패스(앞 패스의 KV에 어텐션)와 열지도 계산 비용이 절약분을 넘습니다.
- 씨앗은 noto의 contrast 열지도(상위 라벨과 배경 문구의 softmax)로 잡습니다. 칸 하나의 SigLIP 쌍 확률은 화면에서
  너무 작아서(씨앗 0개), 처음에는 확장이 한 칸도 열지 못했습니다.

그래서 기본값은 **CPU에서 fovea**(top-1을 지키며 약 25% 절약), **GPU에서는 전체 패스**입니다(`--screen-pass auto`).
`checker`는 절약이 가장 크지만 글자가 많은 화면에서 틀릴 수 있습니다. `-x`(지뢰찾기)는 위치를 더 정확히 잡고 싶을 때
쓰는 옵션입니다.


## 6. LLM 상주하지 않기

`--model`을 주면 Raon-1-nano는 첫 `ask` 요청 때 올라가고, `--llm-idle`(기본 120초) 동안 요청이 없으면 내려갑니다(`LlmWorker::spawn_lazy`).
블록 캐시 덕분에 다시 올리는 데 약 1초가 걸립니다. 로딩 중이나 생성 중에 들어온 요청은 캐시된 대사로 바로 반응합니다.
화면 감시는 비전 타워(0.43B)만 쓰고, 텍스트 타워(0.7B)는 사전을 바꿀 때만 CLI에서 씁니다.
