# KRAFTON Raon Zena — 도메인 내 상대 가치(Premium / Discount) 누적 & 로컬 완결형 마스터 기획서 v2

> **v2 개정 요지**: v1의 상대 평가 엔진을 `Qntology/logis-center`의 **SDS(Score Dynamics, `commerce/src-tauri/src/utils/score_dynamics.rs`)** 위에 다시 세웠습니다. 자연어 처리는 **Stanza(NLP)** 와 **granite-embedding-97m-multilingual-r2(임베딩)** 로 바꾸고, NMS는 둘을 함께 써서 자연어를 도메인에 맞춰 정리하는 단계로 다시 정의했습니다. 이미 구현된 기능(멀티턴 기억, 세션 요약, 스트리밍 툴콜, 능동 발화, 투트랙 에셋, 3패스 렌더러)은 범위에서 제외합니다. 이 문서는 구현 소스 없이 아키텍처와 명세만 다룹니다.

---

## 0. v1 → v2 무엇이 바뀌었나

| 항목 | v1 | v2 (이 문서) | 근거 (logis-center) |
| --- | --- | --- | --- |
| 형태 분석(NLP) | 손으로 만든 한국어 조사 박리(`korean.rs`, `normalize.rs`) | **Stanza** ONNX 변환본(format 2): 토큰 → MWT → 품사(UPOS/XPOS/feats) → 원형 → 의존 구문 | `src/stanza.rs`, `STANZA_DROP_TAGS` |
| 문장 임베딩 | 해시 n-gram 임베더(`HashedNgramEmbedder`) | **granite-embedding-97m-multilingual-r2** 384차원, ModernBERT CLS 풀링 | `src/models/embedding.rs` (`EMBED_DIM = 384`) |
| NMS의 역할 | 중복 억제 위주 | **Stanza 구조 + granite 의미로 자연어를 도메인에 정리**: 내용어 단위 → 구문 스팬 → 이중 센터링 → 적응 마진 배타 배정 → 시간·공간 NMS | `logic.md` Plinko 3법칙, `exclusive_assign_by_score_adaptive` |
| 임계값 | 고정 상수(`accept 0.6`, `base_tau 0.6`, 5분 창, 백분위 등급) | **SDS 적응 기준선**: 스코프별 Welford(μ, σ, n) + 최근 링 + 표본 하한 미달 시 상위 스코프로 백오프 | `resolve_entry`, `Track::min_obs` |
| 도메인 내 정규화 | 도메인 평균 대비 Z-점수 | **스코프 계층 백오프(토픽 → 도메인 → 트랙)** + 기록×결과 행렬 **이중 센터링** + 승패 장부(혼동 사전) | `double_center_matrix`, `confusion_winner`, `tie_verdict` |
| 결과 보상 배분 | 5분 창에서 시간 감쇠로 나눔 | **적격도(eligibility) × 관련도** 행렬 → 이중 센터링 → 마진이 결정적일 때만 한 기록이 크레딧을 독점, 근접 동률은 승패 장부에 기록 | `exclusive_assign`, `ConfusionStat` |
| 신구 대체 | 어휘 포함률 + 수치 변경 감지 | **Stanza 정렬 주장(claim) 비교**(같은 주어·목적어 원형, 다른 NUM·ADJ) + granite 문장 정렬 + 포함률 | — (v2 신규) |
| 도메인 간 궤적 | 공유 Relay Key + 포함률 + "PLINKO 절벽" | 전이 사전 확률(라플라스 평활) × 키 포함률 × granite 코사인. 절벽은 SDS 하한 띠로 판정 | `transition_prior`, `relay_ledger.rs` |
| 통계 저장 | LanceDB에 모두 저장 | 기록·벡터·간선은 **LanceDB**, 적응 통계는 **SDS 원장(JSON) + 실행별 run 파일**(병합 가능), 임베딩 공간별로 분리 | `ledger_root`, `runs_dir`, `seed_space_ledgers` |

---

## 1. 현황 점검 (기구현 vs 잔여)

| 영역 | 이미 구현됨 (제외) | 잔여 과제 (이 문서 범위) |
| --- | --- | --- |
| 캐릭터 챗 · 에셋 | `Dialogue`, `SessionTracker`, `ToolCallStream`, `ProactiveTimer`, `zena-avatar-mcp`, 3패스 렌더러, 말풍선·채팅창(`Ctrl+Shift+T`) | 3단계 점진 노출 UI(Stage 0/1/2) |
| 자연어 → 도메인 | 사전(`zena-lexicon`) exact·교착어 접두 일치 → 해시 임베딩 bias/prejudice 점수 → 스팬 NMS → 시간 의미 NMS(μ+3σ) → 공간-텍스트 NMS | **Stanza + granite NLP-NMS v2**, **SDS 적응 임계** |
| 창 제어 | 투명 오버레이, 호버 30% 투과, 좌클릭 드래그 | 클릭(<5px)·드래그(≥5px) 분리, 아래 창 식별·클린 ROI·OCR 번역, Auto-Dock(82%) |
| 로컬 DB | 표현 사전 LanceDB 6개 테이블(`meta`, `domains`, `phrases`, `vision`, `anchors`, `candidates`), 자가 학습·핫 리로드·롤백 | 기록·간선 테이블, SDS 원장, 상대 Premium/Discount 엔진 |
| 브라우저 · 모델 | Raon-1-nano 추론, 피라미드 양자화 블록 캐시 | `chromiumoxide` 듀얼 브라우저, 장문 노트, 인앱 모델 다운로더(Stanza·granite 포함) |

**용어**
- **NLP 모델** = Stanza. logis-center와 같은 ONNX 변환본(`huggingface.co/Qntology/stanza-<lang>/format2`)을 onnxruntime으로 돌립니다. 지원 언어는 ko, en, ja, zh-hans 등 12개입니다.
- **임베딩 모델** = `ibm-granite/granite-embedding-97m-multilingual-r2` (384차원, Apache-2.0).
- **NMS** = Stanza와 granite로 발화·OCR·웹 본문을 **도메인 단위로 정리**하는 단계입니다. 대표 하나를 고르고, 겹치는 것은 억제하고, 1:1로 배정합니다.
- **SDS** = logis-center Score Dynamics. 스코프별 적응 기준선, 감쇠 형상, 혼동 사전, 전이 사전 확률을 모아 둔 통계 장부입니다.

---

## 2. 파트 0 (신규 · 기반): Stanza + Granite NLP-NMS v2 — 자연어를 도메인에 맞춰 정리

### 2.1 파이프라인

```text
발화 / OCR / 웹 본문
  │
  ├─①  언어 판정 ── 문자 체계 → 언어 전용 글자 → 사전 어휘 투표 → whatlang(라틴 8개로 제한)
  │                 따옴표 속 이름, 외국어 이름 어절은 판정에서 제외 (logis `detect_query_lang`)
  ├─②  Stanza 분석 ── 토큰 · MWT · UPOS/XPOS/feats · lemma · head/deprel (한 번에 한 언어만 상주)
  ├─③  단위 정리 ── 내용어가 없는 단위만 제거(ADP, PUNCT, PART, SCONJ, CCONJ, PRON).
  │                  전부 기능어면 원형 유지(과잉 삭제 방어).
  │                  수식 표지 추출: 부정(Polarity=Neg, 안/못/not), 하향(덜/줄여), 순서(먼저/다음), 수치 한정
  ├─④  구문 스팬 ── 임의 n-gram 창 대신 의존 구문 덩어리: 머리 + compound/flat/amod/obj/nmod 자식.
  │                  기존 2~5어절 창은 보조로 남김
  ├─⑤  granite 배치 임베딩 ── 스팬 전부를 한 번에 순전파.
  │                  사전 앵커(bias/prejudice 문구)는 사전 리비전 × 임베딩 공간별 캐시
  ├─⑥  점수 행렬 ── 스팬 × 도메인: exact·접두 일치(0.95/0.90)는 그대로 먼저,
  │                  나머지는 bias − w·prejudice 풀링 → 이중 센터링(행 평균·열 평균 제거, 잔차만 사용)
  ├─⑦  적응 배타 배정 ── 마진 하한은 SDS 감쇠 기준선(스코프의 평소 1–2위 격차),
  │                  근접 동률은 혼동 사전(역사 승률 ≥ 0.65 + 특이도 차 ≤ 0.15)일 때만 깸
  └─⑧  시간·공간 NMS ── τ = SDS 상한 띠(μ + 3σ, 스코프별 누적 분포)로 중복 억제(기존 semantic.rs),
                          ROI 배타 할당(spatial.rs)
```

### 2.2 raon-zena 모듈 대응

| 단계 | 지금 코드 | v2 변경 |
| --- | --- | --- |
| ①② | — | 신규 `crates/zena-nlp`: `stanza.rs`(logis 실행기 이식, onnxruntime), `lang.rs`(`detect_query_lang` 이식) |
| ③ | `nlp-nms-router/src/korean.rs`, `normalize.rs`(조사·어미 박리), 사전 `function` 뱅크 | `Analyzer` 트레이트 도입: `StanzaAnalyzer`(기본), `KoreanRuleAnalyzer`(모델 미설치 시 폴백). `function` 뱅크는 사전 수준 보조로 유지 |
| ④ | `span.rs`(최대 5어절 창) | 구문 스팬 우선, 창은 보조 |
| ⑤ | `embed.rs`의 `TextEmbedder` + `HashedNgramEmbedder` | `GraniteEmbedder: TextEmbedder`(candle ModernBERT, logis `embedding.rs` 이식). 해시 임베더는 폴백 |
| ⑥⑦ | `domain.rs`: 고정 `accept 0.6`, 2위 마진 | 이중 센터링 + `exclusive_assign_by_score_adaptive` 방식 배정 + SDS 기준선 |
| ⑧ | `semantic.rs`: 배치 안 코사인 풀의 μ+3σ, 하한 `base_tau 0.6` | 배치 풀과 SDS 누적 분포 중 더 안정적인 쪽을 씀(표본 하한 충족 시 SDS) |

### 2.3 v2로 고쳐지는 실제 사례

- **"안경이랑 헤드셋 써 봐"가 화면 질문으로 새던 문제**: 이번 세션 실행에서 "봐"가 `screen.what_is_this`에 해시 벡터 0.81로 잡혔습니다. 지금은 사전 `function` 뱅크와 라우터 규칙으로 막았습니다. v2에서는 Stanza가 "봐"를 보조 용언(AUX)으로, "써 봐"를 "쓰다"의 보조 구성으로 분석합니다. 구문 스팬은 "안경 + 쓰다"가 되어 `avatar.glasses`에 붙습니다.
- **부정**: "보스 패턴 알려주지 마", "안경 벗지 마"는 Polarity=Neg 표지로 해당 스팬의 의도를 뒤집거나 보류합니다. 지금은 부정이 없는 문장과 같은 도메인으로 잡힙니다.
- **교착어**: "클리어했다", "인벤토리를"의 원형(클리어하다, 인벤토리)을 Stanza lemma로 얻습니다. 손으로 만든 받침 규칙에 덜 의존합니다.
- **바꿔 말하기**: 해시 n-gram은 어휘가 겹치지 않는 표현에 약했습니다(README 한계 5). granite 다국어 임베딩은 "보스 못 잡겠어" ↔ "또 죽었어"처럼 어휘가 달라도 같은 도메인으로 묶습니다.

### 2.4 자원 예산과 폴백

| 모델 | 크기 (logis 측정) | 상주 정책 |
| --- | --- | --- |
| Stanza (ko) | 언어당 0.3–0.8 GB CPU RAM | 첫 발화 때 로드, 한 언어만 상주, 유휴 N초 뒤 해제(LLM `spawn_lazy`와 같은 방식) |
| granite-embedding-97m | 약 380 MB (F32) | CPU 배치 기본(짧은 스팬 위주), 선택적으로 GPU. raon-quant Q8_0 적재는 품질 확인 후 결정 |

- 모델이 없으면 지금의 규칙 분석기와 해시 임베더로 그대로 동작합니다. 다운로더가 받은 뒤 핫 스왑합니다.
- **임베딩 공간 분리**: SDS 원장, 앵커 캐시, 기록 벡터에 공간 태그(예: `granite-cls-v1`, `hash-v1`)를 붙입니다. 공간이 바뀌면 logis처럼 이전 원장을 씨앗으로 복사하고 이후 따로 키웁니다. 코사인 눈금이 다른 공간끼리 섞이지 않게 하기 위해서입니다.
- "100% Raon" 원칙(README 한계 5)은 이번 결정으로 텍스트 처리에 한해 완화합니다. Stanza와 granite 모두 Apache-2.0이므로 NOTICE에 추가합니다.

---

## 3. 파트 1 (개정): SDS 기반 도메인 내 상대 Premium / Discount 엔진

### 3.1 SDS 구성 요소 → raon-zena 대응

| SDS 요소 (logis) | 하는 일 | raon-zena에서의 쓰임 |
| --- | --- | --- |
| `Scope { track, primary, secondary }` + 키 3단 | 통계 격리 단위 | `track` = 기록 종류(`drop`, `web`, `coach`, `nms`). `primary` = 도메인 뿌리(`app://eldenring.exe`, `web://store.steampowered.com`). `secondary` = 토픽(Relay Key 군집, 예: `boss-malenia`) |
| `Track::min_obs() = (secondary, primary, global)` | 단계별 표본 하한. 미달이면 위 단계로 백오프, 모두 미달이면 **판정 없음(None)** | 개인 사용량에 맞춘 초기값(예: drop (6, 3, 10), web (8, 4, 12)). 평가 세트로 조정 |
| `Welford { n, mean, m2, ring }` + `merge` | 병합 가능한 평균·분산 + 최근 링(`ring_len = min_obs.1`) | 축별 기준선: `valence`, `credit.margin`, `align.cos`, `traj.link`, `nms.dup_cos` 등 |
| `drift_z = (최근 평균 − 전체 평균) / σ` | 분포 이동 감지 | **메타 변화 감지**: 패치 후 이전 해법이 갑자기 실패하면 그 스코프의 오래된 Premium 크레딧을 감쇠 |
| `decay_shape`(top1−top2 마진, top_gap_ratio, tail_flatness, entropy_norm) | 점수 분포가 결정적인지 평평한지 | 크레딧 배분과 비교 브리핑이 **결정적일 때만** 단정. 평평하면 "아직 비슷해"로 말함 |
| `double_center_matrix` | 행·열 평균 제거로 "다 높게 나오는" 현상 제거 | 결과×기록 크레딧 행렬, 스팬×도메인 점수 행렬 |
| `exclusive_assign_by_score_adaptive` | 마진 큰 순 1:1 배정, 동률은 호출부가 허용 판단 | 결과 하나가 기록 하나에 주 크레딧. 동률은 승패 장부 판단 |
| `ConfusionStat`, `confusion_winner`, `tie_verdict`(`TIE_WIN_RATE = 0.65`, `TIE_SPECIFICITY_SLACK = 0.15`) | 쌍별 역사 승률로 동률 깨기 | **경쟁 기록 대결 장부**: 같은 토픽 두 기록의 승패. `outperforms` 간선은 승률 ≥ 0.65일 때만 |
| `FieldRejectStat` → `learned_specificity` | 거절률을 학습 페널티로 | 사용자가 ▼ 강등하거나 결과가 반복 실패한 기록의 학습 페널티 |
| `transition_prior` = (hit+1)/(total+k) | 도메인 전이 확률(라플라스 평활, 하한 미달이면 None) | 도메인 간 궤적 간선 강도의 사전 확률 |
| `refine_primary` | 더 정밀한 스코프가 밝혀지면 이번 실행 관측을 이관 | 창 제목·URL이 늦게 확정되면(로딩 중 → 실제 페이지) 그 사이 관측을 새 스코프로 이관 |
| 원장(base) + 실행별 run 파일, 읽을 때 병합, 종료 시 접어 넣음 | 충돌 안전, 동시 실행 안전 | 같은 방식으로 `%LOCALAPPDATA%\RaonZena\score_dynamics\spaces\<공간>\{drop,web,coach,nms}.json` + `runs/` |

### 3.2 기록 하나의 가치: 원시 가치 → 상대 순위 → 등급

**(1) 원시 가치 $v_r$**: 결과 신호를 크레딧으로 받은 누적값입니다.

$$v_r = \sum_{o} s_o \, w_o \, c_{o,r}\, e^{-(t_{\text{now}} - t_o)/T_{\text{scope}}} \;+\; \beta_{\text{dwell}} + \beta_{\text{redrop}} + \beta_{\text{pin}} \;-\; \lambda\,\text{spec}_r$$

- $s_o \in \{+1,-1\}$: 결과 부호. `vision.game_victory`, `emotion.celebrate`, `emotion.thanks`는 +1, `vision.game_defeat`, `emotion.frustrated`는 −1입니다(이미 `pipeline.rs`가 잡는 신호).
- $w_o$: 신호 세기. 화면 이벤트 > 사용자 감탄 > 감사 순입니다.
- $c_{o,r}$: 3.3의 크레딧 배분 결과입니다.
- $T_{\text{scope}}$: 반감기입니다. `drift_z`가 크면(메타 변화) 짧아집니다.
- $\text{spec}_r$: 학습 페널티(거절률)입니다.

**(2) 상대 순위**: 스코프 기준선 $(\mu, \sigma, n)$은 토픽 → 도메인 → 트랙 순으로 백오프해 구합니다.

$$\text{relative\_rank}_r = \tanh\!\left(\frac{v_r - \mu}{\sigma + \epsilon}\right) \in (-1, 1)$$

기준선이 하한 미달로 None이면 순위를 내지 않습니다. 이 경우 같은 토픽 경쟁 기록과의 대결 장부가 결정적일 때만 그 결과를 씁니다. 장부도 결정적이지 않으면 `neutral`입니다. logis의 "표본이 없으면 상수로 단정하지 않는다" 원칙과 같습니다.

**(3) 등급과 히스테리시스**

| 등급 | 진입 조건 | 이탈 조건 |
| --- | --- | --- |
| `premium` | rank ≥ +0.5 **이고** 2위와의 마진이 SDS 감쇠 기준선의 평소 마진 이상, 또는 대결 승률 ≥ 0.65 | rank < +0.3 또는 `superseded` |
| `discount` | rank ≤ −0.5, 대결 패배율 ≥ 0.65, 또는 `superseded` | rank > −0.3이고 대체 해제 |
| `neutral` | 그 밖, 또는 표본 하한 미달 | — |

경계에서 등급이 매번 뒤집히지 않도록 진입·이탈 띠를 분리했습니다. 등급이 바뀔 때마다 근거를 한 줄 로그로 남깁니다(logis `[SDS]` 로그 형식). 예: `[REL] boss-malenia #12 → premium | n=7 μ=0.10 σ=0.31 rank=+0.71 대결 4승 1패`.

### 3.3 결과 크레딧 배분 (v1의 "5분 창 시간 감쇠"를 대체)

1. **적격 기록**: 결과 $o$ 직전 창(스코프별 SDS 기준선, 기본 5분)에서 사용자가 실제로 접한 기록만 후보입니다. 접촉이란 드롭 브리핑을 끝까지 들음, 체류 ≥ k초, 복사, 재드롭을 말합니다. 그냥 화면에 있었던 기록은 후보가 아닙니다.
2. **행렬**: 행 = 결과(창 안에 여러 개면 모두), 열 = 후보 기록입니다. 값은 적격도 $e^{-\Delta t/\tau} \times$ 접촉 가중치 $\times$ granite 관련도입니다. 관련도는 기록 요약과 결과 맥락(장면, 토픽 키, 직전 발화) 사이의 코사인입니다.
3. **이중 센터링 → 적응 배타 배정**: 마진이 결정적이면 한 기록이 주 크레딧 $c = 1$을 받습니다. 근접 동률이면 두 기록에 나눠 주고(0.5/0.5) `ConfusionStat`에 동률을 기록합니다. 다음에 같은 쌍이 붙으면 역사 승률로 깹니다.
4. **SDS 기록**: `credit.margin` 축 Welford, 감쇠 형상, 대결 장부를 갱신합니다.

### 3.4 신구 대체 (Superseding & Obsolescence)

같은 스코프·토픽에 새 기록 N이 들어오면 기존 기록 O와 다음 순서로 비교합니다.

1. **문장 정렬**: Stanza 문장 분할 → granite 임베딩 → N×O 문장 유사도 행렬 → 이중 센터링 → 배타 배정합니다. 짝이 되려면 `align.cos` 축 SDS 상한 띠를 넘어야 합니다.
2. **주장 비교**: 짝지은 문장에서 핵심 원형(nsubj/obj/root lemma)이 같은데 NUM, 단위, ADJ가 다르면 **변경 주장**입니다(예: "피해 20% → 15%", "쿨다운 8초 → 10초").
3. **포함률**: $|K_O \cap K_N| / |K_O|$ (Relay Key 기준). SDS 기준선 이상이어야 같은 대상으로 봅니다.
4. **판정**
   - 변경 주장 ≥ 1, 포함률 충족, N이 더 최신이면 `supersedes(N → O)`입니다. O는 `discount`(구버전)가 되고 사유로 변경 주장 목록을 남깁니다.
   - 변경 없이 N이 O를 포함하면 **중복**입니다. 시간 NMS처럼 합치고 가치가 높은 쪽을 대표로 둡니다.
   - 일부만 겹치면 대체하지 않고 `semantic_link`만 연결합니다.

### 3.5 사용자 수동 보정

Stage 2의 ▲ 승격과 ▼ 강등은 가장 강한 신호입니다($w$ 최대). ▼는 `FieldRejectStat`처럼 거절로 세어 학습 페널티에 반영합니다. 같은 토픽의 다른 기록은 즉시 다시 순위를 매기되, 대결 장부에는 사용자 판정을 별도 표시로 남깁니다. 사용자 판정이 자동 결과에 덮이지 않게 하기 위해서입니다.

### 3.6 예시: 말레니아 공략 (값은 설명용)

```text
스코프: drop | app://eldenring.exe | boss-malenia     (하한 secondary 6 / primary 3 / global 10)

1) Drop #1 "방패 막기 위주" ─ 브리핑 끝까지 들음(접촉)        → neutral (표본 0, 판정 없음)
2) vision.game_defeat ─ 적격 기록 {#1} → 크레딧 c=1, v₁ = −0.9
3) Drop #2 "구르기 회피 + 빙결 항아리" ─ 접촉
4) emotion.celebrate "와 잡았다 고마워" ─ 적격 {#2(Δt 2분), #1(Δt 9분, 창 밖)} → #2 단독 크레딧, v₂ = +1.0
5) 대결 장부 #1 vs #2: #2 1승. valence 관측은 토픽 2개(하한 6 미달) → 도메인 app://eldenring.exe 2개(하한 3 미달)
   → 트랙 drop 전체 2개(하한 10 미달) → 기준선 None
   └─ 판정 가능해질 때까지 등급은 neutral로 두고, 브리핑은 "지난번엔 구르기로 잡았어"처럼 사실만 말함
6) 같은 패턴 반복 후 n ≥ 6 → #2 premium(rank +0.8), #1 discount(대결 3승 0패 → 승률 1.0 ≥ 0.65)
```

v1과 달리 기록 두 개로 Premium/Discount를 단정하지 않습니다. 근거가 쌓일 때까지 제나는 사실만 말합니다.

---

## 4. 파트 2 (개정): Relay Key · Trajectory 그래프 · 저장 스키마

### 4.1 Relay Key 추출 (Stanza + granite)

- **후보**: Stanza 기준 PROPN, NUM이 섞인 식별자, X/SYM이 섞인 토큰(`SurfaceError`, `E0425`, `wgpu`), 그리고 compound/flat으로 이어진 명사 덩어리("말레니아 2페이즈")입니다.
- **정규화**: lemma, 소문자, 따옴표 속 이름은 원문 유지(logis `quoted_ranges`)입니다.
- **군집**: granite 코사인이 `relay.cos` SDS 상한 띠를 넘는 후보끼리 같은 정규 키로 묶습니다("말레니아" ↔ "Malenia").
- **역색인**: 키 → 노드 ID. 도메인 간 연결은 logis `relay_ledger.rs`처럼 키를 매개로 한 다리(bridge)로 합니다.

### 4.2 도메인 간 궤적과 에피소드 경계

$$S(a \to b) = P_{\text{trans}}(\text{dom}_a \to \text{dom}_b) \times \frac{|K_a \cap K_b|}{|K_a \cup K_b|} \times \cos_{\text{granite}}(a, b)$$

- $P_{\text{trans}}$는 SDS `transition_prior`(라플라스 평활)입니다. 하한 미달이면 1로 두지 않고 해당 항을 빼서 계산합니다.
- **PLINKO 절벽(새 `trajectory_id` 발급)**: 직전 노드와의 $S$가 `traj.link` 축 SDS 하한 띠(μ − 2σ) 아래로 떨어지거나, 공유 키가 0이고 시간 간격이 스코프 평소 간격의 상한을 넘을 때입니다.

### 4.3 저장소 배치

| 저장소 | 담는 것 | 이유 |
| --- | --- | --- |
| LanceDB `trajectory_nodes` | 기록 본문, 요약, 벡터, 등급 | 벡터 검색, n-gram BM25, 버전 롤백이 필요함 |
| LanceDB `trajectory_edges` | 궤적, 대결, 대체 간선 | 그래프 질의, 버전 관리 |
| LanceDB `relay_index` | 정규 키 → 노드 ID 목록, 동의 표면형 | 키로 빠르게 연결 |
| SDS 원장(JSON) + `runs/` | 스코프별 Welford, 감쇠 형상, 대결 장부, 전이 횟수, 거절 통계 | 병합 가능한 통계, 실행별 파일로 충돌 안전, 임베딩 공간별 분리(logis 방식 그대로) |

### 4.4 `trajectory_nodes` 스키마 (v1 대비 변경 표시)

| 컬럼 | Arrow 타입 | 설명 |
| --- | --- | --- |
| `node_id` | Int64 | 기록 ID |
| `trajectory_id` | Utf8 | 에피소드 ID (4.2 절벽에서 새로 발급) |
| `timestamp_ms` | Int64 | 생성 시각 |
| **`scope_track` / `scope_primary` / `scope_secondary`** | Utf8 ×3 | **(신규)** SDS 스코프 키 3단 (`drop` / `app://eldenring.exe` / `boss-malenia`) |
| `domain_uri` | Utf8 | `app://<exe>/<scene>/<workspace>` 또는 `web://<host>/<path_category>` |
| `exe_name`, `window_title`, `vision_domain` | Utf8 | 아래 창 정보, 화면 장면 |
| `lang` | Utf8 | **(신규)** Stanza 판정 언어 |
| `raw_text` | Utf8 | OCR, 브라우저 본문 (n-gram BM25) |
| `zena_summary` | Utf8 | 길이 제한 없는 한국어 노트 |
| **`claims`** | Utf8 (JSON) | **(신규)** Stanza 주장 목록: 핵심 원형과 수치·형용사 (3.4에서 사용) |
| `relay_keys` | **List\<Utf8\>** | v1의 JSON 문자열 대신 리스트 (정규 키) |
| **`text_vec`** | FixedSizeList(Float32, **384**) | **(변경)** granite 텍스트 임베딩 (텍스트 비교의 주 벡터) |
| **`image_vec`** | FixedSizeList(Float32, 1152), nullable | **(변경)** 드롭 ROI의 Raon-VisionEncoder 벡터 (화면 유사도 보조) |
| `embed_space` | Utf8 | **(신규)** 벡터 공간 태그 (공간이 다른 벡터끼리 비교 금지) |
| `valence_score` | Float32 | 3.2(1)의 $v_r$ |
| `relative_rank` | Float32, nullable | 3.2(2). 하한 미달이면 **null** (v1은 0.0으로 단정) |
| `tier` | Utf8 | `premium` / `neutral` / `discount` |
| **`tier_reason`** | Utf8 | **(신규)** 등급 근거 한 줄 (n, μ, σ, 대결 전적, 변경 주장) |
| **`outcomes`** | Utf8 (JSON) | **(신규)** 받은 크레딧 이력 (결과 종류, 부호, c, 시각) |
| `superseded_by` | Int64, nullable | 이 기록을 대체한 기록 |
| `prev_node_id` | Int64, nullable | 직전 궤적 노드 |

### 4.5 `trajectory_edges`

| 컬럼 | 타입 | 설명 |
| --- | --- | --- |
| `from_node_id`, `to_node_id` | Int64 | 간선 양 끝 |
| `from_domain`, `to_domain` | Utf8 | 도메인 URI |
| `edge_kind` | Utf8 | 도메인 간: `temporal_flow` / `entity_relay` / `semantic_link`. 도메인 내: `outperforms` / `supersedes` / `duplicate_of` |
| `shared_keys` | List\<Utf8\> | 매개 Relay Key |
| `weight_or_delta` | Float32 | 도메인 간은 $S$, 도메인 내는 $\Delta V$ |
| **`evidence`** | Utf8 (JSON) | **(신규)** 대결 전적(n, 승률), 변경 주장 쌍, 정렬 코사인 |

---

## 5. 파트 3: 드래그 앤 드롭 조사 & Auto-Dock (v1 유지, 연결부만 보강)

1. **클릭(<5px) vs 드래그(≥5px)**: v1과 같습니다. 클릭은 Stage 1을 열고, 드래그 후 놓으면 드롭 조사를 시작합니다.
2. **아래 창 식별과 클린 캡처**: v1과 같습니다. `VisionKind::Privacy`이면 즉시 0으로 지우고 기록하지 않습니다(기존 `watcher.rs` 정책 재사용).
3. **OCR → NLP-NMS v2**: OCR 원문을 ①언어 판정 → ②Stanza → Relay Key·주장 추출 → 스코프 결정(`app://exe` + 장면 + 토픽) 순으로 처리합니다. 창 제목이 늦게 바뀌면 `refine_primary`처럼 스코프를 이관합니다.
4. **브리핑**: Raon-1-nano가 번역·설명을 생성합니다. 같은 토픽 기록이 있을 때 비교 문구는 **등급이 확정된(neutral이 아닌) 기록만** 씁니다. 감쇠 형상이 평평하면 "비슷한 기록이 둘 있어"처럼 단정하지 않습니다.
5. **Auto-Dock**: v1과 같습니다. 발화 종료 5–6초 뒤 0.45초 Cubic Ease-Out으로 우측 하단에 복귀하고, 불투명도 82% + `HWND_TOPMOST`를 유지합니다.

---

## 6. 파트 4: 3단계 클릭 UI (v1 유지, Stage 2만 보강)

- Stage 0 (앰비언트 말풍선) → Stage 1 (3줄 라이브 피드 + 입력창) → Stage 2 (히스토리·필터 모달) 흐름은 v1과 같습니다. 기존 말풍선·채팅창(`text.rs`, `Ctrl+Shift+T`)을 확장합니다.
- **Stage 2 보강**
  - 등급 카드에 **근거 칩**을 표시합니다(`tier_reason`): "n=7 · 대결 4승 1패 · rank +0.71", "구버전: 쿨다운 8초 → 10초".
  - 검색은 n-gram BM25 + **granite 384차원** 벡터 검색을 함께 씁니다. 화면 유사도가 필요하면 `image_vec`를 보조로 씁니다.
  - 표본 하한 미달 토픽은 "평가 중(근거 n/6)"으로 표시해 등급을 단정하지 않습니다.
  - ▲/▼ 수동 보정은 3.5 규칙을 따릅니다.

---

## 7. 파트 5: `chromiumoxide` 듀얼 브라우저 (v1 유지, 비교 방식 개정)

- **DOM Diff → 문장 단위 정렬**: 변경을 감지하면 이전 스냅샷과 새 본문을 3.4 절차(Stanza 문장 분할, granite 정렬, 주장 비교)로 비교합니다. 결과를 추가, 삭제, **변경 주장**(수치·형용사 변화)으로 나눕니다. 브리핑은 변경 주장 위주로 말합니다("보스 체력 15% 하향").
- 스코프는 `web | web://<host> | <path_category 또는 Relay 토픽>`입니다.
- 라이브 GUI 전환(`browser.show_live`)은 v1과 같습니다.

---

## 8. 모델 관리 · 인앱 다운로더

| 모델 | 출처 | 검증 |
| --- | --- | --- |
| Stanza `ko` (+ 선택 언어) | `huggingface.co/Qntology/stanza-<lang>/resolve/main/format2/vocab.json`의 `files` 목록 (logis `remote_base`, `manifest_files`) | format 1 저장소는 오류로 보고 설치하지 않음. 모든 파일이 있어야 설치 완료 |
| granite-embedding-97m-multilingual-r2 | `ibm-granite/granite-embedding-97m-multilingual-r2` | SHA-256 (`scripts/fetch_models.sh`와 같은 HF LFS 해시 대조) |
| Raon-1-nano, Raon-VisionEncoder | 기존 | 기존 |

- 이어받기, 진행률 말풍선은 v1과 같습니다.
- 설치가 끝나면 `Analyzer`와 `TextEmbedder`를 핫 스왑하고, 사전 앵커를 새 임베딩 공간으로 다시 계산합니다(백그라운드).

---

## 9. 개정 로드맵

| 단계 | 내용 | 완료 기준 (DoD) |
| --- | --- | --- |
| **Phase 0 (신규, 기반)** | `crates/zena-nlp`(Stanza 실행기, 언어 판정, granite 임베더), `crates/zena-sds`(Scope, Welford, 백오프 resolve, 감쇠 형상, 이중 센터링, 적응 배타 배정, 혼동 사전, 전이 사전 확률, 원장 + run 파일), NLP-NMS v2 라우터 연결, 폴백 유지 | ① 한국어 발화 평가 세트(게임, 코딩, 시장, 명령; 정답 도메인 라벨)에서 v1 대비 도메인 정확도와 **명령 오탐률**을 측정 ② 모델 미설치 시 v1 동작 그대로 ③ Stanza·granite 지연(발화당 ms)과 RAM을 측정해 문서화 ④ 이번 세션 실사례("써 봐", 부정문)를 회귀 테스트로 고정 |
| Phase 1 | 드래그 앤 드롭 조사, Auto-Dock, 3단계 UI | v1 DoD + 드롭 기록이 스코프 3단 키와 함께 저장됨 |
| Phase 2 | 상대 Premium/Discount 엔진(3장), Relay Key와 궤적(4장), Stage 2 근거 칩 | ① 합성 세션 재생(승패 이벤트 대본)으로 크레딧 배분·등급 전이·히스테리시스 검증 ② 표본 하한 미달 시 `relative_rank = null` 보장 ③ 신구 대체: 수치 변경 패치노트 쌍에서 `supersedes` 생성 |
| Phase 3 | `chromiumoxide` 스케줄러, 문장 정렬 Diff 브리핑 | 같은 페이지 두 스냅샷에서 변경 주장 추출 정확도 측정 |
| Phase 4 | 인앱 다운로더(Stanza, granite 포함), 핫 스왑 | 빈 환경에서 첫 실행 → 다운로드 → 핫 스왑 → 앵커 재계산까지 무중단 |

**평가 방법**: logis-center `SyntheticData/`(관계형 합성 질의, `sds_runs.py`, `sds_crosscheck.py`)처럼 raon-zena에도 `data/eval/`에 합성 발화와 세션 대본을 두고 SDS 실행 간 결과를 대조합니다.

---

## 10. 리스크와 결정이 필요한 사항

1. **런타임 의존성**: Stanza는 logis처럼 onnxruntime(ONNX)으로 돌립니다. candle로 다시 구현하는 것은 범위 밖으로 둡니다. raon-zena에 onnxruntime 의존성이 새로 생깁니다.
2. **RAM**: Stanza(ko)가 0.3–0.8 GB를 씁니다. 지연 로드와 유휴 해제로 평소 상주는 0에 가깝게 유지하지만, 발화가 잦으면 상주합니다. 기존 PRD의 RAM 목표와 충돌하므로 목표를 재설정해야 합니다.
3. **게임 은어·신조어**: Stanza 품사가 틀릴 수 있습니다. 정확 일치 사전이 여전히 벡터보다 먼저 판정하고, 학습 후보(`candidates`) 승격도 유지합니다.
4. **SDS 표본 하한**: 개인 사용량은 logis 업무량보다 훨씬 적습니다. 하한이 높으면 오래 neutral에 머뭅니다. 트랙(global) 백오프와 대결 장부가 초기 판단을 보완하며, 하한 값은 평가 세트로 조정합니다.
5. **임베딩 공간 전환**: 해시에서 granite로 바뀌는 순간 기존 기록의 벡터를 다시 계산해야 합니다. 백그라운드 재색인 동안에는 공간 태그로 비교를 막습니다.

---

## 11. Phase 0 구현 결과 (2026-11-16)

### 만든 것

| 크레이트·모듈 | 내용 |
| --- | --- |
| `crates/zena-sds` | logis-center SDS 이식. `Scope` 3단 키와 백오프, 병합 가능한 `Welford` + 최근 링, `decay_shape`, 혼동 사전(`tie_break_allows`, 0.65 / 0.15), 거절 통계, 라플라스 전이 사전 확률, `matrix::{double_center, exclusive_assign_adaptive}` 를 담았습니다. 전역 상태 없이 `Ledger` 값 하나로 동작하고, 원장 + 실행별 run 파일 + 파일 잠금으로 충돌에 안전합니다. 테스트 31개. |
| `crates/zena-nlp::stanza` | logis-center Stanza 실행기(format 2, 12개 언어) 이식, 언어 판정, decode 대조 시험 이식. |
| `crates/zena-nlp::granite` | granite-embedding-97m-multilingual-r2 ModernBERT 순전파(CLS). HF transformers F32 기준 17문장 모두 코사인 ≥ 0.9999999. |
| `crates/onnxruntime(-sys)` | logis-center 벤더 판을 가져왔습니다. ONNX Runtime 은 **실행 시 로드**하고 링크하지 않으므로, 라이브러리가 없어도 앱이 뜨고 규칙 분석기로 돌아갑니다. |
| `nlp-nms-router` | `Analyzer`(Stanza / 규칙 / 없음): 내용어·기능어·보조 용언("써 **봐**", "해 **줘**")·부정("-지 **마**", "안", `Polarity=Neg`)과 부정 범위(절 단위)를 정합니다. `SharedEmbedder` + `GraniteEmbedder`(배치, 캐시), v2 선택 `span_assign`(이중 센터링 → 마진 게이트 1:1 배정 → 겹침 억제), `Models::{v1, rules, auto}`, 부정된 명령은 실행하지 않고 로그만 남깁니다. SDS `nms` 트랙에는 감쇠 형상과 보류된 동률을 기록하고, 표본이 차면 마진을 적응값으로 바꿉니다. |
| 앱 | `--nlp auto|rules|v1`(기본 auto). SDS 원장은 `%LOCALAPPDATA%\RaonZena\score_dynamics\spaces\<임베딩 공간>\nms.json` 에 두고 15초마다 run 파일로 기록합니다. |
| 평가 | `data/eval/nlp_nms_ko.jsonl`(93문장: 정확 34 · 바꿔 말하기 38 · 다중 의도 3 · 부정 8 · 무관 10), `tests/eval_ko.rs`(비교표, `calibrate_v2` 임계값 탐색). |

### 측정 (Windows 11, CPU, release)

| 구성 | 정확도 | **명령 오탐** | 정확 | 바꿔 말하기 | 다중 | 부정 | 무관 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 (해시) | 64/93 (68.8%) | **9** | 29/34 | 20/38 | 3/3 | 4/8 | 8/10 |
| v1 + 규칙 분석기 | 67/93 (72.0%) | 6 | 29/34 | 19/38 | 3/3 | 8/8 | 8/10 |
| **v2 (granite + Stanza + 혼합)** | **74/93 (79.6%)** | **1** | 33/34 | 21/38 | 3/3 | **8/8** | 9/10 |

- v1은 부정된 명령 8개 중 4개를 그대로 실행했습니다("앱 끄지 마" 등). v2와 규칙 분석기는 하나도 실행하지 않습니다.
- 지연: Stanza 한국어 모델 로드 0.25–0.56 s, granite 로드 약 1 s. 발화 하나는 임베딩 캐시가 비었을 때 약 80 ms, 캐시가 찼을 때 약 15 ms 입니다(CPU, 다른 빌드와 함께 돈 측정이라 상한값으로 봐야 합니다). granite 상주 메모리는 약 +410 MiB 입니다.

### 계획과 달라진 점

1. **혼합 점수(dense + sparse) 추가**: granite CLS 코사인은 짧은 한국어 구에서 0.70–0.87의 좁은 띠에 몰립니다. 그래서 의미가 가까운 도메인끼리(예: `market.stock_down` ↔ `emotion.frustrated`) 마진 게이트에서 막혔습니다. 해시 점수가 0.5를 넘는 만큼의 절반을 더하자 바꿔 말하기가 15 → 21/38로 올랐고, 오탐은 늘지 않았습니다.
2. **사전 문구도 내용어로 임베딩**: avatar 도메인 문구가 모두 "~ 봐/줘"로 끝나 서로 비슷하게 보였습니다. 규칙 분석기로 보조 용언을 뗀 뒤 임베딩합니다.
3. **정확 일치 우선**: 벡터 점수가 0.95보다 높아도 정확 일치는 정확 일치로 판정합니다.
4. **구문 스팬은 보류**: 스팬은 여전히 1–5어절 창입니다. 다만 내용어가 없는 창은 제외하고, 벡터 적중은 내용어 2글자 이상일 때만 인정합니다. 의존 구문 덩어리로 바꾸는 일은 남아 있습니다.
5. **SDS 레벨 누적(roll-up) 없음**: logis 원본 그대로 관측은 스코프 전체 키에만 쌓입니다. 상위 레벨이 하위 관측의 합이 되게 할지(3.6 예시의 전제)는 Phase 2 상대 가치 엔진에서 정합니다.
6. **임계값은 직접 만든 93문장으로 맞췄습니다.** 과적합을 줄이려고 목표 함수가 평평한 구간(accept 0.62–0.70이 같은 점수)의 가운데 값을 골랐습니다. 실제 대화 로그로 다시 평가해야 합니다.

### 남은 것

- 의존 구문 스팬, 바꿔 말하기 재현율(어휘가 겹치지 않는 문장 17개 실패)
- 한국어 외 다른 언어의 평가 세트
- 인앱 모델 다운로더(Phase 4)
- 상대 Premium/Discount 엔진(Phase 2)
