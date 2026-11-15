# 4계층 피라미드 양자화 (`raon-quant`)

PRD §3 구현 내역과 실제 KRAFTON 가중치로 측정한 결과입니다.

## 파이프라인

1. **입력.** 원본 safetensors(BF16/F16/F32)만 받습니다. 양자화된 입력은 거부하므로 이중 양자화 경로는 원천 차단됩니다.
   - `SafeTensors::open_model_dir`가 샤드 인덱스를 따라 모든 파일을 `mmap`합니다.
   - 행 단위로 그 자리에서 f32로 디코드하므로 1.6B PLE 테이블도 전체를 메모리에 올리지 않습니다.
2. **계획.** `pyramid::plan_raon1_nano` / `plan_raon_ve`가 텐서 이름과 shape로 계층, 형식, 상주 위치, 패딩을 정합니다.
3. **인코딩.** `kquant`(logis-center `ggml-kquant`에서 이식)의 중요도 가중 최소제곱 탐색을 씁니다.
   - 가중치는 `w_i = rms(서브블록) + |x_i|`이고, imatrix를 주면 곱합니다.
   - 대칭 형식은 후보 19개, 비대칭 형식은 후보 21개를 닫힌 해로 풉니다.
   - F16 슈퍼스케일은 하위 블록이 상한에서 잘리면 1 ULP 올립니다.
4. **출력.** GGUF v3로 씁니다(정렬 32, `.part`에 쓴 뒤 fsync하고 rename).
   - 상주(Device) 텐서를 앞에, 디스크 gather 텐서를 뒤에 배치해 업로드 구간이 연속되게 했습니다.
   - 메타데이터: `raon.tensor_meta`(텐서별 계층/상주/논리 shape/원본 이름), `raon.quant.config`, `raon.source_config`(원본 `config.json`)
5. **보고서.** `<out>.report.json`에 텐서별 바이트와 상대오차, 계층별·상주별 합계를 남깁니다.

## 계층 정책 (실제 텐서 이름 기준)

| 계층 | 형식 | Raon-1-nano | Raon-VisionEncoder (비전 타워) |
|---|---|---|---|
| T1 무손실 척추 | 원본 BF16 / F32 | 모든 RMSNorm, 스칼라·1-D 텐서<br>Raon-mHC 믹서(`phi_*`, `alpha_*`, `b_*`, `res.*`)<br>RDA 상태 게이트(`A_log`, `dt_bias`, `b_proj`(β), `rda_w_proj`, `f_a_proj`/`f_b_proj`(α))<br>`q/k/v_conv1d` | norm, bias<br>`patch_embed`(4-D conv), `pos_embed`(NaFlex 보간)<br>`attn_pool.latent`, `logit_scale`/`logit_bias` |
| T2 관문·경계 | Q8_0 | 앞 2층과 뒤 2층<br>MTP 층(`model.layers.24.*`, `eh_proj`)<br>PLE `embedding_projection`<br>`lm_head.kept`, `embed_tokens`(디스크) | 앞 2블록과 뒤 2블록<br>MAP 헤드(`attn_pool.*`) |
| T3 어텐션 | Q6_K (`--tier3 q5_k` 가능) | RDA `q/k/v/g/o_proj`<br>MLA `q_a/q_b/kv_a_proj_with_mqa/kv_b/g/o_proj`<br>PLE 입력 게이트 | `attn.qkv`, `attn.proj` |
| T4 대용량 | Q4_K | MLP `gate/up/down_proj`<br>PLE 테이블(kept + cold, 디스크) | `mlp.fc1`, `mlp.fc2` |

- **텍스트 타워 제외.** `text.*` 텐서 330개(0.708B)는 쓰지 않습니다.
- **SO400M 폭 패딩.**
  - 폭 1152는 K-quant를 위해 1280으로 0 패딩하고, 런타임에 활성값도 같이 0 패딩합니다.
  - MLP 은닉 폭 4304는 fc1의 **행**과 fc2의 **열**을 같은 폭으로 패딩합니다. Q8_0이면 4320, K-quant면 4352입니다. bias도 0이고 `gelu(0)=0`이므로 곱이 정확히 보존됩니다(logis-center와 같은 기법).
- **tie 확인.** `lm_head.weight`가 `embed_tokens`와 바이트 단위로 같은지 표본 행으로 확인한 뒤 제외합니다.

## 결과

### Raon-1-nano (+ MTP 55M)

| 계층 | 텐서 | 파라미터 | MiB | bpw | 가중 상대오차 |
|---|---:|---:|---:|---:|---:|
| T1 | 773 | 15.0 M | 28.9 | 16.2 | 0 |
| T2 Q8_0 | 54 | 686.5 M | 695.6 | 8.50 | 0.51% |
| T3 Q6_K | 145 | 359.6 M | 281.3 | 6.56 | 1.76% |
| T4 Q4_K | 62 | 2,162.7 M | 1,160.2 | 4.50 | 7.21% |
| 보조(kept_ids, 가드) | 5 | — | 7.0 | — | — |
| **상주 Device** | 1031 | 1,228 M | **905.2** | 6.18 | |
| **디스크 gather** | 3 | 1,996 M | 1,260.8 | 5.30 | |

- **디스크 gather 대상**: `embed_tokens`(Q8_0, 404 MiB), PLE 테이블 kept/cold(Q4_K, 856 MiB). 토큰마다 필요한 행만 `GgufFile::row_bytes`로 페이지 캐시에서 바로 읽으며, 상주 0 MB입니다.
- **상주 구성**: 본체 약 848 MiB, `lm_head.kept` 38 MiB(24,847행 × 1536, Q8_0), BF16 척추 29 MiB.

### Raon-VisionEncoder 비전 타워

| 계층 | 텐서 | 파라미터 | MiB | bpw(패딩 포함) | 가중 상대오차 |
|---|---:|---:|---:|---:|---:|
| T1 | 231 | 1.6 M | 6.1 | 32 | 0 |
| T2 Q8_0 | 21 | 76.1 M | 77.3 | 8.52 | 0.53% |
| T3 Q6_K | 46 | 122.1 M | 106.1 | 7.29 | 1.75% |
| T4 Q4_K | 46 | 228.1 M | 130.6 | 4.80 | 7.28% |
| **합계** | 344 | 427.9 M | **320.1** | 6.28 | |

## 어휘 절삭 (`kept_ids`)

Raon-1-nano 토크나이저(byte-level BPE, 259,840행)를 GPT-2 바이트 역매핑으로 분석한 결과:

| 분류 | 개수 |
|---|---:|
| ASCII 토큰 | 194,762 |
| 한글 포함 토큰 | 52,679 |
| CJK 토큰 | 6,139 |
| 기타 토큰 | 4,165 |
| 불완전 UTF-8 조각 | 1,455 |
| 특수 토큰 | 640 (그중 `<\|reserved_N\|>` 625개) |

1. **문자 영역 필터만 적용하면** 다음을 유지하고 나머지 10,318개(4%)만 제외됩니다. PRD가 가정한 `K ≪ V`는 성립하지 않습니다.
   - ASCII, 한글/자모, 문장부호·기호·이모지
   - 그런 문자로만 완성될 수 있는 UTF-8 조각
   - 바이트 토큰 256개
   - 특수 토큰 15개
2. BPE id는 병합 순서, 즉 대략 빈도순입니다. 그래서 `--keep-top N`으로 **문자 필터를 통과한 토큰 중 id가 낮은 N개만 상주**시킵니다. 기본값 24,576이면 K=24,847(바이트·특수 포함), 상주 head는 38 MiB입니다.
3. **정확성 가드 (`lmhead.rs`).** 절삭된 cold 행 234,993개를 Q8_0으로 인코딩·디코딩한 값으로 1024개 클러스터(k-means)로 묶고, 클러스터마다 중심 `c`와 반경 `ρ`를 저장합니다. 런타임에는 `w·h ≤ c·h + ρ‖h‖`(Cauchy–Schwarz)를 넘는 클러스터만 디스크에서 다시 계산합니다. 결과는 전체 `[V, D]` argmax와 **항상 동일**합니다.
   - logis-center는 바로 이 정확성 보장 문제 때문에 head 절삭을 기각했습니다(`SyntheticData/eval/README.md`).
   - 실측(`raon-quant check`, 임베딩 행 3개를 프로브로 사용): 3개 모두 전체 argmax와 일치했습니다. 재계산한 클러스터는 128–586개, 행은 158k–229k(cold의 67–97%)입니다.
   - **해석.** 1536차원에서는 반경 상한이 느슨해 가지치기 효율이 낮습니다.
   - **후속 과제.** 다음으로 가지치기 효율을 높여야 합니다.
     - 실제 최종 hidden state로 측정하기
     - 방향/노름 분리 클러스터링(spherical k-means + 노름 상한)
     - 클러스터 수 늘리기
     - K 키우기

## 검증

| 항목 | 방법 | 결과 |
|---|---|---|
| 원본 무결성 | HF API의 LFS SHA-256과 `sha256sum -c` | 5/5 OK |
| 블록 레이아웃 (ggml 호환) | `scripts/xcheck_gguf.py`: llama.cpp `gguf-py`의 `GGUFReader`와 `dequantize`로 읽은 값을 원본과 비교 | Q8_0 0.50%, Q6_K 1.75%, Q4_K 7.2%로 자체 보고값과 일치 |
| 융합 커널 정밀도 불변식 | `raon-quant verify` (48×1024, F64 참조) | 최대 오차: Q8_0 2.6e-7, Q6_K 1.6e-7, Q5_K 4.7e-7, Q4_K 6.2e-7 (기준 1e-3) |
| 절삭 head 정확성 | `raon-quant check` | 전체 argmax와 3/3 일치 |
| 오차 단조성 | 단위 테스트 | Q4_K > Q5_K > Q6_K > Q8_0 |

## 재현

```bash
scripts/fetch_models.sh models
```

```bash
cargo build --release -p raon-quant
```

```bash
./target/release/raon-quant nano --model-dir models/KRAFTON/Raon-1-nano --out out/raon-1-nano.pyramid.gguf --dry-run
```

```bash
./target/release/raon-quant nano --model-dir models/KRAFTON/Raon-1-nano --out out/raon-1-nano.pyramid.gguf
```

```bash
./target/release/raon-quant vision --model-dir models/KRAFTON/Raon-VisionEncoder --out out/raon-ve-vision.pyramid.gguf
```

```bash
pip install gguf numpy && python3 -I scripts/xcheck_gguf.py out/raon-ve-vision.pyramid.gguf models/KRAFTON/Raon-VisionEncoder/model.safetensors
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--edge-front` / `--edge-back` | 2 / 2 | T2로 둘 앞·뒤 층 수 |
| `--tier2` / `--tier3` / `--tier4` | `q8_0` / `q6_k` / `q4_k` | 계층별 형식 |
| `--imatrix imatrix.dat` | — | llama.cpp legacy 형식. 키는 원본 텐서 이름 |
| `--keep-top N` | 24576 | `0`이면 문자 필터만 적용 |
| `--no-slice` | — | 절삭 끔 |
| `--guard-clusters` | 1024 | 정확성 가드의 클러스터 수 |

## 알려진 한계

- **imatrix 생성기가 없습니다.** imatrix를 만들려면 모델 순전파가 필요합니다. 리더는 구현되어 있습니다.
- **logis-center의 측정 이력.** 같은 SigLIP2 so400m에 피라미드 플랜을 적용했을 때 문서 번호 크롭 45개 중 2개를 놓쳐 q8 플랜을 기본으로 삼았습니다(`quant.rs:22-31`). 게임 ROI 분류와 그라운딩에 피라미드가 충분한지는 실제 게임 화면으로 평가해야 합니다.
- **NF4 / FP8(E4M3)는 구현하지 않았습니다.** PRD에서 대안 형식으로 언급된 형식입니다.
