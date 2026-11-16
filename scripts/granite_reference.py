"""Hugging Face reference vectors for zena-nlp's granite embedder (crates/zena-nlp/src/granite.rs).

Model: ibm-granite/granite-embedding-97m-multilingual-r2 (ModernBERT encoder, 384-d).
The upstream sentence-transformers pipeline (modules.json on the Hub) is
Transformer -> Pooling(pooling_mode_cls_token=true) -> Normalize, i.e. the L2-normalised hidden state of the first
token ([CLS] = <|startoftext|>). The local model folder only has the Transformer files (no modules.json /
1_Pooling), so this script builds that pipeline explicitly:

* primary reference: AutoTokenizer + AutoModel (float32, CPU, eager attention), last_hidden_state[:, 0], L2 normalise;
* cross-check: SentenceTransformer(modules=[Transformer, Pooling("cls"), Normalize]) on the same folder, in float32.
  transformers 5 would otherwise load the checkpoint's bfloat16 (config.json "dtype"); that stock result is recorded
  too, for comparison only.

Both truncate at MAX_TOKENS = 512 tokens ([CLS] and [SEP] included, right truncation), the same limit as
granite.rs. sentence_bert_config.json says max_seq_length 32768, so for inputs of at most 512 tokens this is exactly
what a float32 sentence-transformers load produces; longer inputs equal `model.max_seq_length = 512`.

Usage (from the repository root):
    python -I scripts/granite_reference.py [MODEL_DIR]
MODEL_DIR defaults to $RAON_GRANITE_DIR, then %LOCALAPPDATA%/logis-center/models/granite-embedding-97m-multilingual-r2.
Writes crates/zena-nlp/tests/data/granite_reference.json.
"""

import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import transformers  # noqa: F401
except ImportError:
    # `python -I` drops the per-user site-packages, which is where transformers/torch are installed on the dev machine.
    import site

    sys.path.append(site.getusersitepackages())

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer

MAX_TOKENS = 512
DIM = 384
SPACE_TAG = "granite-97m-r2-cls-v1"

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "crates" / "zena-nlp" / "tests" / "data" / "granite_reference.json"

MEDIUM_KO = (
    "이번 보스는 체력이 절반 아래로 떨어지면 패턴이 바뀝니다. 처음에는 바닥을 내려찍는 공격을 세 번 하는데, "
    "세 번째 공격 직후에 뒤로 구르면 반격할 틈이 생깁니다. 체력이 절반 아래가 되면 화면 가장자리에서 불기둥이 "
    "올라오니까 가운데에 머물면서 원거리 무기로 약점인 등 뒤의 수정을 노리세요. 회복 아이템은 두 번째 단계에 "
    "아껴 두는 게 좋고, 동료가 쓰러지면 바로 부활시키기보다 보스가 돌진한 다음에 살리는 편이 안전합니다."
)

LONG_MIXED = " ".join(
    [
        "오늘 개발 일지: 아침에 cargo build 를 돌렸더니 wgpu 쪽에서 SurfaceError::Outdated 가 계속 찍혔다.",
        "창 크기를 바꿀 때마다 surface 를 다시 configure 해야 하는데, resize 이벤트를 받기 전에 프레임을 그리려고 해서 생긴 문제였다.",
        "The fix was to skip the frame when the surface is outdated, reconfigure it with the new size, and request another redraw.",
        "그다음에는 아바타의 안경 모델이 머리 회전과 따로 노는 버그를 잡았다. 본 계층에서 안경을 head 본이 아니라 neck 본에 붙여 둔 탓이었다.",
        "After moving the attachment to the head bone, the glasses followed every nod and tilt, and the shadow artifacts on the cheeks disappeared too.",
        "오후에는 로봇 모드 변신 애니메이션을 다듬었다. 팔이 접히는 순서가 어색해서 키프레임을 다시 잡고, 변신 중에는 입력을 막아 두었다.",
        "There is still a one-frame pop when the transformation ends, probably because the blend weights are reset before the final pose is applied.",
        "게임 테스트에서는 같은 구간에서 계속 죽어서 보스 패턴을 기록해 봤다. 내려찍기 세 번, 돌진 한 번, 그리고 체력이 절반 아래면 불기둥이 올라온다.",
        "Players asked for a hint system, so Zena now explains the next attack when someone dies twice in a row in the same room.",
        "음성 인식은 짧은 감탄사를 잘못 알아듣는 경우가 많아서, 와 잡았다 고마워 같은 문장을 명령으로 오해하지 않도록 의도 분류 문턱을 올렸다.",
        "日本語のテストも少し追加した。このボス強すぎる、もう一回やろう、という発話が正しく雑談として分類されるか確認した。",
        "Memory usage on the laptop stayed under two gigabytes with the embedding model loaded, which is good enough for the overlay.",
        "내일은 인벤토리 열기, 설정 메뉴 열기, 볼륨 낮추기 같은 명령어를 한국어와 영어로 섞어서 말해도 같은 의도로 묶이는지 확인할 예정이다.",
        "We also need to measure how long a single sentence takes on the CPU, and whether batching sixteen short spans is worth the extra padding.",
        "마지막으로 로그 파일이 너무 커지는 문제가 있어서 하루 단위로 나누고 일주일이 지난 파일은 압축하도록 바꿨다.",
        "If the shader cache is corrupted after a driver update, the renderer now deletes it and rebuilds the pipelines on the next launch instead of crashing.",
        "보스전 배경 음악이 반복될 때 끊기는 소리가 나서 루프 지점을 샘플 단위로 다시 맞췄고, 효과음 볼륨도 조금 줄였다.",
        "The last task of the day was writing tests for the sentence embedder, comparing every vector against a Hugging Face reference with a cosine threshold.",
    ]
)

SENTENCES = [
    "보스 패턴 어떻게 피해?",
    "또 죽었어",
    "와 잡았다 고마워",
    "wgpu SurfaceError Outdated",
    "로봇 모드로 변신해 줘",
    "안경 써 봐",
    "?",
    "",
    "보스한테 졌어",
    "인벤토리 열어 줘",
    "보스 패턴 알려줘",
    "tell me the boss pattern",
    "このボス強すぎる、もう一回やろう",
    "cargo build 하다가 linker error LNK1181 났어, 어떻게 고쳐?",
    "Can you open the settings menu and lower the volume a bit?",
    MEDIUM_KO,
    LONG_MIXED,
]


def model_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    if os.environ.get("RAON_GRANITE_DIR"):
        return Path(os.environ["RAON_GRANITE_DIR"])
    local = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(local) / "logis-center" / "models" / "granite-embedding-97m-multilingual-r2"


def f32(v: float) -> float:
    """Shortest decimal that round-trips the float32 value (keeps the fixture small)."""
    return float(f"{v:.9g}")


def main() -> None:
    # Windows consoles default to the ANSI code page (cp949 here), which cannot print Japanese.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    d = model_dir()
    if not (d / "model.safetensors").is_file():
        sys.exit(f"granite model folder not found: {d}")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    tok = AutoTokenizer.from_pretrained(str(d))
    model = AutoModel.from_pretrained(str(d), dtype=torch.float32, attn_implementation="eager").eval()
    assert model.config.hidden_size == DIM, model.config.hidden_size
    print(f"transformers {transformers.__version__} | torch {torch.__version__} | {type(model).__name__} | {d}")

    items = []
    vectors = []
    for text in SENTENCES:
        full = tok(text, truncation=False)["input_ids"]
        enc = tok(text, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
        ids = enc["input_ids"]
        assert ids[0, 0].item() == tok.cls_token_id and ids[0, -1].item() == tok.sep_token_id
        out = model(input_ids=ids, attention_mask=enc["attention_mask"]).last_hidden_state
        vec = F.normalize(out[:, 0].float(), p=2, dim=1)[0]
        vectors.append(vec)
        items.append(
            {
                "text": text,
                "n_tokens": int(ids.shape[1]),
                "n_tokens_untruncated": len(full),
                "embedding": [f32(x) for x in vec.tolist()],
            }
        )
        print(f"{len(full):5d} -> {ids.shape[1]:4d} tokens | {text[:40]!r}")
    assert max(i["n_tokens_untruncated"] for i in items) > 600, "the long paragraph must exceed 600 tokens"
    ref = torch.stack(vectors)

    # Cross-check: the same pipeline through sentence-transformers (Transformer -> Pooling(cls) -> Normalize).
    st_check: dict = {}
    try:
        import warnings

        import sentence_transformers

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.models import Normalize, Pooling, Transformer

        st_check = {"sentence_transformers_version": sentence_transformers.__version__}
        # transformers 5 loads with dtype="auto", i.e. the checkpoint's bfloat16 (config.json "dtype"), unless told
        # otherwise. granite.rs computes in float32, so float32 is the reference; the bfloat16 default is recorded
        # only to show how far a stock load is from it.
        for label, kwargs in (("float32", {"dtype": torch.float32}), ("default_dtype", None)):
            tr = Transformer(str(d), max_seq_length=MAX_TOKENS, model_kwargs=kwargs)
            st = SentenceTransformer(modules=[tr, Pooling(DIM, pooling_mode="cls"), Normalize()], device="cpu")
            emb = torch.from_numpy(st.encode(SENTENCES, batch_size=4, convert_to_numpy=True)).float()
            cos = (emb * ref).sum(dim=1)
            dtype = str(next(tr.parameters()).dtype).removeprefix("torch.")
            st_check[label] = {"dtype": dtype, "min_cos": float(cos.min()), "max_abs_diff": float((emb - ref).abs().max())}
            print(
                f"sentence-transformers {sentence_transformers.__version__} ({label} = {dtype}): "
                f"min cos {cos.min():.7f}, max |diff| {(emb - ref).abs().max():.2e}"
            )
    except Exception as e:  # the AutoModel vectors stay the reference either way
        st_check = {"skipped": f"{type(e).__name__}: {e}"}
        print(f"sentence-transformers cross-check skipped: {st_check['skipped']}")

    # Semantic sanity (the Rust test asserts the same orderings).
    by_text = {t: v for t, v in zip(SENTENCES, vectors)}

    def c(a: str, b: str) -> float:
        return float(by_text[a] @ by_text[b])

    print(f"cos(또 죽었어, 보스한테 졌어)     = {c('또 죽었어', '보스한테 졌어'):.4f}")
    print(f"cos(또 죽었어, 인벤토리 열어 줘) = {c('또 죽었어', '인벤토리 열어 줘'):.4f}")
    print(f"cos(보스 패턴 알려줘, tell me the boss pattern) = {c('보스 패턴 알려줘', 'tell me the boss pattern'):.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "model": "ibm-granite/granite-embedding-97m-multilingual-r2",
        "space_tag": SPACE_TAG,
        "pooling": "cls",
        "normalize": "l2",
        "max_tokens": MAX_TOKENS,
        "dim": DIM,
        "reference": "transformers AutoModel float32 CPU eager, last_hidden_state[:, 0], L2-normalised",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "sentence_transformers_check": st_check,
        "items": items,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(items)} sentences)")


if __name__ == "__main__":
    main()
