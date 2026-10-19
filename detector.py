"""
Level 2: Multimodal Verification Network (parallel detectors)
Uses frozen Hugging Face pretrained models as feature extractors (CPU friendly)
and trains small classifiers on top.  Label convention: fake = 1, real = 0.

Usage:
    python detectors.py visual
    python detectors.py audio
    python detectors.py text
    python detectors.py lavdf     (needs visual + audio models first)
    python detectors.py all
"""
import os, re, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch, joblib, librosa
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from transformers import (CLIPModel, CLIPProcessor, Wav2Vec2Model,
                          Wav2Vec2FeatureExtractor, AutoTokenizer, AutoModel)

# ================= CONFIG =================
ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "clean"
MODELS = ROOT / "models"
SCORES = ROOT / "data" / "scores"
MODELS.mkdir(exist_ok=True)
SCORES.mkdir(parents=True, exist_ok=True)

CLIP_NAME = "openai/clip-vit-base-patch32"
W2V_NAME = "facebook/wav2vec2-base"
TEXT_NAME = "distilbert-base-uncased"
SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count() or 4)
print("Device:", DEVICE)

_cache = {}


# ================= EMBEDDING HELPERS =================
def _clip():
    if "clip" not in _cache:
        _cache["clip"] = (CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE).eval(),
                          CLIPProcessor.from_pretrained(CLIP_NAME))
    return _cache["clip"]


@torch.no_grad()
def clip_embed(paths, bs=32, desc=None):
    m, proc = _clip()
    out = []
    rng = range(0, len(paths), bs)
    for i in (tqdm(rng, desc=desc) if desc else rng):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        x = proc(images=imgs, return_tensors="pt").to(DEVICE)
        f = m.get_image_features(**x)
        if not torch.is_tensor(f):
            f = getattr(f, "pooler_output", f[0])
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out)


def _w2v():
    if "w2v" not in _cache:
        _cache["w2v"] = (Wav2Vec2FeatureExtractor.from_pretrained(W2V_NAME),
                         Wav2Vec2Model.from_pretrained(W2V_NAME).to(DEVICE).eval())
    return _cache["w2v"]


@torch.no_grad()
def w2v_embed(waves, total=None, desc=None):
    """waves: iterable of 1-D float arrays at 16 kHz (lazy generators are fine)."""
    fe, m = _w2v()
    out = []
    for y in tqdm(waves, total=total, desc=desc):
        x = fe(y, sampling_rate=SR, return_tensors="pt").input_values.to(DEVICE)
        h = m(x).last_hidden_state
        out.append(torch.cat([h.mean(1), h.std(1)], dim=-1).squeeze(0).cpu().numpy())
    return np.stack(out)


@torch.no_grad()
def text_embed(texts, bs=32):
    if "txt" not in _cache:
        _cache["txt"] = (AutoTokenizer.from_pretrained(TEXT_NAME),
                         AutoModel.from_pretrained(TEXT_NAME).to(DEVICE).eval())
    tok, m = _cache["txt"]
    out = []
    for i in tqdm(range(0, len(texts), bs), desc="text"):
        b = tok(texts[i:i + bs], padding=True, truncation=True,
                max_length=128, return_tensors="pt").to(DEVICE)
        h = m(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.concatenate(out)


def load_wav(p):
    y, _ = librosa.load(str(p), sr=SR, mono=True)
    return y


# ================= TRAIN + EVALUATE =================
def fit_eval(X, y, name):
    y = np.asarray(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    print(f"[{name}] test accuracy = {accuracy_score(yte, p > 0.5):.3f} | "
          f"AUC = {roc_auc_score(yte, p):.3f}  (n_train={len(ytr)}, n_test={len(yte)})")
    clf.fit(X, y)                      # refit on all data for later scoring
    joblib.dump(clf, MODELS / f"{name}.joblib")
    return clf


# ================= 1. VISUAL FORGERY DETECTOR =================
def train_visual():
    print("\n=== Visual Forgeries Detector (CLIP + LogReg) ===")
    paths, y = [], []
    for lab, v in [("real", 0), ("fake", 1)]:
        ps = sorted((CLEAN / "faces" / lab).glob("*.jpg"))
        paths += ps
        y += [v] * len(ps)
    if not paths:
        print("No face images found in data/clean/faces"); return
    X = clip_embed(paths, desc="faces")
    fit_eval(X, y, "visual")


# ================= 2. AUDIO FORGERY DETECTOR =================
def train_audio():
    print("\n=== Audio Forgeries Detector (wav2vec2 + LogReg) ===")
    paths, y = [], []
    for lab, v in [("real", 0), ("fake", 1)]:
        ps = sorted((CLEAN / "audio" / lab).glob("*.wav"))
        paths += ps
        y += [v] * len(ps)
    if not paths:
        print("No audio found in data/clean/audio"); return
    X = w2v_embed((load_wav(p) for p in paths), total=len(paths), desc="audio")
    fit_eval(X, y, "audio")


# ================= 3. CLAIM & INTENT VERIFIER =================
def train_text():
    print("\n=== Claim & Intent Verifier (DistilBERT + LogReg) ===")
    tdir = CLEAN / "text"
    f = tdir / "liar_clean.csv"
    if not f.exists():
        f = tdir / "isot_clean.csv"
    if not f.exists():
        print("No liar_clean.csv or isot_clean.csv in data/clean/text"); return
    df = pd.read_csv(f).dropna(subset=["text", "label"])
    df["label"] = 1 - df["label"].astype(int)   # cleaned files use true=1; flip so fake=1
    print("Using:", f.name, df.shape)
    X = text_embed(df["text"].astype(str).tolist())
    fit_eval(X, df["label"].values, "text")


# ================= 4. SPLICE DETECTOR + LAV-DF SCORING =================
def _rms_at(y, t, w=0.25):
    seg = y[int(max(0, t - w) * SR): int((t + w) * SR)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0


def _safe_corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _audio_windows(y, win=2 * SR):
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    return [y[i:i + win] for i in range(0, len(y) - win + 1, win)] or [y[:win]]


def score_lavdf():
    print("\n=== LAV-DF scoring: visual, audio and splice features per video ===")
    for need in ["visual", "audio"]:
        if not (MODELS / f"{need}.joblib").exists():
            print(f"Missing models/{need}.joblib - run 'python detectors.py {need}' first")
            return
    vis = joblib.load(MODELS / "visual.joblib")
    aud = joblib.load(MODELS / "audio.joblib")
    meta = pd.read_csv(CLEAN / "lavdf" / "clean_metadata.csv")
    fdir, adir = CLEAN / "lavdf" / "frames", CLEAN / "lavdf" / "audio"
    pat = re.compile(r"_f(\d+)_t([\d.]+)\.jpg$")

    rows = []
    for _, r in tqdm(meta.iterrows(), total=len(meta), desc="lavdf"):
        vid = str(r["video"])
        items = []
        for p in fdir.glob(f"{vid}_f*_t*.jpg"):
            mt = pat.search(p.name)
            if mt:
                items.append((int(mt.group(1)), float(mt.group(2)), p))
        items.sort()
        wav = adir / f"{vid}.wav"
        if len(items) < 4 or not wav.exists():
            continue
        paths = [it[2] for it in items]
        times = np.array([it[1] for it in items])

        # --- visual score (per-frame fake probability) ---
        E = clip_embed(paths)
        vp = vis.predict_proba(E)[:, 1]

        # --- audio score (per 2-second window) ---
        y = load_wav(wav)
        ap = aud.predict_proba(w2v_embed(_audio_windows(y)))[:, 1]

        # --- splice features: visual jumps + audio/motion agreement ---
        cos_d = 1 - np.sum(E[1:] * E[:-1], axis=1)
        g = [np.asarray(Image.open(p).convert("L").resize((64, 64)), dtype=np.float32)
             for p in paths]
        motion = np.array([np.abs(g[i + 1] - g[i]).mean() for i in range(len(g) - 1)])
        rms = np.array([_rms_at(y, t) for t in times])
        rms_change = np.abs(np.diff(rms))

        rows.append({
            "video": vid,
            "label": 1 if r["label"] == "fake" else 0,
            "visual_mean": vp.mean(), "visual_max": vp.max(),
            "audio_mean": ap.mean(), "audio_max": ap.max(),
            "emb_jump_mean": cos_d.mean(), "emb_jump_max": cos_d.max(),
            "emb_jump_std": cos_d.std(),
            "emb_jump_ratio": cos_d.max() / (cos_d.mean() + 1e-6),
            "motion_mean": motion.mean(), "motion_max": motion.max(),
            "audio_rms_std": rms.std(),
            "av_corr": _safe_corr(motion, rms_change),
        })

    df = pd.DataFrame(rows)
    if len(df) < 20 or df["label"].nunique() < 2:
        print("Not enough videos scored:", len(df)); return

    # How well do the transferred detectors do on LAV-DF?
    for c in ["visual_mean", "audio_mean"]:
        print(f"  AUC of {c} on LAV-DF: {roc_auc_score(df['label'], df[c]):.3f} "
              "(0.5 = no better than chance)")

    # Splice model with cross-validated predictions (no leakage into fusion table)
    feats = ["emb_jump_mean", "emb_jump_max", "emb_jump_std", "emb_jump_ratio",
             "motion_mean", "motion_max", "audio_rms_std", "av_corr"]
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=42)
    df["splice_score"] = cross_val_predict(gb, df[feats], df["label"], cv=5,
                                           method="predict_proba")[:, 1]
    print(f"[splice] CV accuracy = {accuracy_score(df['label'], df['splice_score'] > 0.5):.3f} | "
          f"AUC = {roc_auc_score(df['label'], df['splice_score']):.3f}")
    gb.fit(df[feats], df["label"])
    joblib.dump(gb, MODELS / "splice.joblib")

    out = SCORES / "lavdf_scores.csv"
    df.to_csv(out, index=False)
    print("Saved:", out, df.shape)


# ================= MAIN =================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["visual", "audio", "text", "lavdf", "all"])
    step = ap.parse_args().step
    if step in ("visual", "all"): train_visual()
    if step in ("audio", "all"): train_audio()
    if step in ("text", "all"): train_text()
    if step in ("lavdf", "all"): score_lavdf()
    print("\nDone. Models are in:", MODELS, "| scores in:", SCORES)