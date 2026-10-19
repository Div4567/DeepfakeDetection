import os, re, json, hashlib, random, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import cv2, librosa, soundfile as sf, noisereduce as nr
from tqdm import tqdm
import imageio_ffmpeg

# ================= CONFIG =================
PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "data" / "clean"

N_FACES_PER_CLASS = 2000
N_AUDIO_PER_CLASS = 2000
N_FAKEDDIT_PER_CLASS = 2500
N_TEXT_MAX = 6000
N_FRAMES_PER_VIDEO = 8
SR = 16000
IMG_EXT = (".jpg", ".jpeg", ".png")

# --- LAV-DF speed settings ---
N_LAVDF_VIDEOS = 3000       # total videos to process (half real, half fake)
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)   # leave one core free
DENOISE_AUDIO = True        # set False for maximum speed

random.seed(42); np.random.seed(42)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

for d in ["faces/real", "faces/fake", "audio/real", "audio/fake",
          "lavdf/frames", "lavdf/audio", "lavdf/_done", "text"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)


def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s.,!?'\"-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def under(p, *names):
    parts = [x.lower() for x in p.parts]
    return any(n.lower() in parts for n in names)


# ================= 1. FACES =================
def clean_faces():
    print("\n[1/5] Cleaning face images")
    buckets = {"real": [], "fake": []}
    for p in RAW.rglob("*"):
        if p.suffix.lower() not in IMG_EXT: continue
        if under(p, "for-2sec", "LAV-DF"): continue
        label = p.parent.name.lower()           # folder directly holding the image
        if label in buckets:
            buckets[label].append(p)
    for label, paths in buckets.items():
        # prefer the training split when it exists
        train = [p for p in paths if under(p, "train", "training")]
        paths = train if train else paths
        random.shuffle(paths)
        seen, kept, bad = set(), 0, 0
        for p in tqdm(paths, desc=f"faces/{label}"):
            if kept >= N_FACES_PER_CLASS: break
            img = cv2.imread(str(p))
            if img is None: bad += 1; continue
            h = hashlib.md5(img.tobytes()).hexdigest()
            if h in seen: continue
            seen.add(h)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(OUT / "faces" / label / f"{label}_{kept:05d}.jpg"),
                        img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            kept += 1
        print(f"  {label}: found {len(paths)}, kept {kept}, corrupt {bad}")


# ================= 2. FoR AUDIO =================
def clean_audio():
    print("\n[2/5] Cleaning FoR audio")
    root = next((p for p in RAW.iterdir() if p.is_dir() and "for" in p.name.lower()), None)
    if root is None: print("  for-2sec folder not found"); return
    for label in ["real", "fake"]:
        files = [p for p in root.rglob("*.wav") if p.parent.name.lower() == label]
        train = [p for p in files if under(p, "training", "train")]
        files = train if train else files
        random.shuffle(files); kept = 0
        for f in tqdm(files, desc=f"audio/{label}"):
            if kept >= N_AUDIO_PER_CLASS: break
            try:
                y, _ = librosa.load(str(f), sr=SR, mono=True)
            except Exception:
                continue
            if len(y) < SR * 0.5 or np.max(np.abs(y)) < 1e-3: continue
            y = y / (np.max(np.abs(y)) + 1e-9) * 0.95
            sf.write(str(OUT / "audio" / label / f"{label}_{kept:05d}.wav"), y, SR)
            kept += 1
        print(f"  {label}: found {len(files)}, kept {kept}")


# ================= 3. LAV-DF VIDEO (fast, parallel, resumable) =================
def _process_video(args):
    vp, label, info = args
    vp = Path(vp)
    cv2.setNumThreads(1)                      # avoid CPU oversubscription across workers
    vid = vp.stem
    wav = OUT / "lavdf/audio" / f"{vid}.wav"
    done_flag = OUT / "lavdf/_done" / f"{vid}.done"
    try:
        # ---- skip if already processed (resumable) ----
        if done_flag.exists():
            return {"video": vid, "label": label, **info}

        # ---- frames: read sequentially, keep only the wanted indices ----
        cap = cv2.VideoCapture(str(vp))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        if n < 1:
            cap.release(); return None
        wanted = set(np.linspace(0, n - 1, N_FRAMES_PER_VIDEO).astype(int).tolist())
        last = max(wanted)
        k = 0
        for idx in range(n):
            if not cap.grab():
                break
            if idx in wanted:
                ok, fr = cap.retrieve()
                if ok:
                    fr = cv2.resize(fr, (320, 320), interpolation=cv2.INTER_AREA)
                    fr = cv2.bilateralFilter(fr, 5, 40, 40)      # fast denoise
                    cv2.imwrite(str(OUT / "lavdf/frames" / f"{vid}_f{k}_t{idx/fps:.2f}.jpg"),
                                fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    k += 1
            if idx >= last:
                break                                           # stop reading early
        cap.release()

        # ---- audio: ffmpeg extract (+ fast stationary denoise) ----
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-threads", "1",
                        "-i", str(vp), "-vn", "-ac", "1", "-ar", str(SR), str(wav)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if DENOISE_AUDIO and wav.exists():
            y, _ = sf.read(str(wav), dtype="float32")
            if len(y) > SR // 2:
                sf.write(str(wav), nr.reduce_noise(y=y, sr=SR, stationary=True,
                                                   n_jobs=1), SR)

        done_flag.touch()
        return {"video": vid, "label": label, **info, "duration_s": round(n / fps, 2)}
    except Exception as e:
        return {"video": vid, "label": label, "error": str(e)}


def clean_lavdf():
    print("\n[3/5] Cleaning LAV-DF videos")
    root = next((p for p in RAW.iterdir() if p.is_dir() and "lav" in p.name.lower()), None)
    if root is None: print("  LAV-DF folder not found"); return
    metas = list(root.rglob("metadata.json"))
    if not metas: print("  metadata.json not found"); return
    meta = pd.DataFrame(json.load(open(metas[0])))
    videos = {p.name: p for p in root.rglob("*.mp4")}
    print("  mp4 files on disk:", len(videos))
    meta["name"] = meta["file"].astype(str).str.replace("\\", "/").str.split("/").str[-1]
    sub = meta[meta["name"].isin(videos)].copy()
    sub["label"] = np.where(sub["n_fakes"] > 0, "fake", "real")

    # balanced subset
    per = N_LAVDF_VIDEOS // 2
    sub = pd.concat([g.sample(min(per, len(g)), random_state=42)
                     for _, g in sub.groupby("label")])
    print("  videos used:", len(sub), sub["label"].value_counts().to_dict(),
          "| workers:", N_WORKERS)

    jobs = []
    for _, r in sub.iterrows():
        info = {"modify_video": r.get("modify_video"),
                "modify_audio": r.get("modify_audio"),
                "fake_periods": json.dumps(r.get("fake_periods", []))}
        jobs.append((str(videos[r["name"]]), r["label"], info))

    rows, errors = [], 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(_process_video, j) for j in jobs]
        for f in tqdm(as_completed(futures), total=len(futures), desc="lavdf"):
            res = f.result()
            if res is None: continue
            if "error" in res: errors += 1; continue
            rows.append(res)

    out_csv = OUT / "lavdf/clean_metadata.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  done: {len(rows)} videos, {errors} errors -> {out_csv}")


# ================= 4. LIAR / ISOT TEXT =================
LIAR_COLS = ["id", "label", "statement", "subject", "speaker", "job", "state",
             "party", "c1", "c2", "c3", "c4", "c5", "context"]
LIAR_MAP = {"pants-fire": 0, "false": 0, "barely-true": 0,
            "half-true": 1, "mostly-true": 1, "true": 1}


def clean_text_datasets():
    print("\n[4/5] Cleaning text (ISOT / LIAR)")
    found = False
    fake_csv = next(RAW.rglob("Fake.csv"), None)
    true_csv = next(RAW.rglob("True.csv"), None)
    if fake_csv and true_csv:
        f = pd.read_csv(fake_csv).assign(label=0)
        t = pd.read_csv(true_csv).assign(label=1)
        d = pd.concat([f, t])
        d["text"] = (d["title"].fillna("") + " " + d["text"].fillna("")).map(clean_text)
        d = d[d["text"].str.len() > 20].drop_duplicates("text")
        d = d.sample(frac=1, random_state=42).head(N_TEXT_MAX)
        d[["text", "label"]].to_csv(OUT / "text/isot_clean.csv", index=False)
        print("  ISOT:", d.shape, d["label"].value_counts().to_dict()); found = True
    liar = next((p for p in RAW.rglob("*") if p.name.lower() in ("train.tsv", "liar_train.tsv")), None)
    if liar and not under(liar, "LAV-DF"):
        d = pd.read_csv(liar, sep="\t", header=None, names=LIAR_COLS)
        d["label"] = d["label"].map(LIAR_MAP)
        d["text"] = d["statement"].map(clean_text)
        d = d.dropna(subset=["label", "text"]).drop_duplicates("text")
        d["label"] = d["label"].astype(int)
        d[["text", "label"]].to_csv(OUT / "text/liar_clean.csv", index=False)
        print("  LIAR:", d.shape, d["label"].value_counts().to_dict()); found = True
    if not found: print("  no ISOT (Fake.csv/True.csv) or LIAR (train.tsv) found")


# ================= 5. FAKEDDIT =================
def clean_fakeddit():
    print("\n[5/5] Cleaning Fakeddit")
    tsv = RAW / "all_train.tsv"
    if not tsv.exists():
        tsv = next(RAW.rglob("all_train.tsv"), None)
    if tsv is None: print("  all_train.tsv not found"); return
    df = pd.read_csv(tsv, sep="\t")
    df = df.dropna(subset=["clean_title", "image_url", "2_way_label"])
    df = df.drop_duplicates("id")
    df = df[df["image_url"].astype(str).str.startswith("http")]
    df = pd.concat([g.sample(min(N_FAKEDDIT_PER_CLASS, len(g)), random_state=42)
                    for _, g in df.groupby("2_way_label")])
    df["text"] = df["clean_title"].map(clean_text)
    df[["id", "text", "image_url", "2_way_label"]].to_csv(OUT / "text/fakeddit_clean.csv", index=False)
    print("  Fakeddit:", df.shape, df["2_way_label"].value_counts().to_dict())


# ================= MAIN =================
if __name__ == "__main__":
    print("RAW folder:", RAW, "| exists:", RAW.exists())
    for p in sorted(RAW.iterdir()): print("  -", p.name)
    clean_faces()
    clean_audio()
    clean_lavdf()
    clean_text_datasets()
    clean_fakeddit()
    print("\nDone. Clean data is in:", OUT)