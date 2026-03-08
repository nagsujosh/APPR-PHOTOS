#!/usr/bin/env python
"""Dataset preparation, download instructions, and verification.

Usage:
    # Verify what is available
    python scripts/prepare_datasets.py --verify

    # Generate metadata.csv for MDER-MA after extracting the zip
    python scripts/prepare_datasets.py --build-mderma --root data/raw/mderma

    # Generate metadata.csv for TAME after extracting PhysioNet download
    python scripts/prepare_datasets.py --build-tame --root data/raw/tame

    # Full stats for all available datasets
    python scripts/prepare_datasets.py --stats
"""
import argparse
import csv
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

CREMAD_EMOTION_MAP = {"ANG": "anger", "DIS": "disgust", "FEA": "fear",
                      "HAP": "happy", "NEU": "neutral", "SAD": "sad"}


def verify_cremad(root: Path) -> dict:
    audio_dir = root / "AudioWAV"
    if not audio_dir.exists():
        return {"status": "MISSING", "path": str(root)}
    wavs = list(audio_dir.glob("*.wav"))
    demo = (root / "VideoDemographics.csv").exists()
    return {
        "status": "OK",
        "path": str(root),
        "num_files": len(wavs),
        "has_demographics": demo,
        "note": "" if demo else "VideoDemographics.csv missing — gender/age will be -1",
    }


def verify_mderma(root: Path) -> dict:
    if not root.exists():
        return {"status": "MISSING", "path": str(root)}
    wavs = list(root.rglob("*.wav"))
    meta = (root / "metadata.csv").exists()
    return {
        "status": "OK" if wavs else "EMPTY",
        "path": str(root),
        "num_files": len(wavs),
        "has_metadata_csv": meta,
    }


def verify_tame(root: Path) -> dict:
    if not root.exists():
        return {"status": "MISSING", "path": str(root)}
    wavs = list(root.rglob("*.wav"))
    meta = (root / "metadata.csv").exists()
    records = (root / "RECORDS").exists()
    return {
        "status": "OK" if wavs else "EMPTY",
        "path": str(root),
        "num_files": len(wavs),
        "has_metadata_csv": meta,
        "has_records": records,
    }


# ------------------------------------------------------------------ #
# MDER-MA metadata builder                                            #
# ------------------------------------------------------------------ #

MDERMA_EMOTION_MAP = {"angry": "angry", "anger": "angry",
                      "happy": "happy", "happiness": "happy",
                      "neutral": "neutral",
                      "sad": "sad", "sadness": "sad"}


def build_mderma_metadata(root: Path) -> None:
    """Scan directory structure and write metadata.csv."""
    rows = []
    seen = set()

    # Try emotion-folder layout
    for emotion_dir in sorted(root.iterdir()):
        if not emotion_dir.is_dir():
            continue
        ename = emotion_dir.name.strip().lower()
        if ename not in MDERMA_EMOTION_MAP:
            continue
        emotion = MDERMA_EMOTION_MAP[ename]
        for wav in sorted(emotion_dir.rglob("*.wav")):
            rel = wav.relative_to(root)
            if str(rel) in seen:
                continue
            seen.add(str(rel))
            # Infer speaker: subfolder name or filename prefix
            if wav.parent != emotion_dir:
                spk = wav.parent.name
            else:
                spk = wav.stem.replace("-", "_").split("_")[0]
            rows.append({
                "filename": str(rel),
                "emotion": emotion,
                "speaker_id": spk,
                "gender": "",
            })

    # Try speaker-folder layout if above found nothing
    if not rows:
        for spk_dir in sorted(root.iterdir()):
            if not spk_dir.is_dir():
                continue
            for emotion_dir in sorted(spk_dir.iterdir()):
                if not emotion_dir.is_dir():
                    continue
                ename = emotion_dir.name.strip().lower()
                if ename not in MDERMA_EMOTION_MAP:
                    continue
                emotion = MDERMA_EMOTION_MAP[ename]
                for wav in sorted(emotion_dir.glob("*.wav")):
                    rel = wav.relative_to(root)
                    rows.append({
                        "filename": str(rel),
                        "emotion": emotion,
                        "speaker_id": spk_dir.name,
                        "gender": "",
                    })

    if not rows:
        print("  [WARN] No audio files found — check extraction path.")
        return

    out = root / "metadata.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "emotion", "speaker_id", "gender"])
        writer.writeheader()
        writer.writerows(rows)

    emotion_counts = Counter(r["emotion"] for r in rows)
    print(f"  Written {len(rows)} rows to {out}")
    print(f"  Emotions: {dict(emotion_counts)}")
    print(f"  Speakers: {len(set(r['speaker_id'] for r in rows))}")


# ------------------------------------------------------------------ #
# TAME metadata builder                                               #
# ------------------------------------------------------------------ #

def build_tame_metadata(root: Path) -> None:
    """Scan PhysioNet structure and write metadata.csv with best-effort annotations."""
    import re

    rows = []

    # Check for RECORDS file
    records_file = root / "RECORDS"
    if records_file.exists():
        lines = records_file.read_text().strip().splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            wav = root / (line if line.endswith(".wav") else line + ".wav")
            if not wav.exists():
                continue
            # Try sidecar annotation
            pain = 0
            txt = wav.with_suffix(".txt")
            if txt.exists():
                try:
                    pain = int(float(txt.read_text().strip().split()[0]))
                except (ValueError, IndexError):
                    pass
            # Subject ID from path
            parts = wav.relative_to(root).parts
            spk = parts[0] if len(parts) > 1 else wav.stem.split("_")[0]
            rows.append({
                "filename": str(wav.relative_to(root)),
                "pain_level": pain,
                "speaker_id": spk,
                "gender": "",
                "age": "",
            })
    else:
        # Subject folder scan
        subject_dirs = sorted(
            d for d in root.iterdir()
            if d.is_dir() and re.match(r"(subject|sub|spk|s)[-_]?\d+", d.name, re.IGNORECASE)
        )
        if not subject_dirs:
            subject_dirs = [root]

        for subj_dir in subject_dirs:
            spk = subj_dir.name
            for wav in sorted(subj_dir.rglob("*.wav")):
                pain = 0
                txt = wav.with_suffix(".txt")
                if txt.exists():
                    try:
                        pain = int(float(txt.read_text().strip().split()[0]))
                    except (ValueError, IndexError):
                        pass
                rows.append({
                    "filename": str(wav.relative_to(root)),
                    "pain_level": pain,
                    "speaker_id": spk,
                    "gender": "",
                    "age": "",
                })

    if not rows:
        print("  [WARN] No audio files found — check extraction path.")
        return

    out = root / "metadata.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "pain_level", "speaker_id", "gender", "age"]
        )
        writer.writeheader()
        writer.writerows(rows)

    pain_counts = Counter(r["pain_level"] for r in rows)
    print(f"  Written {len(rows)} rows to {out}")
    print(f"  Pain levels found: {dict(sorted(pain_counts.items()))}")
    print(f"  Speakers: {len(set(r['speaker_id'] for r in rows))}")
    print("  NOTE: Pain annotations default to 0 if no sidecar .txt found.")
    print("        Edit metadata.csv manually or from PhysioNet annotation files.")


# ------------------------------------------------------------------ #
# Dataset stats                                                        #
# ------------------------------------------------------------------ #

def print_stats():
    try:
        from aapr.data.cremad import CremaDDataset
        ds = CremaDDataset("data/raw/cremad")
        spk_ids = ds.get_speaker_ids()
        print(f"CREMA-D: {len(ds)} samples | {ds.num_speakers} speakers")
        print(f"  Labels: {Counter(ds[i]['utility_label'] for i in range(len(ds)))}")
        print(f"  Gender missing: {sum(1 for i in range(len(ds)) if ds[i]['privacy_labels']['gender'] == -1)}")
    except Exception as e:
        print(f"CREMA-D: could not load — {e}")

    try:
        from aapr.data.mderma import MDERMADataset
        ds = MDERMADataset("data/raw/mderma")
        print(f"MDER-MA: {len(ds)} samples | {ds.num_speakers} speakers")
        print(f"  Labels: {Counter(ds[i]['utility_label'] for i in range(len(ds)))}")
    except Exception as e:
        print(f"MDER-MA: could not load — {e}")

    try:
        from aapr.data.tame import TAMEDataset
        ds = TAMEDataset("data/raw/tame")
        print(f"TAME:    {len(ds)} samples | {ds.num_speakers} speakers")
        print(f"  Pain bins: {Counter(ds[i]['utility_label'] for i in range(len(ds)))}")
    except Exception as e:
        print(f"TAME:    could not load — {e}")

    try:
        from aapr.data.combined import CombinedEmotionDataset
        ds = CombinedEmotionDataset("data/raw/cremad", "data/raw/mderma")
        st = ds.stats()
        print(f"Combined: {st}")
    except Exception as e:
        print(f"Combined: could not load — {e}")


# ------------------------------------------------------------------ #
# Download instructions                                               #
# ------------------------------------------------------------------ #

DOWNLOAD_INSTRUCTIONS = """
=== Dataset Download Instructions ===

--- CREMA-D (already available at data/raw/cremad/) ---
  7442 WAV clips, 91 actors, 6 emotions.
  Kaggle: kaggle datasets download -d ejlok1/cremad
  Demographics (optional, for gender labels):
    Download VideoDemographics.csv from the same Kaggle page and place in
    data/raw/cremad/VideoDemographics.csv

--- MDER-MA ---
  5288 clips, Moroccan Arabic, 4 emotions (angry/happy/neutral/sad).
  1. Go to: https://www.sciencedirect.com/science/article/pii/S2352340925007292
  2. Download from Mendeley Data (link in the article)
  3. Extract to: data/raw/mderma/
  4. Run: python scripts/prepare_datasets.py --build-mderma --root data/raw/mderma
     (generates metadata.csv from the directory structure)

--- TAME Pain ---
  7039 utterances, 51 speakers, pain levels 0-10.
  Requires PhysioNet credentialed account.
  1. Request access at: https://doi.org/10.13026/20e2-1g10
  2. Download with wget after authentication:
       wget -r -N -c -np --user <username> --ask-password \\
            https://physionet.org/files/tame-pain/1.0.0/
  3. Extract to: data/raw/tame/
  4. Run: python scripts/prepare_datasets.py --build-tame --root data/raw/tame
"""


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Dataset preparation and verification")
    parser.add_argument("--verify", action="store_true", help="Check what datasets are available")
    parser.add_argument("--stats", action="store_true", help="Load and print dataset statistics")
    parser.add_argument("--build-mderma", action="store_true", help="Build metadata.csv for MDER-MA")
    parser.add_argument("--build-tame", action="store_true", help="Build metadata.csv for TAME")
    parser.add_argument("--root", type=str, default=None, help="Root path for --build-* commands")
    parser.add_argument("--instructions", action="store_true", help="Print download instructions")
    args = parser.parse_args()

    if args.instructions:
        print(DOWNLOAD_INSTRUCTIONS)
        return

    if args.verify:
        print("=== Dataset Verification ===")
        r = verify_cremad(Path("data/raw/cremad"))
        print(f"CREMA-D : {r}")
        r = verify_mderma(Path("data/raw/mderma"))
        print(f"MDER-MA : {r}")
        r = verify_tame(Path("data/raw/tame"))
        print(f"TAME    : {r}")
        return

    if args.build_mderma:
        root = Path(args.root or "data/raw/mderma")
        print(f"Building MDER-MA metadata from {root} ...")
        build_mderma_metadata(root)
        return

    if args.build_tame:
        root = Path(args.root or "data/raw/tame")
        print(f"Building TAME metadata from {root} ...")
        build_tame_metadata(root)
        return

    if args.stats:
        print_stats()
        return

    # Default: show everything
    print("=== Dataset Verification ===")
    print(f"CREMA-D : {verify_cremad(Path('data/raw/cremad'))}")
    print(f"MDER-MA : {verify_mderma(Path('data/raw/mderma'))}")
    print(f"TAME    : {verify_tame(Path('data/raw/tame'))}")
    print()
    print("Run with --instructions to see download steps.")
    print("Run with --stats to load and count samples.")


if __name__ == "__main__":
    main()
