"""
Train the classifier used by the Streamlit demo (and for your own experiments).

Trains on ALL plain-text prompts in the built dataset (dev + eval). This model
is for the demo only -- reported numbers always come from the cross-validated
scripts/evaluate.py, never from this model.

Usage:  python scripts/train_final.py --embedder minilm
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from src.classifier import SafetyClassifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="minilm", choices=["minilm", "tfidf"])
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "veilbreak_v2.csv"))
    plain = df[df.variant == "plain"]
    print(f"Training on {len(plain)} plain prompts ({(plain.label == 'unsafe').sum()} unsafe)")
    clf = SafetyClassifier(args.embedder).fit(plain["text"], plain["label"])
    out = os.path.join(ROOT, "models", f"classifier_{args.embedder}.pkl")
    clf.save(out)
    print("Saved", out)


if __name__ == "__main__":
    main()
