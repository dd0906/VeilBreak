"""
Sweep the ROT13 thresholds on the DEV split only (never the eval split).

For each (min_gain, min_score) pair we report:
  rot13 recall   = share of dev ROT13 prompts that get decoded correctly
  false triggers = share of ~5,200 benign dev prompts wrongly ROT13-decoded
                   (benign pool = Alpaca rows with index % 10 == 0)

A good setting sits in a FLAT region of the table (small changes do not
matter) -- that is what you point to when asked "why these numbers?".

Usage:  python scripts/tune_thresholds.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import src.pipeline as P  # noqa: E402


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "veilbreak_v2.csv"))
    dev = df[(df.split == "dev") & (df.variant == "rot13")]
    alp = pd.read_csv(os.path.join(ROOT, "data", "raw", "alpaca.csv"))
    benign = [str(t) for i, t in enumerate(alp["prompt"]) if i % 10 == 0]

    print(f"dev ROT13 prompts: {len(dev)}   benign dev prompts: {len(benign)}\n")
    print(f"{'min_gain':>8} {'min_score':>9} {'rot13 recall':>13} {'false triggers':>15}")
    for gain in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5):
        for score in (-3.4, -3.0, -2.6):
            P.ROT13_MIN_GAIN, P.ROT13_MIN_SCORE = gain, score
            hit = sum(P.decode(r.text).decoded.lower().split() == str(r.expected).lower().split()
                      for r in dev.itertuples())
            fp = sum("rot13" in P.decode(t).encodings for t in benign)
            mark = "   <-- chosen" if (gain, score) == (0.6, -3.0) else ""
            print(f"{gain:>8} {score:>9} {hit / len(dev):>12.1%} "
                  f"{fp / len(benign):>14.2%}{mark}")


if __name__ == "__main__":
    main()
