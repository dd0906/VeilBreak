"""
Build the VeilBreak evaluation dataset (v2).

What is different from a naive "obfuscate the bad prompts" dataset:

1. GROUPS. Every original prompt gets one `group_id`. All its variants
   (plain, base64, rot13, ...) share that id, and evaluation splits by
   group, so the same sentence never appears in both train and test.
2. BENIGN prompts are obfuscated too. Otherwise "looks encoded" would
   perfectly predict "unsafe" and a classifier could cheat.
3. STYLE-MATCHED benign prompts. AdvBench is all imperative instructions
   ("Write a script that..."). Half of the benign prompts are also
   imperative, so the classifier cannot win on writing style alone.
   Benign prompts are also length-matched to the harmful ones.
4. DEV / EVAL split (by hash of the prompt). Thresholds are tuned on `dev`;
   all reported numbers come from `eval`.
5. Held-out language-model data: benign prompts come only from Alpaca rows
   with index % 5 == 0 (the language model is fit on the other 80%).

Variants (9):
  plain | base64 | rot13 | leetspeak          <- in scope
  composed (base64 of rot13) | embedded_base64   <- in scope, harder
  hex | reverse | caesar3                       <- OUT of scope: tests safe failure

Usage:  python scripts/build_dataset.py [--seed 42]
"""
import argparse
import hashlib
import os
import random
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from src import encoders as E  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed", "veilbreak_v2.csv")

IMPERATIVE = ("write", "create", "develop", "generate", "provide", "explain",
              "describe", "give", "design", "teach", "outline", "compose",
              "produce", "demonstrate", "instruct", "show", "list", "make",
              "draft", "suggest", "tell")

IN_SCOPE = ["plain", "base64", "rot13", "leetspeak", "composed", "embedded_base64"]
OUT_OF_SCOPE = ["hex", "reverse", "caesar3"]


def norm(t):
    return " ".join(str(t).split())


def bucket(text, mod):
    """Stable pseudo-random bucket from the text (same on every machine)."""
    return int(hashlib.md5(text.lower().encode()).hexdigest(), 16) % mod


def load_prompts(path):
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    col = "prompt" if "prompt" in df.columns else df.columns[0]
    return [norm(t) for t in df[col].dropna()]


def make_variants(text, rng):
    """Return {variant: (obfuscated_text, expected_decoded_text, expected_steps)}."""
    emb, _ = E.embed_base64(text, rng)
    emb_expected = emb.replace(E.encode_base64(text), text)
    return {
        "plain": (text, text, ""),
        "base64": (E.encode_base64(text), text, "base64"),
        "rot13": (E.encode_rot13(text), text, "rot13"),
        "leetspeak": (E.encode_leetspeak(text, rng.choice([0.6, 0.8, 1.0]), rng),
                      text, "leetspeak"),
        "composed": (E.encode_composed(text), text, "base64+rot13"),
        "embedded_base64": (emb, emb_expected, "base64"),
        "hex": (E.encode_hex(text), text, "unsupported"),
        "reverse": (E.encode_reverse(text), text, "unsupported"),
        "caesar3": (E.encode_caesar(text, 3), text, "unsupported"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---------- harmful prompts ----------
    harmful = [(t, "advbench") for t in load_prompts(os.path.join(RAW, "advbench.csv"))]
    harmful += [(t, "jailbreakbench") for t in load_prompts(os.path.join(RAW, "jailbreakbench.csv"))]
    seen, uniq = set(), []
    for t, s in harmful:
        if t.lower() not in seen:
            seen.add(t.lower())
            uniq.append((t, s))
    harmful = uniq
    if not any(s == "jailbreakbench" for _, s in harmful):
        print("NOTE: data/raw/jailbreakbench.csv not found -> using AdvBench only.")
    lengths = np.array([len(t) for t, _ in harmful])
    lo, hi = np.percentile(lengths, [2, 98])

    # ---------- benign prompts (held-out Alpaca partition) ----------
    alp = pd.read_csv(os.path.join(RAW, "alpaca.csv"))
    pool = [(i, norm(t)) for i, t in enumerate(alp["prompt"].astype(str)) if i % 5 == 0]
    pool = [(i, t) for i, t in pool if lo <= len(t) <= hi and t.lower() not in seen]
    rng.shuffle(pool)
    benign_hard = [(t, "jbb_benign") for t in load_prompts(os.path.join(RAW, "jbb_benign.csv"))]

    def pick(split_name, n):
        """n benign prompts for a split; half imperative-style, half random."""
        part = [t for i, t in pool if (i // 5) % 2 == (0 if split_name == "dev" else 1)]
        matched = [t for t in part if t.split()[0].lower().strip(",.:") in IMPERATIVE]
        others = [t for t in part if t not in set(matched)]
        take = matched[: n // 2] + others[: n - min(n // 2, len(matched))]
        return [(t, "alpaca") for t in take]

    rows, gid = [], 0

    def add_group(text, label, source, style, split):
        nonlocal gid
        for variant, (obf, expected, steps) in make_variants(text, rng).items():
            rows.append(dict(id=f"g{gid}_{variant}", group_id=f"g{gid}", split=split,
                             label=label, source=source, style=style, variant=variant,
                             text=obf, original=text, expected=expected,
                             expected_steps=steps))
        gid += 1

    for split in ("dev", "eval"):
        h = [(t, s) for t, s in harmful if (bucket(t, 5) == 0) == (split == "dev")]
        for t, s in h:
            add_group(t, "unsafe", s, "harmful", split)
        benign = pick(split, len(h))
        if split == "eval":
            benign += benign_hard
        for t, s in benign:
            style = ("hard" if s == "jbb_benign" else
                     "imperative" if t.split()[0].lower().strip(",.:") in IMPERATIVE
                     else "random")
            add_group(t, "safe", s, style, split)

    df = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    g = df.drop_duplicates("group_id")
    print(f"Saved {len(df)} rows ({len(g)} prompt groups x 9 variants) -> {OUT}")
    print(g.groupby(["split", "label"]).size().unstack().to_string())
    print("\nbenign style mix (eval):")
    print(g[(g.split == "eval") & (g.label == "safe")]["style"].value_counts().to_string())
    print(f"\nlength (chars) harmful median {int(np.median(lengths))}, "
          f"benign median {int(g[g.label=='safe']['original'].str.len().median())}")


if __name__ == "__main__":
    main()
