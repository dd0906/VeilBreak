"""
Does better TRAINING data make the classifier better?

Error analysis showed two weak spots: safe prompts that only sound risky, and
harmful prompts unlike AdvBench. So we try adding two public datasets:
  xstest    safe prompts that sound unsafe, plus unsafe look-alikes (Rottger et al. 2024)
  harmbench varied harmful behaviours, 'standard' category only (Mazeika et al. 2024)

Why you can trust the comparison
--------------------------------
* The test prompts and the folds are the same as scripts/evaluate.py, so the
  'baseline' row must reproduce RESULTS.md. New data only ever goes into training.
* Near-duplicates are dropped: JailbreakBench partly reuses other benchmarks, so
  a new training prompt too similar to any existing prompt would leak the answer.
* XSTest is split by hash: 60% train, 20% tune, 20% test.
* The winner and its threshold are chosen on the TUNE set (dev split + XSTest
  tune), never on the test numbers. The test numbers are only reported.
* The classifier still trains on plain text only; only the decoded text is scored.

Usage:  python scripts/evaluate_training_data.py [--embedder minilm]
Run it alone: it loads a model and takes several GB of memory.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from evaluate import OBFUSCATED, SMUGGLING, boot, md_table, run_decoder  # noqa: E402
from src.classifier import SafetyClassifier  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
NEAR_DUPLICATE = 0.85       # cosine similarity above which a new prompt is dropped
THRESHOLDS = np.round(np.arange(0.20, 0.81, 0.01), 2)
SYSTEMS = {"baseline": [], "+xstest": ["xstest"], "+harmbench": ["harmbench"],
           "+both": ["xstest", "harmbench"]}


def xstest_part(text):
    """Stable split of XSTest: 'train' (60%), 'tune' (20%) or 'test' (20%)."""
    b = int(hashlib.md5(text.lower().encode()).hexdigest(), 16) % 5
    return "tune" if b == 3 else "test" if b == 4 else "train"


def balanced_accuracy(p, y, t):
    unsafe = np.asarray(y) == "unsafe"
    return ((p[unsafe] >= t).mean() + (p[~unsafe] < t).mean()) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="minilm", choices=["minilm", "tfidf"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    data = pd.read_csv(os.path.join(ROOT, "data", "processed", "veilbreak_v2.csv"))
    df = run_decoder(data[data.split == "eval"].reset_index(drop=True))
    dev = data[(data.split == "dev") & (data.variant == "plain")]
    xs = pd.read_csv(os.path.join(RAW, "xstest.csv"))
    hb = pd.read_csv(os.path.join(RAW, "harmbench.csv"))
    xs["part"] = xs.prompt.map(xstest_part)

    # Embeddings are shared between folds so each text is computed only once.
    shared = SafetyClassifier(args.embedder, seed=args.seed)

    # ---- drop near-duplicates of anything already in the dataset ----
    known = data[data.variant == "plain"].text.tolist()
    kept = {}
    for name, pool in (("xstest", xs), ("harmbench", hb)):
        if args.embedder == "minilm":
            sim = cosine_similarity(shared._embed(pool.prompt), shared._embed(known)).max(axis=1)
            keep = sim < NEAR_DUPLICATE
        else:                                # tfidf smoke test: exact matches only
            keep = ~pool.prompt.str.lower().isin({t.lower() for t in known}).values
        kept[name] = pool[keep]
        print(f"{name}: {len(pool)} prompts, {int((~keep).sum())} near-duplicates dropped")
    extra = {"xstest": kept["xstest"][kept["xstest"].part == "train"], "harmbench": kept["harmbench"]}
    tune = pd.concat([dev.rename(columns={"text": "prompt"})[["prompt", "label"]],
                      kept["xstest"][kept["xstest"].part == "tune"][["prompt", "label"]]])
    xs_test = kept["xstest"][kept["xstest"].part == "test"]
    print(f"train additions: xstest {len(extra['xstest'])}, harmbench {len(extra['harmbench'])}; "
          f"tune {len(tune)}, xstest test {len(xs_test)}")

    # ---- same grouped folds as evaluate.py ----
    groups = df.drop_duplicates("group_id")[["group_id", "label"]].reset_index(drop=True)
    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    scored = {s: [] for s in SYSTEMS}
    tune_p = {s: np.zeros(len(tune)) for s in SYSTEMS}
    xs_p = {s: np.zeros(len(xs_test)) for s in SYSTEMS}
    for k, (tr, te) in enumerate(skf.split(groups, groups.label, groups.group_id), 1):
        train_ids, test_ids = set(groups.group_id.iloc[tr]), set(groups.group_id.iloc[te])
        assert not (train_ids & test_ids)                    # leakage guard
        base = df[df.group_id.isin(train_ids) & (df.variant == "plain")]
        test = df[df.group_id.isin(test_ids)]
        for system, sources in SYSTEMS.items():
            texts = base.text.tolist() + [t for s in sources for t in extra[s].prompt]
            labels = base.label.tolist() + [l for s in sources for l in extra[s].label]
            clf = SafetyClassifier(args.embedder, seed=args.seed)
            if args.embedder == "minilm":
                clf._st, clf._cache = shared._minilm(), shared._cache
            clf.fit(texts, labels)
            p = clf.predict_proba(test.decoded)                  # same rule as evaluate.py
            for i, pl in enumerate(test.payloads.fillna("")):
                if pl:
                    p[i] = max(p[i], clf.predict_proba(pl.split("\n<<>>\n")).max())
            part = test[["group_id", "label", "style", "source", "variant"]].copy()
            part["p"] = p
            scored[system].append(part)
            tune_p[system] += clf.predict_proba(tune.prompt) / args.folds
            xs_p[system] += clf.predict_proba(xs_test.prompt) / args.folds
        print(f"  fold {k}/{args.folds} done")

    # ---- choose on the tune set only ----
    choices = []
    for s in SYSTEMS:
        accs = [balanced_accuracy(tune_p[s], tune.label, t) for t in THRESHOLDS]
        best = int(np.argmax(accs))
        choices.append(dict(system=s, tune_at_050=balanced_accuracy(tune_p[s], tune.label, 0.5),
                            best_threshold=THRESHOLDS[best], tune_at_best=accs[best]))
    choices = pd.DataFrame(choices)
    winner = choices.loc[choices.tune_at_050.idxmax(), "system"]
    win_t = float(choices.set_index("system").loc[winner, "best_threshold"])
    print(choices.round(3).to_string(index=False), f"\nwinner: {winner}, threshold {win_t}")

    # ---- report on the test data ----
    rng = np.random.default_rng(args.seed)
    rows = []
    runs = [(s, s, 0.5) for s in SYSTEMS] + [(f"{winner} @ {win_t}", winner, win_t)]
    for label, s, t in runs:
        d = pd.concat(scored[s])
        d["flag"] = d.p >= t
        pv = d.pivot(index="group_id", columns="variant", values="flag").astype(float)
        meta = d.drop_duplicates("group_id").set_index("group_id").reindex(pv.index)
        y = meta.label == "unsafe"
        slices = {
            "plain: catch": pv.plain[y],
            "plain: catch, AdvBench": pv.plain[y & (meta.source == "advbench")],
            "plain: catch, JailbreakBench": pv.plain[y & (meta.source == "jailbreakbench")],
            "plain: false positives": pv.plain[~y],
            "plain: false positives, hard-benign": pv.plain[~y & (meta["style"] == "hard")],
            "encoded: catch": pv[OBFUSCATED][y].mean(axis=1),
            "encoded: false positives": pv[OBFUSCATED][~y].mean(axis=1),
            "smuggling: catch": pv[SMUGGLING][y].mean(axis=1),
            "smuggling: false positives": pv[SMUGGLING][~y].mean(axis=1),
            "XSTest test: catch": pd.Series((xs_p[s] >= t)[xs_test.label.values == "unsafe"], dtype=float),
            "XSTest test: false positives": pd.Series((xs_p[s] >= t)[xs_test.label.values == "safe"], dtype=float),
        }
        for name, v in slices.items():
            m, lo, hi = boot(v.values, rng)
            rows.append(dict(system=label, metric=name, value=m, lo=lo, hi=hi, n=len(v)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "training_data_metrics.csv"), index=False)

    names = [r[0] for r in runs]
    table = [[m, int(res[(res.metric == m)].n.iloc[0])] +
             [(lambda r: f"{100 * r.value:.1f}% [{100 * r.lo:.1f}–{100 * r.hi:.1f}]")(
                 res[(res.metric == m) & (res.system == n)].iloc[0]) for n in names]
             for m in res.metric.unique()]
    L = ["# VeilBreak — does better training data help?\n",
         "_Generated by `scripts/evaluate_training_data.py` — do not edit by hand._\n",
         f"- Embedder: **{args.embedder}**, seed {args.seed}, {args.folds}-fold grouped CV. Same test folds as RESULTS.md.",
         f"- Training additions after near-duplicate removal: XSTest {len(extra['xstest'])}, HarmBench {len(extra['harmbench'])}.",
         f"- **Chosen on the tune set ({len(tune)} prompts), not on the numbers below: `{winner}`, threshold {win_t}.**",
         "- Every row runs through VeilBreak decoding first. Brackets = 95% bootstrap CI.\n",
         md_table(["metric", "n"] + names, table), "",
         "## Selection on the tune set\n",
         md_table(list(choices.columns), [[r.system, f"{r.tune_at_050:.3f}", r.best_threshold, f"{r.tune_at_best:.3f}"]
                                          for r in choices.itertuples()]), ""]
    path = os.path.join(args.out, "TRAINING_DATA.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print("Saved", path)


if __name__ == "__main__":
    main()
