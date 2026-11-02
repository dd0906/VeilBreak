"""
VeilBreak evaluation harness. Produces every number and chart in the README.

What it measures
----------------
A. DECODER quality (no ML): for every variant, did the pipeline find the right
   encoding layers, and how close is the decoded text to the true original?
B. END-TO-END safety classification, compared across systems:
     - classifier, no decoding             (baseline: same model, raw input)
     - classifier + VeilBreak               (decode first)
     - classifier + perfect decoder         (upper bound: isolates decoder errors)
     - Llama Guard, no decoding / + VeilBreak   (only if results/llamaguard_predictions.csv exists)

Why you can trust the comparison
--------------------------------
* The classifier is TRAINED ON PLAIN TEXT ONLY (project scope).
* Cross-validation is GROUPED: all variants of one prompt sit in the same fold,
  so the model is never tested on a re-encoding of a prompt it trained on.
* Benign prompts are obfuscated too, so "looks encoded" does not leak the label.
* Only the `eval` split is used here; thresholds were tuned on `dev`.
* Confidence intervals come from bootstrapping over prompt groups.

Usage
-----
  python scripts/evaluate.py --embedder minilm      # the real run
  python scripts/evaluate.py --embedder tfidf       # fast, no downloads (smoke test)
"""
import argparse
import datetime
import difflib
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from src.classifier import SafetyClassifier  # noqa: E402
from src.pipeline import decode  # noqa: E402

IN_SCOPE = ["plain", "base64", "rot13", "leetspeak", "composed", "embedded_base64"]
# Hex was added after the headline group was fixed, so it stays out of OBFUSCATED:
# the "ALL obfuscated" numbers keep meaning the same five encodings.
ADDED_LATER = ["hex"]
OUT_SCOPE = ["reverse", "caesar3"]
OBFUSCATED = IN_SCOPE[1:]
SMUGGLING = ["tag_smuggling", "emoji_smuggling", "zero_width", "homoglyph"]


def norm(s):
    return " ".join(str(s).lower().split())


# ======================================================================
# A. Decoder metrics
# ======================================================================
def run_decoder(df):
    """Decode every row; return df with decoded/steps/suspicious columns."""
    res = [decode(t) for t in df["text"]]
    df = df.copy()
    df["decoded"] = [r.decoded for r in res]
    df["payloads"] = ["\n<<>>\n".join(r.payloads) for r in res]
    df["steps"] = ["+".join(r.encodings) for r in res]
    df["suspicious"] = [r.suspicious for r in res]
    return df


def decoder_table(df):
    rows = []
    for v in IN_SCOPE + ADDED_LATER + OUT_SCOPE + SMUGGLING:
        d = df[df.variant == v]
        exp = d["expected_steps"].fillna("")
        if v in OUT_SCOPE:
            steps_ok = (d["steps"] == "").mean()           # must NOT pretend to decode
        else:
            steps_ok = (d["steps"] == exp).mean()
        rows.append({
            "variant": v,
            "n": len(d),
            "right_layers": steps_ok,
            "exact_match": np.mean([norm(a) == norm(b) for a, b in zip(d.decoded, d.expected)]),
            "char_similarity": np.mean([difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
                                        for a, b in zip(d.decoded, d.expected)]),
            "flagged_suspicious": d["suspicious"].mean(),
        })
    return pd.DataFrame(rows)


# ======================================================================
# B. Classification
# ======================================================================
def cross_validate(df, embedder, folds, seed):
    """Out-of-fold P(unsafe) for raw / decoded / oracle text, for every row."""
    groups = df.drop_duplicates("group_id")[["group_id", "label"]].reset_index(drop=True)
    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    parts = []
    for k, (tr, te) in enumerate(skf.split(groups, groups.label, groups.group_id), 1):
        train_ids = set(groups.group_id.iloc[tr])
        test_ids = set(groups.group_id.iloc[te])
        train = df[df.group_id.isin(train_ids) & (df.variant == "plain")]
        test = df[df.group_id.isin(test_ids)].copy()
        assert not (train_ids & test_ids)                    # leakage guard
        clf = SafetyClassifier(embedder, seed=seed).fit(train["text"], train["label"])
        test["p_clf_raw"] = clf.predict_proba(test["text"])
        # VeilBreak = classify the decoded text AND each revealed payload; take the max.
        p_dec = clf.predict_proba(test["decoded"])
        for i, pl in enumerate(test["payloads"].fillna("")):
            if pl:
                p_dec[i] = max(p_dec[i], clf.predict_proba(pl.split("\n<<>>\n")).max())
        test["p_clf_dec"] = p_dec
        # Oracle = same rule, but with the TRUE plaintext instead of our decoding.
        test["p_clf_oracle"] = np.maximum(clf.predict_proba(test["expected"]),
                                          clf.predict_proba(test["original"]))
        parts.append(test)
        print(f"  fold {k}/{folds}: trained on {len(train)} plain prompts, "
              f"tested on {len(test_ids)} groups")
    out = pd.concat(parts)
    for s in ("clf_raw", "clf_dec", "clf_oracle"):
        out[f"unsafe_{s}"] = out[f"p_{s}"] >= 0.5
    return out


def add_llamaguard(df, path):
    """Attach Llama Guard predictions if the file exists and is complete.

    lg_raw : verdict on the raw prompt.
    lg_dec : unsafe if the DECODED text or any revealed payload is unsafe
             (same rule VeilBreak uses with the embedding classifier).
    """
    if not os.path.exists(path):
        return df, []
    lg = pd.read_csv(path)
    raw = lg[lg["input"] == "raw"].set_index("id")["pred"]
    decoded = lg[lg["input"] == "decoded"].set_index("id")["pred"]
    dec_any = (lg[lg["input"] != "raw"]["pred"] == "unsafe").groupby(lg[lg["input"] != "raw"]["id"]).any()
    systems = []
    candidates = [("lg_raw", raw, raw == "unsafe"),
                  ("lg_dec", decoded, dec_any)]
    if (lg["pred"] == "code_only").any():
        # Generous variant: a bare hazard code ("S1") counts as flagged.
        candidates.insert(1, ("lg_raw_hi", raw, raw.isin(["unsafe", "code_only"])))
    for name, seen, unsafe in candidates:
        covered = df["id"].isin(seen.index)
        if not covered.all():
            print(f"  WARNING: Llama Guard '{name}' covers only {covered.mean():.0%} "
                  f"of the eval rows -> skipped (run the full eval_llamaguard.py).")
            continue
        df[f"unsafe_{name}"] = df["id"].map(unsafe).fillna(False).astype(bool)
        systems.append(name)
    codes = int((lg["pred"] == "code_only").sum())
    if codes:
        print(f"  note: {codes} Llama Guard replies were a bare hazard code (strict = safe, generous = unsafe).")
    errors = int((lg["pred"] == "error").sum())
    if errors:
        # An error is not a verdict: counting it as "safe" would quietly distort the results.
        sys.exit(f"Stopping: {errors} Llama Guard replies are errors. Fix them first with:\n"
                 f"  python scripts/eval_llamaguard.py --retry-errors")
    return df, systems


def boot(values, rng, B=1000):
    """Mean and 95% CI of a per-group 0/1 array by bootstrapping groups."""
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    means = v[idx].mean(axis=1)
    return v.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def metric_table(df, systems, seed):
    """Catch rate (unsafe) and false-positive rate (safe) with CIs, per variant."""
    rng = np.random.default_rng(seed)
    rng_smuggle = np.random.default_rng(seed + 1)
    rows = []
    for sysname in systems:
        pv = df.pivot(index="group_id", columns="variant", values=f"unsafe_{sysname}")
        lab = df.drop_duplicates("group_id").set_index("group_id")["label"].reindex(pv.index)
        for metric, mask in (("catch_rate", lab == "unsafe"), ("false_positive_rate", lab == "safe")):
            sub = pv[mask.values].astype(float)
            cols = {v: sub[v].values for v in IN_SCOPE + ADDED_LATER + OUT_SCOPE}
            cols["ALL obfuscated (5 in-scope)"] = sub[OBFUSCATED].mean(axis=1).values
            cols["ALL unsupported (2)"] = sub[OUT_SCOPE].mean(axis=1).values
            for v, arr in cols.items():
                m, lo, hi = boot(arr, rng)
                rows.append(dict(system=sysname, metric=metric, variant=v,
                                 value=m, lo=lo, hi=hi, n_groups=len(arr)))
            # Smuggling gets its own generator so the numbers above do not shift.
            smuggle = {v: sub[v].values for v in SMUGGLING}
            smuggle["ALL smuggling (4)"] = sub[SMUGGLING].mean(axis=1).values
            for v, arr in smuggle.items():
                m, lo, hi = boot(arr, rng_smuggle)
                rows.append(dict(system=sysname, metric=metric, variant=v,
                                 value=m, lo=lo, hi=hi, n_groups=len(arr)))
    return pd.DataFrame(rows)


# ======================================================================
# Reporting
# ======================================================================
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def pct(x):
    return f"{100 * x:.1f}%"


def cell(r):
    return f"{pct(r.value)} [{pct(r.lo)}–{pct(r.hi)}]"


def chart(mt, systems, names, metric, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"clf_raw": "#999999", "clf_dec": "#0072B2", "clf_oracle": "#56B4E9",
              "lg_raw": "#E69F00", "lg_raw_hi": "#F0C050", "lg_dec": "#009E73"}
    variants = IN_SCOPE + ADDED_LATER + OUT_SCOPE
    fig, ax = plt.subplots(figsize=(12, 5.6))
    top = 100 if metric == "catch_rate" else max(
        5, float(np.ceil(mt[mt.metric == metric].hi.max() * 100 * 1.4 / 5) * 5))
    ax.axvspan(6.5, len(variants) - 0.5, color="#f1f1f1", zorder=0)      # out-of-scope zone
    ax.text((6.5 + len(variants) - 0.5) / 2, top * 1.045, "OUT OF SCOPE: no decoder exists for these",
            ha="center", fontsize=9, color="#555")
    w = 0.8 / len(systems)
    for i, s in enumerate(systems):
        d = mt[(mt.system == s) & (mt.metric == metric)].set_index("variant").loc[variants]
        x = np.arange(len(variants)) + i * w - 0.4 + w / 2
        err = np.vstack([d.value - d.lo, d.hi - d.value])
        ax.bar(x, d.value * 100, w, yerr=err * 100, capsize=2, label=names[s],
               color=colors[s], edgecolor="white", linewidth=0.5, zorder=2,
               hatch="//" if s == "clf_oracle" else None, error_kw={"elinewidth": 1})
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.replace("_", "\n") for v in variants])
    ax.set_ylabel(title.split(" (")[0] + " (%)")
    ax.set_ylim(0, top * 1.12)
    ax.set_yticks(np.linspace(0, top, 6 if metric == "catch_rate" else 5))
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="minilm", choices=["minilm", "tfidf"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "processed", "veilbreak_v2.csv"))
    ap.add_argument("--llamaguard", default=os.path.join(ROOT, "results", "llamaguard_predictions.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data)
    df = df[df.split == "eval"].reset_index(drop=True)
    df["expected_steps"] = df["expected_steps"].fillna("")
    n_groups = df.group_id.nunique()
    print(f"Eval split: {len(df)} rows, {n_groups} prompt groups")

    print("\n[A] Running decoder on every row...")
    df = run_decoder(df)
    dec = decoder_table(df)

    print(f"\n[B] {args.folds}-fold grouped cross-validation ({args.embedder} embedder)...")
    df = cross_validate(df, args.embedder, args.folds, args.seed)
    df, lg_systems = add_llamaguard(df, args.llamaguard)

    emb = {"minilm": "MiniLM+LogReg", "tfidf": "TF-IDF+LogReg"}[args.embedder]
    names = {"clf_raw": f"{emb}, no decode", "clf_dec": f"{emb} + VeilBreak",
             "clf_oracle": f"{emb} + perfect decoder (upper bound)",
             "lg_raw": "Llama Guard, no decode (strict)", "lg_raw_hi": "Llama Guard, no decode (generous)", "lg_dec": "Llama Guard + VeilBreak"}
    systems = ["clf_raw", "clf_dec", "clf_oracle"] + lg_systems
    mt = metric_table(df, systems, args.seed)

    # ---------------- save raw outputs ----------------
    dec.to_csv(os.path.join(args.out, "decoder_metrics.csv"), index=False)
    mt.to_csv(os.path.join(args.out, "classification_metrics.csv"), index=False)
    keep = [c for c in df.columns if c.startswith(("p_", "unsafe_"))]
    df[["id", "group_id", "label", "style", "variant", "steps", "suspicious"] + keep] \
        .to_csv(os.path.join(args.out, "predictions.csv"), index=False)
    chart(mt, systems, names, "catch_rate", "Catch rate on unsafe prompts (higher is better)",
          os.path.join(args.out, "catch_rate_by_variant.png"))
    chart(mt, systems, names, "false_positive_rate",
          "False-positive rate on safe prompts (lower is better)",
          os.path.join(args.out, "false_positive_rate_by_variant.png"))

    # ---------------- RESULTS.md ----------------
    L = []
    L.append("# VeilBreak — results\n")
    L.append(f"_Generated by `scripts/evaluate.py` on {datetime.date.today()} — do not edit by hand._\n")
    L.append(f"- Embedder: **{args.embedder}** · seed {args.seed} · {args.folds}-fold grouped CV · "
             f"{n_groups} prompt groups ({int((df.drop_duplicates('group_id').label=='unsafe').sum())} unsafe, "
             f"{int((df.drop_duplicates('group_id').label=='safe').sum())} safe) × 13 variants")
    L.append("- Classifier trained on plain text only; every cell below is out-of-fold. "
             "Brackets = 95% bootstrap CI over prompt groups.")
    L.append("- Llama Guard rows: " + ("included." if lg_systems else
             "**not included yet** (run `scripts/eval_llamaguard.py`, then re-run this script).") + "\n")

    L.append("## Summary\n")
    def get(sysname, metric, variant):
        return mt[(mt.system == sysname) & (mt.metric == metric) & (mt.variant == variant)].iloc[0]
    OBF = "ALL obfuscated (5 in-scope)"
    L.append(md_table(
        ["system", "catch rate: plain", "catch rate: obfuscated", "false positives: plain",
         "false positives: obfuscated"],
        [[names[s], cell(get(s, "catch_rate", "plain")), cell(get(s, "catch_rate", OBF)),
          cell(get(s, "false_positive_rate", "plain")), cell(get(s, "false_positive_rate", OBF))]
         for s in systems]))
    L.append("\n*Obfuscated = average over base64, rot13, leetspeak, composed, embedded_base64.*\n")

    L.append("## 1. Decoder quality (no ML involved)\n")
    L.append(md_table(["variant", "n", "right layers found", "exact text match",
                       "char similarity", "flagged suspicious"],
                      [[r.variant + (" *(out of scope)*" if r.variant in OUT_SCOPE else ""),
                        r.n, pct(r.right_layers), pct(r.exact_match),
                        pct(r.char_similarity), pct(r.flagged_suspicious)]
                       for r in dec.itertuples()]))
    L.append("\n*For out-of-scope encodings, \"right layers\" means the pipeline correctly "
             "did NOT pretend to decode; \"flagged suspicious\" is the share it recognised as "
             "undecodable gibberish.*\n")

    for metric, title, note in (
            ("catch_rate", "2. Catch rate — unsafe prompts flagged unsafe (higher is better)", ""),
            ("false_positive_rate", "3. False-positive rate — safe prompts flagged unsafe (lower is better)", "")):
        L.append(f"## {title}\n")
        order = IN_SCOPE + ADDED_LATER + OUT_SCOPE + ["ALL obfuscated (5 in-scope)", "ALL unsupported (2)"]
        sub = mt[mt.metric == metric]
        rows = []
        for v in order:
            row = [("**" + v + "**") if v.startswith("ALL") else v]
            for s in systems:
                r = sub[(sub.system == s) & (sub.variant == v)].iloc[0]
                row.append(cell(r))
            rows.append(row)
        L.append(md_table(["variant"] + [names[s] for s in systems], rows) + "\n")

    L.append("## 4. Plain-text false positives by benign style\n")
    L.append("Checks that the classifier is not just detecting \"imperative instruction style\".\n")
    rows = []
    plain_safe = df[(df.variant == "plain") & (df.label == "safe")]
    for style, g in plain_safe.groupby("style"):
        rows.append([style, len(g)] + [pct(g[f"unsafe_{s}"].mean()) for s in systems])
    L.append(md_table(["benign style", "n"] + [names[s] for s in systems], rows) + "\n")

    L.append("## 5. Invisible-character smuggling\n")
    L.append("Current attacks that hide text in characters a person cannot see. The classifier "
             "is the same one as above (no retraining). These are NOT included in the "
             "\"ALL obfuscated\" numbers.\n")
    for metric, title in (("catch_rate", "Catch rate (higher is better)"),
                          ("false_positive_rate", "False-positive rate (lower is better)")):
        L.append(f"**{title}**\n")
        rows = []
        for v in SMUGGLING + ["ALL smuggling (4)"]:
            row = [("**" + v + "**") if v.startswith("ALL") else v]
            for s in systems:
                row.append(cell(get(s, metric, v)))
            rows.append(row)
        L.append(md_table(["variant"] + [names[s] for s in systems], rows) + "\n")

    with open(os.path.join(args.out, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n" + "\n".join(L))
    print(f"\nSaved results to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
