# Methodology

How the numbers in the README are produced, and which choices could bias them.

## 1. Data

| Set | Source | Role |
|---|---|---|
| Unsafe | AdvBench (520) + JailbreakBench "harmful" (100, if downloaded) | direct harmful requests |
| Safe, general | Alpaca, rows with `index % 5 == 0` only | everyday instructions |
| Safe, hard | JailbreakBench "benign" (100, if downloaded) | benign prompts written to mirror the harmful ones |

`scripts/build_dataset.py` then:

1. deduplicates prompts and assigns one `group_id` per original prompt;
2. splits groups into **dev** (hash bucket 0 of 5, used for tuning) and **eval** (everything else, used for every reported number);
3. filters benign prompts to the harmful prompts' length range (2nd–98th percentile), and takes half of them in imperative style ("Write…", "Create…") so style alone can't separate the classes;
4. creates **13 variants of every prompt, safe and unsafe alike**: plain, base64, rot13, leetspeak (intensity drawn from {0.6, 0.8, 1.0}), composed (`base64(rot13(x))`), embedded_base64 (inside one of 4 wrapper sentences), hex (first out of scope, decoder added later with no retraining, kept out of the "ALL obfuscated" average), two **out-of-scope** encodings (reversed, Caesar-3), and four invisible-character attacks (tag smuggling, emoji smuggling, zero-width, homoglyph), reported separately. The four attack variants use their own random generator and are appended after the original rows, so the original rows reproduce exactly.

The language model behind ROT13/leetspeak detection is fit on Alpaca rows with `index % 5 != 0`, so it has never seen an evaluation prompt.

## 2. Classification protocol

- 5-fold `StratifiedGroupKFold` over prompt groups (seed 42): a group's 13 variants are always together in train or in test (an `assert` checks the overlap is empty).
- The classifier trains on the **plain** rows of the training groups only. It never sees an encoded prompt.
- On the test groups every row is scored three ways: raw text, decoded text (VeilBreak), and the true plaintext (a "perfect decoder" upper bound that isolates decoder errors from classifier errors).
- VeilBreak's verdict = max P(unsafe) over the decoded text and each revealed Base64 payload (a harmless wrapper sentence otherwise dilutes the signal).
- Llama Guard (optional) is run on the same rows: raw, and decoded (+payloads). It needs no training so it sees all eval rows.

## 3. Statistics

Catch rate = share of unsafe prompts flagged unsafe; false-positive rate = share of safe prompts flagged unsafe. Intervals are 95 % percentile bootstrap CIs, resampling **prompt groups** (1,000 resamples), because variants of one prompt are not independent. "ALL obfuscated" averages the five in-scope obfuscated variants per group before bootstrapping.

## 4. Decoder thresholds

| Constant | Value | Meaning |
|---|---|---|
| `ROT13_MIN_GAIN` | 0.60 | decoded text must score this much higher (avg. log-prob per character) |
| `ROT13_MIN_SCORE` | -3.0 | and itself look like English |
| `ROT13_MIN_LETTERS` | 12 | don't guess on very short text |
| `SUSPICIOUS_BELOW` | -3.4 | unreadable and undecodable → flag |
| Leetspeak evidence rule | ≥2 tokens, or 1 token that becomes a known 5+ letter word | a single `Input1` / `F1` is not leetspeak |

Chosen on the dev split with `scripts/tune_thresholds.py`; ROT13 recall and false triggers are flat across the whole grid (gain 0.2–1.5, score -3.4…-2.6), so the exact values don't matter. The leetspeak evidence rule was added after measuring false triggers on ~5,200 benign dev prompts (21 → 5).

## 5. What the numbers do NOT show

- Robustness to an attacker who adapts to these thresholds.
- Whether a target LLM obeys the decoded instruction (we score classifier verdicts).
- Behaviour on non-English text or on jailbreak templates (roleplay, DAN-style, adversarial suffixes), which are out of scope.
- Tight absolute accuracy: AdvBench is repetitive and small, so even grouped CV is optimistic about plain-text accuracy. The *relative* effect of decoding, which is the claim, is far less sensitive to this.
