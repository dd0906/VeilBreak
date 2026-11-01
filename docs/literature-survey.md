> **Corrections added Oct 2026 (verified against the sources):**
> - The Cisco / Robust Intelligence post (#2) attacked Meta's **Prompt-Guard-86M**, *not* Llama Guard. Reported: accuracy 100% -> 0.2% (99.8% bypass) on 450 prompts by spacing out characters and removing punctuation. It does NOT show that Llama Guard is vulnerable; VeilBreak's Llama Guard benchmark tests that directly.
> - Author attributions in earlier notes were unreliable; cite by title and arXiv id instead.
> - Venues listed below (e.g. "ICLR 2025") were not verified; cite arXiv ids.
> - "Recover, Decode, Reguard" (arXiv 2607.26574) was later retitled "Attack Ensembles Expose a Safety-Utility Trade-off in Black-Box Guard Defenses Against Encoded VLM Jailbreaks".
> - Numbers below that were not re-checked (per-guardrail evasion rates, ASR tables) should be re-verified before being quoted.

---
name: literature-survey
description: Key papers on LLM jailbreak obfuscation, guardrail bypass, and decode-then-classify defenses — use to check if a VeilBreak design choice overlaps existing work
type: reference
---

## ATTACK PAPERS (prove the problem is real)

### 1. Plentiful Jailbreaks with String Compositions (NeurIPS SoLaR 2024)
- URL: https://arxiv.org/abs/2411.01084
- Builds framework of 20 invertible string transformations as building blocks
- Covers EXACTLY our three encodings: Base64, leetspeak, ROT13 — plus Caesar, Atbash, Morse, binary, reversal, vowel repetition, alternating case, palindrome, spoonerism, stuttering, JSON, LaTeX, interleaving delimiter, prefix rotation
- Key insight: transformations are COMPOSABLE — chain multiple encodings for combinatorially large attack surface
- Invertibility requirement: each transform f must have f⁻¹ where f⁻¹(f(s)) = s — enables automated encode/decode
- Achieves competitive ASR on HarmBench across frontier models
- RELEVANCE: Our strongest motivation citation. Composition of encodings = natural stretch goal for VeilBreak

### 2. Bypassing Meta's LLaMA Classifier (Cisco Security Blog 2024)
- URL: https://blogs.cisco.com/security/bypassing-metas-llama-classifier-a-simple-jailbreak
- Attack: simply spacing out characters + removing punctuation
- Python: re.sub('[punctuation]', '', ' '.join(prompt_injection))
- Result: Llama Guard accuracy drops 100% → 0.2% (99.8% bypass success)
- WHY it works: single-character embeddings had avg MAE < 0.0003 between fine-tuned and base model — barely changed during safety fine-tuning
- Model learned semantic patterns, not character-level manipulation
- RELEVANCE: Direct evidence our benchmark target (Llama Guard) fails on character-level obfuscation

### 3. Bypassing LLM Guardrails: Evasion Attacks (arXiv Apr 2025)
- URL: https://arxiv.org/abs/2504.11168
- 12 character injection methods: number substitution, homoglyph, zero-width chars, diacriticals, spacing, underline accents, upside-down, full-width, bidirectional, random deletion, emoji smuggling, Unicode tags
- 8 adversarial ML methods: Bert-Attack, BAE, Deep Word Bug, Alzantot, TextFooler, PWWS, Pruthi, TextBugger
- Tested against 6 guardrails with evasion rates:
  - Vijil: 91.67% jailbreak / 87.95% injection
  - ProtectAI v1: 51.39% / 77.32%
  - NeMo Guard: 72.54% / N/A
  - Azure Prompt Shield: 60.15% / 71.98%
  - Meta Prompt Guard: 73.08% / 70.44%
  - ProtectAI v2: limited / 20.26%
- Emoji smuggling achieved 100% evasion across both threat types
- White-box→black-box transfer: word importance rankings from ProtectAI v2 improved attacks on Azure by up to 33%
- RELEVANCE: Problem is industry-wide, not just Llama Guard

### 4. Cipher Jailbreaks — ACE & LACE (ICLR 2025)
- URL: https://arxiv.org/abs/2402.10601
- 5 ciphers in ACE: Keyboard cipher, Upside-down, Word reversal, Grid encoding (chess-notation 5x5), Word substitution
- LACE: layers two ciphers (Word Substitution + one of ROT13/Keyboard/Grid/Word Reversal)
- ASR by model: GPT-4o 40%→78% (ACE→LACE), Llama-3.1-8B 88%→88%, Llama-3.1-70B 54%→54%, GPT-5 0%→8%
- CipherBench: 240 instances testing decryption across 10 ciphers (Base64, ROT13, Pig Latin, LeetSpeak = common; Keyboard, Upside-Down, Word Reversal = uncommon; Word-Sub, Grid, ASCII Art = novel)
- KEY FINDING: Models with higher decryption ability are MORE vulnerable — smarter = more exploitable
- Common ciphers: high decryption success but low ASR (defenses work for known encodings)
- Novel ciphers: comparable ASR despite lower decryption — safety training can't cover unknown ciphers
- RELEVANCE: Explains WHY our approach works — we handle known encodings with deterministic decoders, which is exactly where rule-based beats ML. Novel ciphers = Future Work.

### 5. StructuralSleight (arXiv Jun 2024)
- URL: https://arxiv.org/abs/2406.08754
- 12 UTOS templates: Tree, Graph, Data dict, JSON, XML, Mind map, Flowchart, Sequence diagram, HTML/LaTeX/Markdown tables, Python classes
- 6 obfuscation methods: Letter-to-number, GB18030, Double reverse, Custom letter mapping, Letter indexed permutation, Multi-stage task obfuscation
- 3 attack stages: SA (structural only) → SCA (structural+character/context) → FSA (full obfuscation)
- ASR: GPT-4o SCA=94.62%, Llama3-70B SCA=92%, Claude3-Opus SCA=82.31%
- RELEVANCE: Different attack class (structural, not encoding). Out of scope for VeilBreak but should mention in Related Work as acknowledged-but-not-addressed.

## DEFENSE PAPERS (validate our architecture)

### 6. Recover, Decode, Reguard (arXiv Jul 2026) ⚠️ CLOSEST TO VEILBREAK
- URL: https://arxiv.org/abs/2607.26574
- Pipeline: recover-and-decode amplifier → transcribes/restates encoded text to plain language → feeds to existing guard → optional reguard layer
- Targets VISION-LANGUAGE models (VLMs), not text-only
- Recovery is ML-based (uses a model to restate), NOT rule-based like ours
- Tested 5 guards × 2 target VLMs against 11-attack ensemble
- Results: undefended ensemble breaks 89-91% of behaviors; best guard+amplifier still leaves 63-65% vulnerable
- Reguard layer closes residual BUT drives benign over-refusal to 81-92%
- CRITICAL DIFFERENCES FROM VEILBREAK:
  1. They target VLMs, we target text-only LLMs
  2. They use ML-based recovery (another model), we use deterministic rule-based decoders
  3. They show fundamental safety-utility tradeoff we should also measure
  4. Their approach is heavier compute; ours runs on a laptop
- POSITIONING: VeilBreak is lightweight, deterministic, text-focused complement to their ML-based VLM approach

### 7. Broken-Token: CPT Filtering (arXiv Oct 2025)
- URL: https://arxiv.org/abs/2510.26847
- Detection method: measures Characters-Per-Token (CPT) — obfuscated text produces more, shorter tokens
- Optimal CPT thresholds: Qwen-3=2.69, GPT-4o=2.92, Llama-3.2=2.71, Gemma-3=2.81
- Tested on 5 encodings: Caesar (shift=3), Leetspeak, Reversed text, Binary, Base64
- Performance: 99.7-99.9% accuracy, 0.991-0.997 F1 across all tokenizers
- Edge cases: works on inputs as short as ~3 tokens; sliding window (5-token) catches mixed encoded/benign; secondary monolingual tokenizer (GPT-2) handles Chinese/Arabic
- Computational cost: negligible — tokenization happens before LLM anyway
- Dataset: ~120k prompts from 7 HuggingFace datasets × 5 encodings, available at huggingface.co/datasets/jfrog/obfuscation-identification
- RELEVANCE: Alternative detection approach to our regex/heuristic. CPT metric could IMPROVE our encoding detector. Their HuggingFace dataset could supplement our test set.

## DATASET PAPERS

### 8. JailbreakBench (NeurIPS 2024 Datasets & Benchmarks)
- URL: https://arxiv.org/abs/2404.01318
- 100 standardized harmful behaviors aligned with OpenAI usage policies
- Sources: original + Zou et al. (2023) + Mazeika et al. (2023, 2024)
- Components: jailbreak artifacts repo, behavior dataset, evaluation framework (threat models, system prompts, chat templates, scoring), public leaderboard
- GitHub: github.com/JailbreakBench/jailbreakbench
- HuggingFace: JailbreakBench/JBB-Behaviors
- RELEVANCE: Our primary jailbreak prompt source. Report results per behavior category, not just overall.

### AdvBench
- From Zou et al. "Universal and Transferable Adversarial Attacks on Aligned Language Models" (2023)
- CSV at: github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv
- Contains harmful behavior strings used for adversarial suffix attacks
- RELEVANCE: Secondary jailbreak prompt source, supplements JailbreakBench

## GAP ANALYSIS — WHERE VEILBREAK FITS

No existing work combines ALL of:
1. Rule-based deterministic encoding detection (not ML-based)
2. Deterministic decoding for Base64/leetspeak/ROT13 specifically
3. Embedding-based classification on decoded text
4. Direct benchmark comparison against Llama Guard on obfuscated inputs
5. Laptop-runnable, no paid APIs

Closest is "Recover, Decode, Reguard" but it's VLM-focused and ML-based recovery.
Broken-Token does detection but not decoding+classification.
Attack papers prove the problem but don't build the defense pipeline.
