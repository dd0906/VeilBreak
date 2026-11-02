# Papers and the ideas they gave VeilBreak

Each entry below says what the source contributed to this project. Sources are split honestly: ideas we **used**, work we only found **afterwards**, and ideas **not built yet**.

Everything was checked against the source in November 2026. The "Read" column says how much: *abstract*, *partial* (part of the full text), *full post*, or *excerpts*. Many are recent preprints, and peer review is noted only where the source states it.

## Ideas we used

| Source | The idea | What it became in VeilBreak | Read |
|---|---|---|---|
| [Wei, Haghtalab, Steinhardt 2023, *Jailbroken*](https://arxiv.org/abs/2307.02483) | *Mismatched generalization*: safety training can miss a domain where the model is still capable. | The core framing: a guard should see what the model can read. Also the question behind the Exposure Map. | Abstract |
| [Huang 2024, *Plentiful Jailbreaks with String Compositions*](https://arxiv.org/abs/2411.01084) (NeurIPS SoLaR Workshop) | Invertible encodings can be stacked. | The decoder loop that peels up to 3 layers, and the `composed` test variant. | Abstract |
| [Priyanshu 2024, Robust Intelligence / Cisco](https://blogs.cisco.com/security/bypassing-metas-llama-classifier-a-simple-jailbreak) | Spacing out characters dropped **Prompt-Guard-86M**'s accuracy from 100% to 0.2% on 450 prompts. | The motivation to test a real guard empirically instead of assuming it fails. | Full post |
| [Hackett et al. 2025, *Bypassing LLM Guardrails*](https://arxiv.org/abs/2504.11168) | Character injection beats guardrails: emoji smuggling reached 100% success, Unicode tags 90.15% on injections and 81.79% on jailbreaks. | The four invisible-character attacks and decoders: tag and emoji smuggling, zero-width, look-alikes. | Partial |
| [OWASP LLM01:2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) | Scenario #9 covers instructions encoded in Base64 or emojis, and injections can be imperceptible to humans. | The default-deny sanitizer, which closes one precisely defined part of the risk, and how we scope our claims. | Full page summary |
| [Handa et al., *When "Competency" in Reasoning Opens the Door to Vulnerability*](https://arxiv.org/abs/2402.10601) (arXiv v5, 2025) | CipherBench measures decoding on benign text, across models. | The design of the **Exposure Map**, with harmless prompts only. We added invisible-character attacks, a size sweep and a link to guard results. | Partial |
| [Zhang et al. 2026, *Blind, Not Weak*](https://arxiv.org/abs/2607.26574) (first titled *Recover, Decode, Reguard*) | Defenses that decode for a guard carry a safety-utility cost, so measure both. | Reporting false alarms next to every catch rate. | Partial |
| [Inan et al. 2023, *Llama Guard*](https://arxiv.org/abs/2312.06674) | A dedicated guard model for prompts and responses. | The guard we benchmark against. | Abstract |
| [Chao et al., *JailbreakBench*](https://arxiv.org/abs/2404.01318) (NeurIPS 2024 Datasets and Benchmarks) | 100 harmful behaviors, partly reused from earlier work. | Harmful prompts in the benchmark, and its benign set became the hard-benign test. | Abstract |
| [Zou et al. 2023, AdvBench](https://arxiv.org/abs/2307.15043) | A public list of harmful behaviors. | Most of the harmful prompts in the benchmark. | Data file |
| [Röttger et al., *XSTest*](https://arxiv.org/abs/2308.01263) (NAACL 2024) | Safe prompts that can look unsafe, and unsafe contrasts. | Extra training data in our experiment. It didn't fix hard-benign false alarms. | Abstract |
| [Mazeika et al. 2024, *HarmBench*](https://arxiv.org/abs/2402.04249) | A varied harmful-behavior benchmark. | Extra training data; lifted JailbreakBench catch, with a false-alarm cost. | Abstract |
| [AWS Bedrock Guardrails guidance](https://aws.amazon.com/blogs/security/protect-your-generative-ai-applications-against-encoding-based-attacks-with-amazon-bedrock-guardrails) | Decoding everything can add false positives on legitimate content like product codes and code examples. | Supported testing false-alarm costs on non-instruction content. | Search summary |

## Found afterwards: not idea sources

We found these after building the feature, or while checking whether our work was new. They're listed because they support or limit our claims, and we don't present them as inspiration.

| Source | What it shows | How it relates | Read |
|---|---|---|---|
| [Zhang et al. 2026, *Whose Refusal Is It?*](https://arxiv.org/abs/2608.08641) | Separates guard refusals from model refusals. Lists benign traffic that resembles an encoding as uncovered, and measures benign utility only on unencoded prompts. | **Supports our most promising gap:** false alarms on harmless encoded content. | Partial |
| [Orgad et al. 2026](https://arxiv.org/abs/2604.09544) | No tested model decoded ROT13; Llama-3.1-8B couldn't decode Base64 (a side result). | Independent support for the Exposure Map, and shows that part isn't new. | Partial |
| [Li et al. 2025, *CipherBank*](https://arxiv.org/abs/2504.19093) | 2,358 decryption problems on reasoning. | Background on decoding ability. | Abstract |
| [Fedorov et al. 2024, *Llama Guard 3-1B-INT4*](https://arxiv.org/abs/2411.17713) | A 440 MB derivative reporting scores comparable to or better than the larger model. | Overall scores; our build comparison looks at encoded input and a different build. | Abstract |
| *How Lightweight Can LLM Guardrails Become? Evaluating Quantized Llama Guard 3* ([ResearchGate](https://www.researchgate.net/publication/415821578_How_Lightweight_Can_LLM_Guardrails_Become_Evaluating_Quantized_Llama_Guard_3_for_Efficient_AI_Safety)) | GGUF builds Q8 to Q3 against full precision. Excerpts suggest careful quantization can keep effectiveness. | **Closest to our build comparison. Get the full text before calling that finding new.** | Excerpts |
| [Shahin, Alsmadi 2026](https://arxiv.org/abs/2601.19970) | 100 adversarial prompts: Llama-Guard-3-8B 33%, INT8 28%, Llama-Guard-3-1B 76%. Encoded prompts aren't reported separately. | Related build comparison, small sample. | Partial |
| [Chen et al. 2025, *Q-resafe*](https://arxiv.org/abs/2506.20251) (ICML 2025); [Kharinaev et al. 2025](https://arxiv.org/abs/2502.15799) | Quantization can weaken LLM safety. | General background. | Abstract |
| [Kanwat 2026, *Fooling LLM Safety Guards into Judging the Wrong Text*](https://arxiv.org/abs/2610.31695) | Forged query and document pairs make guards grade a harmless decoy, and scoring segments and taking the maximum helps. | Similar in spirit to our judging payloads separately. We got there independently, through a wrapper dilution result. | Partial |
| Open-source tools: [palisade](https://github.com/RootVandal/palisade), [hidden-text-kit](https://github.com/rajatslakhina/hidden-text-kit), [GlyphGuard](https://github.com/NicPWNs/GlyphGuard), [disarm](https://github.com/raeq/disarm/issues/1143) | Detect, decode or normalize hidden Unicode. | **Our decoders and sanitizer aren't new ideas.** | Descriptions only |

## Ideas not built yet

Candidates for future features, each from a source we read:

| Idea | Source | What it could add | Effort and catch |
|---|---|---|---|
| **Characters-per-token detector** | [Broken-Token](https://arxiv.org/abs/2510.26847) (Zychlinski, Kainan 2025; partial) | Spot encoded blobs we can't decode, so a policy can flag them. | Small. Reports 99.1–99.9% accuracy but no separate false-positive rates, and code and hashes are untested. |
| **Decoding that uses an extra model** | [Blind, Not Weak](https://arxiv.org/abs/2607.26574) | Handle ciphers rules can't. | Large. It fails when the model can't read the encoding, and it can lower block rates: LlamaGuard-3 on CodeAttack fell from about 92% to 12%. |
| **Separating guard and model refusals** | [Whose Refusal Is It?](https://arxiv.org/abs/2608.08641) | Measure what the guard itself contributes in a full chatbot pipeline. | Medium. Needs a target model and harmful-output handling. |
| **Testing more Llama Guard builds** | The quantized Llama Guard 3 paper, [threshold-shift project](https://github.com/zarif007/Quantized-OSS-GuardRails-Extended) (small, unreviewed) | Test Q4, Q5 and Q8 in one software stack, to find out whether quantization causes the divergence. | Medium. Would directly answer our open question. |
| **Forged-decoy attack as a test** | [Kanwat 2026](https://arxiv.org/abs/2610.31695) | Check whether our payload-separate rule resists target rebinding. | Small to medium. Outside the encoded-attack scope. |
| **A cheaper classifier from a pruned model** | [Sawtell et al. 2024](https://arxiv.org/abs/2412.13435) (partial) | Try intermediate-layer classification instead of MiniLM, which may help hard-benign prompts. | Medium. Reports strong scores, but on a different dataset and with no encoded inputs. |
| **OR-Bench-Hard as a benign test** | Used by [Blind, Not Weak](https://arxiv.org/abs/2607.26574) | A stronger hard-benign set than our 100 prompts. | Small. |
| **Attacks on larger models** | [Handa et al.](https://arxiv.org/abs/2402.10601), [Orgad et al.](https://arxiv.org/abs/2604.09544) | Extend the Exposure Map past 3B, where models start reading these encodings. | Small to medium. Harmless prompts only, and limited by GPU memory. |

## Where this leaves our claims

- **Most promising:** false alarms on harmless encoded content. Not measured in the closest work we read, but our search was limited.
- **Narrow:** build disagreement on encoded input. Cause unproven, and one close paper is still unread in full.
- **Mostly confirmation:** the Exposure Map's small-model result. Our additions are the size sweep, invisible-character attacks and the guard link.
- **Not new:** decoding before a filter, and removing invisible characters.

Out of scope: [Li et al. 2024, *StructuralSleight*](https://arxiv.org/abs/2406.08754) combines unusual text structures with obfuscation, a different attack class.
