"""
Download the raw datasets into data/raw/ (skips files you already have).

  advbench.csv        520 harmful behaviours   (Zou et al. 2023, GitHub)
  alpaca.csv          52k benign instructions  (Stanford Alpaca, HuggingFace)
  jailbreakbench.csv  100 harmful behaviours   (JailbreakBench, HuggingFace)
  jbb_benign.csv      100 BENIGN behaviours written to mirror the harmful ones
                      (same topics, same style) -> the hardest benign test set
  xstest.csv          safe prompts that sound unsafe + unsafe look-alikes
                      (Rottger et al. 2024, CC-BY-4.0). Extra TRAINING data only.
  harmbench.csv       'standard' harmful behaviours (Mazeika et al. 2024, MIT).
                      Extra TRAINING data only.

All files share one schema: prompt, source, label.

Needs internet. If a download fails the script says which file to fetch by
hand; build_dataset.py works with whatever is present (AdvBench + Alpaca are
the minimum).

Usage:  python scripts/collect_datasets.py
"""
import os
import urllib.request

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW, exist_ok=True)


def save(prompts, source, label, name):
    df = pd.DataFrame({"prompt": [str(p).strip() for p in prompts if str(p).strip()],
                       "source": source, "label": label})
    df.to_csv(os.path.join(RAW, name), index=False)
    print(f"  saved {len(df)} prompts -> data/raw/{name}")


def have(name):
    if os.path.exists(os.path.join(RAW, name)):
        print(f"  {name} already present, skipping")
        return True
    return False


def advbench():
    if have("advbench.csv"):
        return
    url = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
           "data/advbench/harmful_behaviors.csv")
    df = pd.read_csv(url)
    save(df["goal"], "advbench", "unsafe", "advbench.csv")


def alpaca():
    if have("alpaca.csv"):
        return
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca")["train"]
    prompts = [f"{r['instruction']} {r['input']}".strip() for r in ds]
    save(prompts, "alpaca", "safe", "alpaca.csv")


def jailbreakbench():
    if have("jailbreakbench.csv") and have("jbb_benign.csv"):
        return
    from datasets import load_dataset
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    # The dataset has two splits: "harmful" and "benign" (100 rows each).
    save(ds["harmful"]["Goal"], "jailbreakbench", "unsafe", "jailbreakbench.csv")
    save(ds["benign"]["Goal"], "jbb_benign", "safe", "jbb_benign.csv")


def xstest():
    if have("xstest.csv"):
        return
    url = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
    df = pd.read_csv(url)
    pd.DataFrame({"prompt": df["prompt"].str.strip(), "source": "xstest",
                  "label": df["label"]}).to_csv(os.path.join(RAW, "xstest.csv"), index=False)
    print(f"  saved {len(df)} prompts -> data/raw/xstest.csv")


def harmbench():
    if have("harmbench.csv"):
        return
    url = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
           "data/behavior_datasets/harmbench_behaviors_text_all.csv")
    df = pd.read_csv(url)
    # 'contextual' needs extra context text and 'copyright' is a different problem.
    save(df[df["FunctionalCategory"] == "standard"]["Behavior"], "harmbench", "unsafe", "harmbench.csv")


def main():
    print("VeilBreak - dataset collection")
    for name, fn in (("AdvBench", advbench), ("Alpaca", alpaca),
                     ("JailbreakBench", jailbreakbench),
                     ("XSTest", xstest), ("HarmBench", harmbench)):
        print(f"[{name}]")
        try:
            fn()
        except Exception as e:                                  # noqa: BLE001
            print(f"  FAILED: {e}\n  -> fetch it manually (see docstring) and re-run.")


if __name__ == "__main__":
    main()
