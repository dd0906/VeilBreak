"""Integrity checks on the built evaluation dataset (skipped if not built yet)."""
import os

import pandas as pd
import pytest

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "veilbreak_v2.csv")
pytestmark = pytest.mark.skipif(not os.path.exists(PATH), reason="run scripts/build_dataset.py first")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(PATH)


def test_ids_unique_and_nine_variants_per_group(df):
    assert df.id.is_unique
    assert (df.groupby("group_id").size() == 9).all()


def test_no_group_spans_dev_and_eval(df):
    assert (df.groupby("group_id").split.nunique() == 1).all()


def test_no_prompt_text_shared_between_dev_and_eval(df):
    plain = df[df.variant == "plain"]
    assert not (set(plain[plain.split == "dev"].text.str.lower())
                & set(plain[plain.split == "eval"].text.str.lower()))


def test_benign_prompts_are_obfuscated_too(df):
    safe = df[df.label == "safe"]
    assert set(safe.variant) == set(df.variant)


def test_classes_balanced_in_eval(df):
    g = df[df.split == "eval"].drop_duplicates("group_id")
    assert abs((g.label == "unsafe").sum() - (g.label == "safe").sum()) <= 120
