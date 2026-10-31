"""Extra baselines and best-of-N confidence intervals reported in the paper (Table V, Section VI-B).

Runs on the saved outputs in ../results/ (no GPU needed):
    python analysis/extra_baselines.py

Computes:
  - answer-length baseline (pooled and within-question AUROC)
  - TF-IDF text-only baseline (5-fold CV grouped by question, within the test set)
  - best-of-N accuracy with paired bootstrap CIs (probe vs greedy, probe vs majority vote, NF4 vs FP16)
  - mixed-correctness counts
"""
import os
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def normalize(s):
    s = str(s).lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def wq_auc(scores, labels, qids):
    """Within-question AUROC: compare only correct vs incorrect answers to the same question."""
    num, den = 0.0, 0
    df = pd.DataFrame({"s": scores, "y": labels, "q": qids})
    for _, g in df.groupby("q"):
        pos, neg = g.s[g.y == 1].values, g.s[g.y == 0].values
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None, :]
            num += (d > 0).sum() + 0.5 * (d == 0).sum()
            den += d.size
    return num / den if den else float("nan")


def main():
    t = pd.read_csv(os.path.join(RES, "test_scores.csv"))
    a = pd.read_csv(os.path.join(RES, "answers.csv"))
    y, q = t.label.values, t.qid.values

    # Length baseline (shorter answer = more likely correct)
    L = t.answer.astype(str).str.len().values
    print(f"Length baseline   pooled {roc_auc_score(y, -L):.3f}   within-question {wq_auc(-L, y, q):.3f}")

    # TF-IDF text-only baseline
    sc = np.zeros(len(t))
    text = t.answer.astype(str).values
    for tr, te in GroupKFold(5).split(t, y, q):
        vec = TfidfVectorizer(ngram_range=(1, 2))
        clf = LogisticRegression(max_iter=2000).fit(vec.fit_transform(text[tr]), y[tr])
        sc[te] = clf.decision_function(vec.transform(text[te]))
    print(f"TF-IDF baseline   pooled {roc_auc_score(y, sc):.3f}   within-question {wq_auc(sc, y, q):.3f}")

    # Best-of-N with paired bootstrap over test questions
    qs = np.unique(q)

    def probe_pick(col):
        return t.loc[t.groupby("qid")[col].idxmax()].set_index("qid").label

    def majority_vote():
        out = {}
        for qq, g in t.groupby("qid"):
            norm = g.answer.map(normalize)
            top = Counter(norm).most_common(1)[0][0]
            out[qq] = g.label.values[(norm == top).values][0]
        return pd.Series(out)

    pf, pn, mv = probe_pick("s_fp16"), probe_pick("s_nf4"), majority_vote()
    greedy = a[a["sample"] == -1].set_index("qid").label.reindex(qs).fillna(0)
    oracle = t.groupby("qid").label.max()
    print(f"\nBest-of-N accuracy: greedy {greedy.mean():.3f}  majority {mv.mean():.3f}  "
          f"probe FP16 {pf.mean():.3f}  probe NF4 {pn.mean():.3f}  oracle {oracle.mean():.3f}")

    rng = np.random.RandomState(0)
    diffs = {"probe FP16 - greedy": [], "probe FP16 - majority": [], "probe NF4 - probe FP16": []}
    for _ in range(5000):
        s = rng.choice(qs, len(qs), replace=True)
        diffs["probe FP16 - greedy"].append(pf[s].mean() - greedy[s].mean())
        diffs["probe FP16 - majority"].append(pf[s].mean() - mv[s].mean())
        diffs["probe NF4 - probe FP16"].append(pn[s].mean() - pf[s].mean())
    for k, d in diffs.items():
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {k:24s} {np.mean(d):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    # Mixed-correctness counts
    s = a[a["sample"] >= 0]
    m = s.groupby("qid").label.mean()
    print(f"\nQuestions: all correct {(m == 1).sum()}, all wrong {(m == 0).sum()}, "
          f"mixed {((m > 0) & (m < 1)).sum()} (of {len(m)})")


if __name__ == "__main__":
    main()
