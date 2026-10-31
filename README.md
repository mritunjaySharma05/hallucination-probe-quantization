# Are Hallucination Probes Evaluated Correctly?

Code and results for the pilot study in:

> **Mritunjay Sharma.** *Are Hallucination Probes Evaluated Correctly? A Critical Review and a Pilot Study under 4-bit Quantization.* Preprint, 2026. [Link to be added]

Hallucination probes are small classifiers trained on a language model's hidden states to predict whether its answer is correct. This repository tests whether such probes still work when the model is quantized, using two metrics side by side:

- **Pooled AUROC**: compares answers across all questions. It can look high simply because the probe recognises hard questions.
- **Within-question AUROC**: compares only right and wrong answers *to the same question*. This is what matters for picking the best answer.

## Key results

Qwen2.5-1.5B-Instruct, 1,000 TriviaQA questions, 10 sampled answers each. Test set: 200 questions (2,000 answers).

| Metric | FP16 | INT8 | NF4 |
| --- | --- | --- | --- |
| Greedy accuracy | 0.320 | 0.322 | 0.292 |
| Answer probe, pooled AUROC | 0.852 | 0.855 | 0.834 |
| Answer probe, within-question AUROC | 0.764 | 0.792 | 0.760 |
| Question-only probe, pooled AUROC | 0.726 | 0.726 | 0.740 |
| Log-probability, within-question AUROC | 0.603 | 0.618 | 0.587 |

- A probe that **never sees the answer** reaches 0.73 pooled AUROC, so most of the pooled score reflects question difficulty.
- A genuine answer-level signal exists (within-question AUROC 0.76), stronger than log-probability and sample agreement.
- 4-bit (NF4) quantization caused **no measurable loss**: within-question AUROC changed by −0.002 (95% CI −0.08 to +0.09). Only 58 test questions had mixed correctness, so only large drops can be ruled out.
- A probe trained at FP16 works on NF4 hidden states without retraining.
- Choosing the answer the probe trusts most beat greedy decoding by 6.0 points (95% CI 1.5 to 11.0).

## Repository contents

```
quant_probe_mve.ipynb        Full experiment (generation, labelling, probes, bootstrap)
analysis/extra_baselines.py  Length and TF-IDF baselines, best-of-N confidence intervals
results/
  results.json               All metrics: per-layer scores, transfer matrix, bootstrap CIs
  summary.csv                Main table above
  answers.csv                Every generated answer (greedy + 10 samples) with its label
  test_scores.csv            Probe scores for each test answer at FP16, INT8 and NF4
paper/                       LaTeX source of the paper
```

Saved hidden-state arrays (`XA_*.npy`, a few hundred MB) are not included; the notebook recreates them.

## How to reproduce

**Full experiment (GPU, about 40 minutes on a Kaggle T4 in our run):**

1. On Kaggle, create a notebook and import `quant_probe_mve.ipynb`.
2. Set Accelerator to **GPU T4 x1** and Internet to **On**.
3. Click **Run All**. Outputs are written to `/kaggle/working/`.

No Hugging Face token is needed; the model is not gated.

**Extra baselines (CPU only, seconds):**

```bash
pip install -r requirements.txt
python analysis/extra_baselines.py
```

## Setup details

| Setting | Value |
| --- | --- |
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| Data | TriviaQA `rc.nocontext` validation, 1,000 random questions (seed 0) |
| Sampling | 1 greedy + 10 sampled answers, temperature 0.7, top-p 0.95 |
| Labels | Normalized exact match or contains any TriviaQA alias (not hand-audited) |
| Precisions | FP16, INT8 (bitsandbytes LLM.int8), NF4 (bitsandbytes, double quantization) |
| Design | Teacher-forced: the same FP16 answers are fed through each precision |
| Features | Residual stream at last answer token (answer probe) and last prompt token (question-only probe), every 4th layer |
| Probe | L2 logistic regression; C and layer chosen on validation (layer 16 at all precisions) |
| Split | Grouped by question, 60/20/20, shared across precisions |
| Statistics | Paired bootstrap over test questions |

## Limitations

One 1.5B model, one dataset, rule-based labels, and a small number of mixed-correctness test questions. Results are a pilot, not a general conclusion. See Section VI-C of the paper.

## Citation

```bibtex
@misc{sharma2026probes,
  author = {Sharma, Mritunjay},
  title  = {Are Hallucination Probes Evaluated Correctly? A Critical Review and a Pilot Study under 4-bit Quantization},
  year   = {2026},
  note   = {Preprint}
}
```

## License

Code is released under the MIT License. TriviaQA and Qwen2.5 are subject to their own licenses.
