# Evaluation

Phase 2 provides a pre-fine-tuning clause-type classification harness. CUAD's
41 categories are authoritative labels. It does not evaluate legal risk,
generate explanations, or provide legal advice.

## Running baselines

```bash
python -m clauseforge.evaluation.runner \
  --data data/processed/cuad/1.0.0-run-a \
  --output eval/results/phase2 \
  --model all \
  --split validation
```

Supported models are `majority`, `rules`, `tfidf-logreg`, `tfidf-svm`, and
`all`. Use validation for development. Test evaluation is reserved for a
configuration locked from validation results. Generated files under
`eval/results/` are ignored.

## Methodology

Macro F1 is primary because CUAD is highly imbalanced and every authoritative
category should contribute equally. Accuracy and weighted F1 are retained to
show aggregate behavior. Recall@k is the fraction of examples whose true label
appears among the k highest ranked class scores; it is reported only where a
baseline provides a complete deterministic score ranking.

Confidence intervals use 500 seeded percentile-bootstrap iterations at the
contract level. Resampling contracts preserves within-contract correlation.
ECE, multiclass Brier score, and reliability bins are produced only for
meaningful probabilities: training priors for the majority baseline and
`predict_proba` for logistic regression. Rule scores and LinearSVC margins are
not presented as probabilities.

The keyword baseline extracts literal tokens from the quoted CUAD category
name, ignores a small fixed stopword set, counts keyword presence, and resolves
ties with training-set class priors and lexical order. It is not tuned on test
errors.

TF-IDF uses lowercase Unicode-normalized word 1-2 grams, `min_df=2`,
`max_df=0.995`, sublinear term frequency, and at most 100,000 features. Both
linear models use balanced class weights and seed 42. Validation-only selection
over C={0.5, 1.0, 2.0} chose C=2.0 for logistic regression and C=0.5 for
LinearSVC.

## Actual held-out test results

These measurements were generated locally from the 51-contract held-out test
split after configuration lock.

| Model | Accuracy | Macro F1 | Weighted F1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|---:|---:|
| Majority | 0.180251 | 0.007450 | 0.055057 | 0.268809 | 0.369122 |
| Keyword rules | 0.316614 | 0.201389 | 0.297861 | 0.493730 | 0.563480 |
| TF-IDF logistic | 0.779781 | 0.672683 | 0.776799 | 0.938088 | 0.968652 |
| TF-IDF LinearSVC | 0.786050 | 0.665255 | 0.777295 | 0.938088 | 0.956113 |

Machine-readable metrics, intervals, calibration, confusion matrices, and
errors are generated locally in `eval/results/phase2-final/`.

## Safety and robustness harness

Versioned synthetic public-safe fixtures live in `eval/fixtures/`.

```bash
python scripts/run_safety_eval.py --provider mock --output eval/results/safety
python scripts/run_safety_eval.py --provider classical --output eval/results/safety-rules
```

The runner writes summary, adversarial/paraphrase JSONL, taxonomy metadata, run
configuration, and Markdown artifacts beneath ignored `eval/results/`. Outputs
are **DEVELOPMENT SAFETY HARNESS RESULTS — NOT FINAL MODEL PERFORMANCE**. Final
trained-model safety performance remains pending Phase 3B GPU execution.

## SEC EDGAR unlabeled OOD evaluation

EDGAR segments are unlabeled and separate from CUAD. Valid future measurements
include taxonomy-valid and invalid-output rates, prediction and genuine-score
confidence distributions, abstention rate, and processing failures. Accuracy,
precision, recall, and F1 must not be reported without authoritative labels.
