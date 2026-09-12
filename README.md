# Reproduction of TGN-Based Change Propagation Prediction

> [!IMPORTANT]
> This repository is an **independent academic reproduction** of the experiments and implementation published by **Manuella Germanos, Danielle Azar, and Eileen Marie Hanna**. The underlying research idea, experimental method, original model, and original dataset belong to and must be credited to the original authors.
>
> My work does **not** claim authorship of the original method. It is intended only to reproduce, study, document, and evaluate the published experiment. This repository is not an official release from the original authors, and they have not necessarily reviewed, validated, or endorsed the changes or results presented here.

## Original work

This reproduction is based on the following paper:

> Manuella Germanos, Danielle Azar, and Eileen Marie Hanna, “To change or not to change? Modeling software system interactions using Temporal Graphs and Graph Neural Networks: A focus on change propagation,” *Information and Software Technology*, vol. 166, article 107368, 2024.  
> DOI: [10.1016/j.infsof.2023.107368](https://doi.org/10.1016/j.infsof.2023.107368)

Original authors’ repository:

- [MGermanos/TGN-change-propagation](https://github.com/MGermanos/TGN-change-propagation)

This reproduction repository:

- [maheshgprasad/TGN-change-propagation](https://github.com/maheshgprasad/TGN-change-propagation)

## Purpose of this repository

The purpose of this repository is to reproduce the published change-propagation experiment as part of academic study. The original approach:

- represents software files as nodes in a temporal graph;
- represents historical co-changeability between files as edges;
- uses a Temporal Graph Network (TGN) with a Long Short-Term Memory (LSTM) model; and
- predicts other files that may be impacted when a file is modified.

The reproduction is intended to examine whether the reported experimental workflow and results can be recreated in a separate environment. Any execution changes, compatibility adjustments, scripts, documentation, visualizations, or independently generated results in this repository should be understood as reproduction support—not as a new claim over the original model or methodology.

## Provenance and contribution boundaries

| Item | Attribution |
| --- | --- |
| Research problem and proposed method | Germanos, Azar, and Hanna |
| TGN + LSTM change-propagation architecture | Germanos, Azar, and Hanna |
| Original experimental datasets and shuffle organization | Germanos, Azar, and Hanna |
| Original implementation (`TGN_model.py`) | Original authors’ repository |
| Reproduction execution, environment adjustments, documentation, and independently generated outputs | Mahesh G. Prasad |

Where a file has been copied or adapted from the original repository, its history and source should be preserved wherever practical. Substantial modifications should be identified in the file, commit history, or accompanying documentation.

## Reproduction status

Results produced from this repository are **reproduction results**. They may differ from those reported in the paper because of factors such as:

- operating system and hardware;
- Python, TensorFlow, and dependency versions;
- random initialization and nondeterministic operations;
- thread configuration and numerical precision; and
- local execution or compatibility changes.

A successful execution should therefore not be interpreted as an exact independent reimplementation unless explicitly stated. Likewise, reproduced metrics should not be presented as results generated or certified by the original authors.

## Requirements

The original repository specifies Python 3.8.8 or later and the following core package versions:

```text
pandas==1.4.3
numpy==1.19.5
tensorflow==2.14.1
```

Compatibility may depend on the operating system, Python version, CPU/GPU configuration, and package availability. Record any deviations from the original environment when reporting reproduction results.

## Running the reproduction

Clone this repository and create an isolated Python environment:

```bash
git clone https://github.com/maheshgprasad/TGN-change-propagation.git
cd TGN-change-propagation
python -m venv .venv
```

Activate the environment and install the dependencies appropriate for your platform. If the reproduction script in your checkout exposes the command-line interface, a run can be started in the following form:

```bash
python updated_TGN.py \
  --projects ant \
  --shuffle-indices 0 \
  --threads 8 \
  --results-root Results \
  --verbose
```

Use `python updated_TGN.py --help` to see the options supported by the checked-out version. For the original execution procedure and parameter placement, refer directly to the [original repository](https://github.com/MGermanos/TGN-change-propagation).

## Dataset organization

The original dataset contains chronological change sets for the evaluated software systems. Multiple dataset variants shuffle the order of files **within the same commit**, while preserving the chronological order of commits.

The original repository describes the dataset hierarchy in the following form:

```text
ShuffledData/
└── Iterator (0 to 4)/
    └── ChangeSets/
        └── <software-system>/
```

Please cite the original paper and repository when using these datasets.

## Outputs

Following the original experiment, outputs may include:

- per-run evaluation metrics under `Results/Metrics/`; and
- per-change-set confusion-matrix values under `Results/ConfMatrix/`.

The reported metrics can include sensitivity, specificity, positive predictive value (PPV), geometric mean, F1 score, accuracy, Matthews correlation coefficient (MCC), and area under the ROC curve (AUC).

## Phase 1 extension on this branch

The `phase1-clean-attention` branch adds a controlled Phase-1 improvement while leaving the reproduced Germanos implementation intact.

The improvement replaces repeated candidate-specific LSTM fitting during DFS prediction with one shared causal temporal-attention scorer that is trained once and reused.

It adds:

- richer historical co-change and activity features;
- commit-recency and real elapsed-time features;
- recent source interaction history;
- validation-based threshold selection; and
- persisted decision traces for explaining false positives, false negatives, and prediction-size behaviour.

The graph construction, seed selection, DFS traversal, mu filtering, rho prediction cap, and confusion-matrix accounting remain aligned with the reproduced baseline.

### Shuffle-0 evaluation

Run all ten baseline repositories:

```bash
python phase1_clean/run_all.py --shuffle 0 --device cpu --continue-on-error
```

### Robustness across shuffles 1-4

The remaining four shuffles change file ordering within commits. They are used as robustness tests rather than as independent datasets.

The shuffle-0 model, preprocessing, threshold, and prediction cap are frozen and reused:

```bash
python phase1_clean/evaluate_frozen_shuffles.py \
  --device cpu \
  --continue-on-error
```

No retraining or threshold recalibration is performed for shuffles 1-4.

### Visualization

Use the existing Streamlit dashboard:

```bash
streamlit run visualization/app.py
```

The interface is intentionally limited to:

1. **Baseline results**
2. **Clean attention evidence**
3. **Input graph**

Detailed candidate traces and threshold plots are available only as optional drill-down views.

### Metric terminology

The reproduced implementation labels `(Sensitivity + Specificity) / 2` as AUC. In this project it is reported as **Legacy AUC / Balanced Accuracy** to distinguish it from true ROC-AUC.

## Citation

If you use this repository, reproduce the experiment, or use the original datasets or method, cite the original paper:

```bibtex
@article{GERMANOS2024107368,
  title   = {To change or not to change? Modeling software system interactions using Temporal Graphs and Graph Neural Networks: A focus on change propagation},
  journal = {Information and Software Technology},
  volume  = {166},
  pages   = {107368},
  year    = {2024},
  issn    = {0950-5849},
  doi     = {10.1016/j.infsof.2023.107368},
  url     = {https://www.sciencedirect.com/science/article/pii/S0950584923002239},
  author  = {Manuella Germanos and Danielle Azar and Eileen Marie Hanna}
}
```

When referring specifically to modifications or reproduction results from this repository, link to this repository in addition to citing the original paper. A repository link is supplementary and does not replace the scholarly citation above.

## Acknowledgements

I sincerely thank **Manuella Germanos, Danielle Azar, and Eileen Marie Hanna** for publishing their research and making the accompanying implementation and datasets available. This reproduction would not have been possible without their work.

## License and reuse

This repository may contain or adapt material originating from the original authors’ repository. Reuse is subject to the applicable rights and license terms of the original material, if any, as well as the terms applied to independently authored additions in this repository.

Do not assume that the presence of publicly accessible source code grants permission beyond the rights stated by its copyright holder. Before redistributing or relicensing adapted code or datasets, verify the licensing terms in the original repository or obtain permission from the relevant rights holders.

## Disclaimer

This repository is provided for research and educational purposes. Reproduction results may contain errors and should be independently verified. The original authors are not responsible for modifications, execution choices, interpretations, or results produced from this repository.

