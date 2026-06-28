# Reproducing a Depression-Severity Classifier (LT-EDI-ACL2022)

A reproduction of the winning solution to the **Shared Task on Detecting Signs of Depression from Social Media Text** at LT-EDI-ACL2022, using the original authors' code, data, and pretrained checkpoints as a starting point.

**Kaggle notebook:** [paper-opi-lt-edi-acl2022-detecting-signs-of-depr](https://www.kaggle.com/code/vidushigupta1/paper-opi-lt-edi-acl2022-detecting-signs-of-depr)

**Original paper:** Poświata & Perełkiewicz, *"OPI@LT-EDI-ACL2022: Detecting Signs of Depression from Social Media Text using RoBERTa Pre-trained Language Models"*, Proceedings of the Second Workshop on Language Technology for Equality, Diversity and Inclusion, ACL 2022. [[paper]](https://aclanthology.org/2022.ltedi-1.40/) · [[original code]](https://github.com/rafalposwiata/depression-detection-lt-edi-2022)

---

## Task

Classify a social media post into one of three depression-severity levels:

- `0` — not depressed
- `1` — moderately depressed
- `2` — severely depressed

## What this notebook does

1. Loads the authors' preprocessed competition dataset (train / dev / test CSVs).
2. Loads the authors' **already fine-tuned** model (`rafalposwiata/roberta-large-depression`) and evaluates it on the dev set as a baseline.
3. Fine-tunes a fresh copy of **DepRoBERTa** (`rafalposwiata/deproberta-large-v1` — RoBERTa-large further pretrained by the original authors on ~397K Reddit posts) on the labeled training set, using a custom PyTorch training loop.
4. Compares the reproduction against the published baseline on the same dev set.

A full write-up of the results, charts, and discussion is included as a PDF report in this repo / notebook output.

## Dataset

| Split | Rows | Not depressed | Moderate | Severe |
|---|---|---|---|---|
| Train | 6,006 | 650 | 3,101 | 2,255 |
| Dev | 1,000 | 90 | 510 | 400 |

Source: [original repo's preprocessed dataset](https://github.com/rafalposwiata/depression-detection-lt-edi-2022/tree/main/data/preprocessed_dataset).

## Setup

```bash
pip install -U transformers
```

> Note: the original repo pins `transformers==4.13.0` and `simpletransformers==0.63.7`. Those versions fail to build on current environments (old `tokenizers` wheel, legacy tokenizer loader). This reproduction uses a recent `transformers` version with a custom PyTorch training/eval loop instead of `simpletransformers`, which sidesteps both issues.

Requires a GPU (RoBERTa-large is too slow to fine-tune on CPU in any reasonable time). On Kaggle: **Settings → Accelerator → GPU T4 x2**.

## Results summary

| Metric | Authors' pretrained model | Reproduction |
|---|---|---|
| Accuracy | 0.70 | 0.63 |
| Macro F1 | 0.63 | 0.59 |
| Weighted F1 | 0.69 | 0.62 |

The reproduction (3 epochs, batch size 8, lr 1e-5, no hyperparameter search) lands within a reasonable range of the published checkpoint but underperforms it, most notably on the **severe** class (F1 0.70 → 0.56). Training loss was still decreasing at epoch 3, suggesting the model had not fully converged. See the full PDF report for per-class breakdowns and discussion.

## Repository structure

```
.
├── README.md                              # this file
├── depression_detection_results.pdf       # full results report (tables, charts, discussion)
└── paper-opi-lt-edi-acl2022-...ipynb      # the Kaggle notebook (see link above)
```

## Next steps

- Add per-epoch dev-set evaluation and early stopping.
- Train longer / tune learning rate, since loss had not plateaued.
- Apply class-weighted loss to address the "severe" class recall gap.
- Compare against a DeBERTa-v3 fine-tune as an extension beyond the original paper.

## Disclaimer

This is a research reproduction exercise on a published academic benchmark. It is **not** a validated diagnostic tool and is not intended for clinical or real-world use on individuals.

## Citation

```bibtex
@inproceedings{poswiata-perelkiewicz-2022-opi,
    title = "{OPI}@{LT}-{EDI}-{ACL}2022: Detecting Signs of Depression from Social Media Text using {R}o{BERT}a Pre-trained Language Models",
    author = "Po{\'s}wiata, Rafa{\l} and Pere{\l}kiewicz, Micha{\l}",
    booktitle = "Proceedings of the Second Workshop on Language Technology for Equality, Diversity and Inclusion",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.ltedi-1.40",
    doi = "10.18653/v1/2022.ltedi-1.40",
    pages = "276--282",
}
```
