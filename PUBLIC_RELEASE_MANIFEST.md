# Public release manifest

Proposed repository name `eeg-transfer-benchmarks`. The first commit should use the explicit inclusion list below.

## First-commit inclusion list

```text
.gitignore
LICENSE
README.md
PUBLIC_RELEASE_MANIFEST.md
requirements.txt
clinical_data.py
clinical_pilot.py
clinical_cohort.py
cohort_design.md
digit_data.py
digit_timing_audit.py
digit_benchmark.py
digit_design.md
test_digit_benchmark.py
results/clinical_cohort/calibration.png
results/clinical_cohort/features.png
results/clinical_cohort/paired_summary.csv
results/digit_cohort/performance.png
results/digit_cohort/confusion.png
results/digit_cohort/confusion.json
results/digit_cohort/erd.png
results/digit_cohort/summary.csv
```

Estimated size **under 0.6 MB** for 22 files, before Git metadata. The plots and aggregate tables are existing outputs. The five-finger performance plot shows held-out scores indexed by the original OpenNeuro numeric subject codes. It contains no clinical patient-level data.

## Saved run provenance

| Benchmark | Pinned data and validation | Full local run record |
| --- | --- | --- |
| Clinical | NEMAR `nm000309` v1.0.1. Upstream manifest SHA-256 `8f1d3e7281e98a917937c40b921e00efd14d123e363b05ed2045db4395f7ab89`. Saved run records 77 verified input files and 18 recordings. [Versioned manifest](https://data.nemar.org/nm000309/v1.0.1/manifest.json) | `results/clinical_cohort/run.json`, SHA-256 `9ec2c3d7d6c59b004878204d1a3ef62660262b0c99f1b85ee98597fb68c604b6` |
| Five-finger | OpenNeuro `ds008446` v1.0.1, Git snapshot `8d8448e2dcd731b251f2f3c85b98cce5847cb000`. Saved run records 160 EDF/event file entries, seven stimuli and 80 run audits. [Dataset snapshot](https://github.com/OpenNeuroDatasets/ds008446/tree/8d8448e2dcd731b251f2f3c85b98cce5847cb000) | `results/digit_cohort/run.json`, SHA-256 `2545660153acb6f633b067b2d5d013d50829140028f4e27f4bb9dd1bf62db1bb` |

The full run records are retained locally because they contain source identifiers, exact clinical split samples or per-run metadata. The pipelines regenerate run records in the selected output directories. The saved hashes identify the existing files and are not expected to match across arbitrary software environments.

## Exclusions and source review

- Keep raw recordings, stimuli, annex pointers and dataset metadata in `data/`. Keep `.venv/`, caches, environment files, `tmp/` and `output/` local. The expanded ignore file covers these and defaults to excluding generated results except for the eight explicit files above.
- Exclude both pilot result trees, the standalone digit timing output, all trial predictions and power time courses, and participant-level `cohort_results.csv`, `artifact_counts.csv`, `scores.csv`, and `erd.csv`. Exclude both full run JSONs. The included CSVs and confusion JSON contain aggregate output.
- Exclude the unrelated Unified V1 handoff request, the internal portfolio audit, and the historical cohort reports. The design files remain in the release. The historical reports link to excluded outputs.
- The inspected scripts use relative data and output paths and public HTTPS endpoints. No credential, token, absolute personal path, or vendored dependency was found in the selected source files. The clinical loader contains public NEMAR recording identifiers used to verify the cohort mapping. The digit loader contains public subject and run naming conventions.
- The analysis appears project-specific. No third-party source block, upstream copyright header or incompatible code license was identified. The Damm timing convention is credited in the README. NumPy, SciPy, scikit-learn and MNE are imported dependencies with BSD licenses, and Matplotlib carries its own PSF-derived license. Dependencies are installed separately. The checkout has no commits, so Git cannot independently establish ownership. The included MIT text and copyright line are proposed for owner review. Dataset terms remain [CC BY 4.0 for NEMAR](https://nemar.org/dataset/nm000309?v=v1.0.1) and [CC0 for OpenNeuro](https://raw.githubusercontent.com/OpenNeuroDatasets/ds008446/8d8448e2dcd731b251f2f3c85b98cce5847cb000/dataset_description.json).
