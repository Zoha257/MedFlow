# Reproducible research workflows

The project uses versioned Python modules rather than checked-in notebooks so
that data download, training, evaluation, and figure generation can run in CI,
Docker, or a clean terminal without hidden notebook state.

The handover's planned notebook workflows map to these commands:

| Planned workflow | Reproducible command |
| --- | --- |
| EDA and dataset validation | `python -m src.data.validate` |
| Baseline training and evaluation | `python -m src.models.train --epochs 1 --batch-size 1024` |
| ResNet and augmentation comparison | `python -m src.models.train_day3 --epochs 1 --batch-size 1024 --experiment all` |
| Grad-CAM analysis | `python -m src.models.generate_gradcam` |
| Research figures | `python -m src.research.generate_artifacts` |

All generated outputs are written to the documented `data/`, `models/`,
`results/`, `paper/figures/`, and `research/` paths. This avoids notebook
outputs becoming stale or masking the actual command-line pipeline.
