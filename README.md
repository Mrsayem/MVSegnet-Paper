# MVSegNet for Fetal Ventricle Segmentation

This notebook contains the working implementation I used for fetal ventricle segmentation experiments. It covers the full pipeline in one place: downloading and organizing the data, preprocessing the masks, training MVSegNet, evaluating the trained checkpoints, and running the ablation study.

The code is written mainly for Google Colab, although it can also be run locally with a CUDA-enabled PyTorch setup.

## What is in the notebook

The notebook works with the trans-ventricular ultrasound data downloaded from Zenodo record **8265464**:

https://zenodo.org/records/8265464

The pipeline does the following:

1. downloads and extracts the dataset;
2. pairs ultrasound images with their segmentation masks;
3. resizes the data to `512 x 512`;
4. creates edge maps and auxiliary masks;
5. splits the data into train, validation, and test sets;
6. trains MVSegNet with three random seeds;
7. evaluates segmentation quality and model complexity;
8. generates result tables and qualitative figures;
9. runs a seven-stage ablation study.

In the recorded notebook run, the final preprocessed data loaded by the training/evaluation code contained:

| Split | Samples |
| --- | ---: |
| Train | 235 |
| Validation | 57 |
| Test | 55 |

The preprocessing code starts from a nominal `70/15/15` split. Samples with very small or empty masks are skipped during preprocessing, so the final counts do not follow the percentages exactly.

## Model

MVSegNet uses a **MobileNetV3-Small** encoder with a U-Net-style decoder. The notebook adds several modules around this backbone:

- attention gates on the skip connections;
- MSAM (multi-scale ventricle attention);
- an adaptive boundary refinement branch (ABRB);
- an edge prediction head;
- two auxiliary segmentation heads for deep supervision;
- an optional FiLM layer intended for gestational-age conditioning.

The main segmentation output is binary.

The loss used in the notebook is based on BCE and Tversky loss, with smaller auxiliary terms for edge, boundary, and deep-supervision losses.

For the full-loss setting:

```text
main segmentation = 0.5 * BCE + 0.5 * Tversky
edge weight        = 0.03  (enabled from epoch 5)
auxiliary weight   = 0.10
boundary weight    = 0.01  (enabled from epoch 10)
```

## Training setup

The main experiment uses the following settings:

| Setting | Value |
| --- | --- |
| Input size | 512 x 512 |
| Batch size | 8 |
| Epochs | 50 |
| Optimizer | AdamW |
| Maximum learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | OneCycleLR |
| Seeds | 42, 43, 44 |
| Mixed precision | Enabled on CUDA |
| Gradient clipping | 1.0 |

Training augmentation includes horizontal flipping, small shift/scale/rotation changes, brightness/contrast changes, and Gaussian noise. Images are normalized with ImageNet mean and standard deviation because the encoder starts from ImageNet-pretrained MobileNetV3-Small weights.

## Recorded main result

The following values are the results already stored in the notebook output. They are the average over three runs on the 55-image test set.

| Metric | Result |
| --- | ---: |
| Dice | **80.74 ± 0.17 %** |
| IoU | **68.47 ± 0.23 %** |
| Precision | **79.80 ± 0.34 %** |
| Recall | **86.37 ± 0.24 %** |
| HD95 | **1.41 ± 0.21 mm** |
| ASD | **0.51 ± 0.02 mm** |
| Width MAE | **1.20 ± 0.17 mm** |
| Parameters | **1.67 M** |
| Model size | **6.4 MB** |
| GFLOPs | **2.55** |
| FPS | **179.2** |

The three saved runs used thresholds of `0.30`, `0.30`, and `0.50` respectively during the reported evaluation.

The evaluation code also saves qualitative examples, worst cases, training curves, width-analysis plots, CSV tables, and a JSON file containing the complete results.

## Ablation study

The notebook builds the model step by step so that each component can be checked separately.

| Variant | Added component | Dice (%) | IoU (%) |
| --- | --- | ---: | ---: |
| V1 | MobileNetV3 U-Net baseline | 80.58 ± 0.76 | 68.65 ± 0.76 |
| V2 | + Attention gates | 81.38 ± 0.86 | 69.44 ± 1.10 |
| V3 | + MSAM | 81.05 ± 0.26 | 68.98 ± 0.26 |
| V4 | + FiLM | 79.95 ± 0.37 | 67.67 ± 0.46 |
| V5 | + ABRB | 80.93 ± 0.43 | 68.78 ± 0.57 |
| V6 | + Edge supervision | 80.19 ± 0.65 | 68.29 ± 0.51 |
| V7 | + Deep supervision | 79.60 ± 0.63 | 67.59 ± 1.16 |

These are the numbers printed by the current notebook, not manually reconstructed values.

### Important note about the FiLM ablation

There is one implementation detail that should be fixed before using the FiLM part as evidence for gestational-age conditioning.

The active dataset class used later in the notebook returns:

```python
(image, mask, filename, pixel_size_mm)
```

but `unpack_segmentation_batch()` currently interprets the fourth item as:

```python
ga_weeks
```

That means the V4-V7 ablation runs shown above are feeding `pixel_size_mm` into the FiLM layer rather than actual gestational age.

There is a related point in the main training section: the earlier segmentation dataset returns only `(image, mask)`, so `ga_weeks` is `None` and the FiLM layer is not applied in the reported main MVSegNet training.

For a proper GA-conditioned experiment, the dataset should load a real gestational-age value for each sample and return it explicitly. Until that is changed, the FiLM numbers should not be described as a true gestational-age ablation.

## Evaluation metrics

The notebook reports more than overlap scores because boundary quality matters for this task.

- **Dice** — overlap between predicted and ground-truth masks.
- **IoU** — intersection over union.
- **Precision** and **Recall** — pixel-level segmentation measures.
- **HD95** — 95th-percentile Hausdorff distance.
- **ASD** — average symmetric surface distance.
- **Width MAE** — absolute error between predicted and ground-truth horizontal mask width.
- **GFLOPs / parameters / model size** — model complexity.
- **FPS** — measured inference throughput.

If pixel-size information is not available in the metadata, the evaluation code falls back to `0.152 mm/pixel`.

The width measurement used here is simply the horizontal extent of the binary mask. It should be treated as an image-derived geometric measurement, not as a replacement for a clinical measurement protocol.

## Requirements

The notebook uses the following main packages:

```text
torch
torchvision
opencv-python
numpy
pandas
albumentations
scikit-learn
scipy
matplotlib
tqdm
requests
```

A simple local installation is:

```bash
pip install torch torchvision opencv-python numpy pandas albumentations scikit-learn scipy matplotlib tqdm requests
```

For training, a CUDA-capable GPU is strongly recommended.

## Running the notebook

The easiest way to reproduce the workflow is to run the notebook from top to bottom in Google Colab.

### 1. Download and organize the data

Run the first cells. They create the project directories, download `Trans-ventricular.zip`, and organize images and masks.

### 2. Preprocess the data

Run the preprocessing cell to create the `512 x 512` images, binary masks, edge maps, auxiliary masks, and `metadata.json`.

The generated structure is roughly:

```text
fetal_vm_segmentation/
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── images/
│   │   └── masks/
│   └── preprocessed/
│       ├── train/
│       │   ├── images/
│       │   ├── masks/
│       │   ├── edges/
│       │   ├── aux_d2/
│       │   └── aux_d3/
│       ├── val/
│       ├── test/
│       └── metadata.json
├── models/
│   └── ablation/
└── results/
    └── ablation/
```

### 3. Train MVSegNet

Run the model, dataset, loss, and training cells, then run the three-seed experiment.

The best checkpoint for each run is saved under:

```text
fetal_vm_segmentation/models/
```

Training histories are written under:

```text
fetal_vm_segmentation/results/
```

### 4. Evaluate the saved runs

The evaluation section loads all three checkpoints and calculates the segmentation and complexity metrics.

Typical output files include:

```text
table1_comparison.csv
complete_results.json
mvsegnet_qualitative.png
mvsegnet_worst_cases.png
mvsegnet_worst_cases.csv
training_curves.png
width_analysis.png
```

### 5. Run the ablation study

The final section defines seven model configurations and trains each one with three seeds.

This is a much larger experiment: seven variants x three runs x 50 epochs. Existing checkpoints are reused when possible.

Before treating V4-V7 as GA-conditioned experiments, fix the batch field issue described above.

## A few practical notes

- The notebook contains several model definitions and some classes are redefined in later cells. For reliable reproduction, run the cells in their intended order rather than executing isolated cells from the middle.
- The main results and the later ablation results come from slightly different model/evaluation code paths, so they should be reported as separate experiments.
- The preprocessing function currently has stronger augmentation code commented out. The first preprocessing pass mainly resizes the data; the training loader applies the active augmentations.
- Checkpoints are selected using validation Dice.
- The test set should only be used for final evaluation, not for choosing model settings.
- If this code is moved from the notebook into a repository, splitting it into `dataset.py`, `models.py`, `losses.py`, `train.py`, and `evaluate.py` would make the experiment easier to reproduce.

## Dataset and citation

The data downloader points to:

**Zenodo record 8265464**  
https://zenodo.org/records/8265464

Please use the citation and license information provided on the Zenodo record when publishing results based on the dataset.

If this repository accompanies a paper or thesis, add the final paper citation here once it is available.

## License

No project license is defined inside the notebook. Add a `LICENSE` file before redistributing the code publicly, and keep the dataset under its original license terms.
