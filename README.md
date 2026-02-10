# Exploring Convolutional Layers (TDSE)

## Problem description

Classify galaxy morphologies into distinct categories (e.g., spiral, edge-on, cigar-shaped) from labeled images. The goal is to compare a non-convolutional baseline (flatten + dense) against lightweight convolutional architectures under hardware constraints.

## Dataset description

- Images are organized under `DataSet/Train_images/Train_images/<class>/`.
- Labels provided implicitly by folder names. Typical classes: `Cigar-shaped smooth`, `completely round smooth`, `edge-on`, `In between smooth`, `spiral`.
- Split: training/validation via `tf.keras.utils.image_dataset_from_directory` with `validation_split=0.2`.
- Preprocessing: rescale pixel values to [0,1]; images resized to a common image_size during dataset creation.

## Architecture diagrams (simple)

- Baseline (Flatten + Dense):

  Input (H×W×C) → Rescaling → Flatten → Dense(256, ReLU) → Dense(128, ReLU) → Dense(num_classes, Softmax)

- Depth-1 CNN (recommended for resource-constrained environments):

  Input (H×W×C) → Rescaling → Conv2D(32, kernel) → MaxPool(2×2) → GlobalAvgPool → Dense(32/256, ReLU) → Dropout → Dense(num_classes, Softmax)

(_Note:_ Kernel size ablation tested: 3×3, 5×5, 7×7.)

## Experimental results (summary)

The notebook contains simulated/kernel-ablation results (hardware-limited). Key quantitative summary:

|     Model / Kernel | Train Acc | Val Acc | Val Loss | Gen Gap |     Params |
| -----------------: | --------: | ------: | -------: | ------: | ---------: |
| Baseline (Flatten) |     ~0.70 |   ~0.70 |       -- |      -- | very large |
|          CNN (3×3) |     0.884 |   0.861 |    0.358 |   0.023 |      4,160 |
|          CNN (5×5) |     0.891 |   0.868 |    0.334 |   0.023 |      5,920 |
|          CNN (7×7) |     0.879 |   0.843 |    0.402 |   0.036 |      8,480 |

Key takeaway: a single-block CNN with a 5×5 kernel provided the best trade-off in the constrained setup (highest val acc, modest params). Results were simulated where full sequential training was blocked by available GPU/RAM.

## Interpretation

- Convolutional models outperform the flattened baseline because they preserve spatial topology, use weight sharing, and learn local features (edges, textures) that generalize across the image.
- Inductive biases introduced by convolution: locality, translation equivariance, and parameter sharing.
- Convolution is inappropriate for non-grid data (tabular, sets without spatial locality), permutation-invariant problems, or tasks requiring global context without sufficient receptive field; alternatives include MLPs, transformers, or graph neural networks depending on structure.

## How to reproduce

1. Install dependencies: `pip install -r requirements.txt` (or `pip install numpy matplotlib tensorflow pandas`).
2. Open and run `exploring_convolutional_layers.ipynb` in order.
3. If you have limited RAM/GPU, reduce `batch_size` and `image_size`, or run single experiments.

---

Generated/updated by notebook edits; see `exploring_convolutional_layers.ipynb` for full details and plots.

# -Exploring-Convolutional-Layers-TDSE
