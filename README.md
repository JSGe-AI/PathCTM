# Thinking in Scales: Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning

This repository contains the PathCTM implementation for gigapixel whole-slide image (WSI) analysis. PathCTM performs coarse-to-fine continuous reasoning across multiple magnification levels and supports adaptive inference with attention-guided region pruning and confidence-aware early stopping.


## Environment Installation

Recommended setup:

```bash
cd PathCTM-main

conda create -n pathctm python=3.12 -y
conda activate pathctm
pip install -r requirements.txt
```

Notes:

- The exported environment was tested with Python `3.12.11`.
- `requirements.txt` includes the installed PyTorch and CUDA-related packages from the `pathctm` environment.
- If your local CUDA driver or GPU setup is different, you may need to adjust `torch`, `torchvision`, and the `nvidia-*` packages accordingly.

## Feature Extraction

Patch features should be extracted offline with a pathology foundation model before running PathCTM. We recommend using the CLAM framework for WSI patching and feature extraction:

- CLAM GitHub: https://github.com/mahmoodlab/CLAM

This design avoids repeatedly running multi-scale feature extraction inside the PathCTM training loop.


## Data Format

PathCTM expects three kinds of files:

- A slide list file in `.txt` format.
- Multi-scale feature files in `.npy` format.
- Cross-scale relation files in `.npy` format.

### 1. Slide List File

Each line in the training or validation list should follow:

```text
label;/absolute/path/to/0_1024_3cls/<class_id>/<slide_id>.npy
```

Example:

```text
0;path/0_1024_3cls/0/BRACS_1379.npy
```

The path stored in the `.txt` file points to the 1024-scale feature file of a slide. The scripts then automatically resolve the corresponding 2048, 4096, and 8192 scale feature files, as well as the relation files.

Absolute paths are recommended.

### 2. Expected Directory Layout

A typical feature directory looks like this:

```text
data/
├── 0_1024_3cls/
│   ├── 0/
│   ├── 1/
│   └── 2/
├── 0_2048/
├── 0_4096/
├── 0_8192/
├── relation_2048-1024_index/
├── relation_4096-2048_index/
├── relation_8192-4096_index/
└── 3cls_fold/
    ├── fold_1_train.txt
    └── fold_1_val.txt
    └── fold_1_test.txt

```


### 3. Multi-scale Feature File Format

Each feature file is a Python dictionary saved as `.npy`:

```python
{
    "feature": np.ndarray,   # shape: [N, C]
    "index": list[str],      # length N
}
```

Example `index` entries:

```text
72864_58354_1024.png
72864_58354_2048.png
72864_56306_4096.png
72864_52210_8192.png
```

Notes:

- `feature` stores the patch embeddings extracted by the pathology foundation model.
- `index` stores the patch coordinates together with the corresponding scale suffix.
- The code uses the first two fields in each index string as the 2D patch coordinate, for example `72864` and `58354` in `72864_58354_1024.png`.

### 4. Cross-scale Relation File Format

Each relation file is also a Python dictionary saved as `.npy`:

```python
{
    "72864_52210_8192.png": [0, 1, 5, 8],
    "2176_51057_8192.png": [3, 4, 7],
}
```


In this relation dictionary, the key "72864_52210_8192.png" represents a patch at the coarse scale, while the value [0, 1, 5, 8] denotes the indices of its corresponding child patches at the next finer scale. Therefore, indices 0, 1, 5, and 8 refer to the child patches in the next-scale feature array that spatially belong to or are associated with the coarse patch "72864_52210_8192.png".

The exact suffix used in the relation key may vary by preprocessing pipeline, for example `8192`, `1024`, or `512`. 

## How Coordinates Are Used

Coordinates are central to PathCTM.

1. Each patch embedding carries its spatial coordinate through the `index` field in the feature file.
2. During multi-scale reasoning, the model first attends to coarse-scale patches.
3. The attended coarse coordinates are then matched against the relation files to retrieve the indices of candidate child patches at the next finer scale.

In short, coordinates are not just metadata. They are the link that makes cross-scale routing and patch pruning possible.

## Training

Run training from the repository root:

```bash
cd PathCTM-main
conda activate pathctm

python train/CONCH_4-scale_train.py \
  --device 0 \
  --fold 1 \
  --num_class 3 \
  --train_h5_dir /path/to/fold_1_train.txt \
  --val_h5_dir /path/to/fold_1_val.txt \
  --log_dir /path/to/train_logs
```


## Testing

Run evaluation from the repository root:



```bash
python train/CONCH_4-scale_test.py \
  --device 0 \
  --fold 1 \
  --num_class 3 \
  --threshold 1.0 \
  --test_h5_dir /path/to/fold_1_val.txt \
  --checkpoint_path /path/to/best_AUC_checkpoint.pt \
  --log_dir /path/to/test_logs
```

Notes:

- `--threshold` controls confidence-aware early stopping during inference.
- The speedup of PathCTM is mainly observed in this evaluation stage, where adaptive scale switching and early stopping reduce the number of accessed high-resolution patches.

## Acknowledgement

This repository is developed based on the original CTM codebase. We sincerely thank the CTM authors for releasing their implementation and making this work possible.

- CTM codebase: https://github.com/SakanaAI/continuous-thought-machines/

## Citation

If you find this repository useful, please cite the PathCTM paper:

```bibtex
@article{ge2026thinking,
  title={Thinking in Scales: Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning},
  author={Ge, Jiusong and Zhan, Yingkang and Zhao, Wenjie and Zhang, Di and Wang, Ke and Liu, Jiashuai and Yang, Chunze and Li, Chengzu and Zhang, Jian and Dong, Yuxin and others},
  journal={arXiv preprint arXiv:2605.19491},
  year={2026}
}
```
