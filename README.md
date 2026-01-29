
Thinking in Scales: Accelerating Gigapixel Pathology Image Analysis via  Adaptive Continuous Reasoning
This repository contains the official implementation of a multi-scale Whole-Slide Image (WSI) analysis framework based on the Continuous Thought Machine (CTM). The model mimics the diagnostic process of pathologists by performing continuous reasoning and feature aggregation across multiple spatial resolutions.

🌟 Key Features
Multi-Scale Fusion: Jointly processes features from four distinct scales (1024, 2048, 4096, and 8192) to capture both cellular details and global tissue architecture.

Continuous Reasoning (CTM): Utilizes a trainable "thought" iteration mechanism, allowing the model to dynamically refine its predictions over multiple steps.

Dynamic Inference Control: Supports a confidence threshold during inference to balance diagnostic accuracy with computational efficiency.

Baseline Comparisons: Includes implementations for LSTM and Feed-Forward (FF) architectures for rigorous benchmarking.

📂 Project Structure
📊 Data Preparation
Data is managed via .txt files where each line follows the format: [label];[path_to_feature_file]

Note: The system automatically resolves relative paths for all four scales (0_1024, 1_1024, 1_512, 0_8192) and their corresponding topological relationship files (relation) based on the provided root path in the text file.

🚀 Getting Started
1. Environment Setup
Recommended: Python 3.9+ and PyTorch 2.0+ with CUDA support.

2. Training
Run CONCH_4-scale_train.py to start training. The script supports Mixed Precision (AMP) and Model Compilation for faster throughput.

python CONCH_4-scale_train.py \
    --train_h5_dir /path/to/train.txt \
    --val_h5_dir   /path/to/val.txt \
    --log_dir logs/exp1 \
    --num_class   \
    --device 0

3. Inference & Evaluation
Use CONCH_4-scale_inference.py to evaluate the model and calculate performance metrics and efficiency statistics (average patch count and inference time).

python CONCH_4-scale_inference.py \
    --batch_size_test 1 \
    --device 0 \
    --num_class 7 \
    --threshold 0.9 \
    --fold n \
    --test_h5_dir "/path/to/test_list.txt" \
    --checkpoint_path "/path/to/checkpoint.pth" \
    --log_dir "./logs/"
