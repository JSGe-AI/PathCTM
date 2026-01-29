
# **Thinking in Scales: Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning**

This repository implements an efficient scale-space continuous inference framework for gigapixel whole slide images (WSI). The model mimics the diagnostic process of pathologists by performing continuous reasoning and feature aggregation across multiple spatial resolutions.

🌟 **Key Features**

Scale-Space Continuous Reasoning: PathCTM performs cross-scale continuous reasoning in the spatial dimension, establishing a coherent coarse-to-fine scale-wise inference trajectory. 

Attention-Guided Region Pruning: High-resolution features are selectively loaded for informative regions, thereby eliminating the inherent speed bottlenecks of WSI analysis.

Confidence-Aware Early Stopping: Considering the varying difficulty of diagnostic cases, PathCTM incorporates a confidence-aware early stopping strategy based on entropy minimization.


📂 **Project Structure**

📊 **Data Preparation**

1. Feature Extraction
   Use foundation models (UNI v2 or CONCH v1.5) as the feature extractor. 

2. File OrganizationData is managed via .txt files where each line follows the format:[label];[path_to_feature_file]

Note: The system automatically resolves relative paths for all four scales (0_1024, 1_1024, 1_512, 0_8192) and their corresponding topological relationship files (relation) based on the root path provided in your text file.

🚀 **Getting Started**

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
        --num_class n \
        --threshold 0.9 \
        --fold n \
        --test_h5_dir "/path/to/test_list.txt" \
        --checkpoint_path "/path/to/checkpoint.pth" \
        --log_dir "./logs/"
