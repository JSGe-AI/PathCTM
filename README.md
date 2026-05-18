
# **Thinking in Scales: Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning**

This repository implements an efficient scale-space continuous inference framework for gigapixel whole slide images (WSI). The model mimics the diagnostic process of pathologists by performing continuous reasoning and feature aggregation across multiple spatial resolutions. (The code repository is currently being updated and will be completed soon.)

🌟 **Key Features**

Scale-Space Continuous Reasoning: PathCTM performs cross-scale continuous reasoning in the spatial dimension, establishing a coherent coarse-to-fine scale-wise inference trajectory. 

Attention-Guided Region Pruning: High-resolution features are selectively loaded for informative regions, thereby eliminating the inherent speed bottlenecks of WSI analysis.

Confidence-Aware Early Stopping: Considering the varying difficulty of diagnostic cases, PathCTM incorporates a confidence-aware early stopping strategy based on entropy minimization.


📂 **Project Structure**

📊 **Data Preparation**

1. Feature Extraction
   Use foundation models (UNI v2 or CONCH v1.5) as the feature extractor.
   ```python
   from PIL import Image
   image = Image.open("path_to_image.jpg")
   image = preprocess(image).unsqueeze(0)
   with torch.inference_mode():
       image_embs = model.encode_image(image, proj_contrast=False, normalize=False)
   ```
3. File OrganizationData is managed via .txt files where each line follows the format:[label];[path_to_feature_file]

Note: The system automatically resolves relative paths for all four scales (0_1024, 1_1024, 1_512, 0_8192) and their corresponding topological relationship files (relation) based on the root path provided in your text file.


