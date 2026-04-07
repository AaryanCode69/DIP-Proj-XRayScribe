# Automated Diagnostic Report Generation for Chest X-Rays

## 📌 Project Overview

[cite_start]The primary objective of this project is to bridge the gap between visual image quality and the textual accuracy of automated radiology reports[cite: 14]. [cite_start]This system acts as a hybrid framework, combining mathematical Digital Image Processing (DIP) algorithms with an attention-based Deep Learning Encoder-Decoder architecture[cite: 16].

Runtime standard: use Python 3.10.14 for development and validation. This matches the environment used for the existing Phase 1 and Phase 2 implementation and reduces dependency drift while the rest of the pipeline is built.

[cite_start]By forcing raw X-rays through a strict enhancement and segmentation pipeline before neural network ingestion, this project resolves the issue of AI models hallucinating reports due to low-dynamic-range "blindness" in raw medical scans[cite: 29, 31].

## 🏗️ Architecture & Pipeline Breakdown

This project explicitly implements four distinct phases to process the image and generate the report. To demonstrate algorithmic understanding, core DIP algorithms are written from scratch.

### Phase 1: Explicit Image Enhancement (CLAHE)

Raw DICOM/PNG images suffer from poor contrast. [cite_start]This module mathematically optimizes local tissue contrast[cite: 17].

- [cite_start]**Tile Division:** The image is divided into non-overlapping contextual regions (tiles)[cite: 104].
- [cite_start]**Histogram Clipping:** A histogram is computed for each tile and clipped at a predefined threshold to prevent noise amplification[cite: 104].
- [cite_start]**Bilinear Interpolation:** Interpolation is applied to tile borders to eliminate blocking artifacts, yielding distinct bone and lung boundaries[cite: 105].

### Phase 2: Image Segmentation (Lung Field Isolation)

To prevent the neural network from processing irrelevant background data, the enhanced image is segmented using custom mathematical operations.

- **Otsu's Thresholding:** Automatically calculates the optimal threshold to separate lung tissue from bone and empty space.
- **Morphological Operations:** Custom erosion and dilation matrices clean the binary mask, filling internal gaps and removing external noise.
- **Bitwise Masking:** The enhanced X-ray is multiplied by the mask to isolate the region of interest.

### Phase 3: Feature Extraction & Compression (PyTorch CNN)

[cite_start]The segmented image is fed into a PyTorch-based Convolutional Neural Network (e.g., ResNet/DenseNet)[cite: 112].

- [cite_start]**Feature Extraction:** The final classification layer is removed; instead, the output from the last convolutional layer is extracted to capture high-level visual patterns (opacities, nodules)[cite: 113, 114].
- **Data Compression:** Pooling layers (Max/Average) mathematically downsample the spatial dimensions of the feature maps, drastically reducing the data footprint while retaining dominant diagnostic signals.

### Phase 4: Sequence Generation (Attention-based RNN Decoder)

[cite_start]The compressed feature vectors are passed to an Attention-based LSTM Decoder[cite: 134].

- [cite_start]**Attention Mechanism:** Weighs specific regions of the extracted feature map relevant to the current word being generated[cite: 135].
- [cite_start]**Text Synthesis:** Generates clinically accurate sentences (e.g., "Heart size is normal", "No pleural effusion detected") aligned strictly with visual evidence[cite: 135, 153].

## 💾 Dataset

[cite_start]This project utilizes the **Indiana University Chest X-Ray Collection (IU X-Ray)**[cite: 149].

- [cite_start]**Source:** OpenI (Open Access Biomedical Image Search Engine) via the National Library of Medicine (NLM)[cite: 155].
- [cite_start]**Contents:** 7,470 chest X-ray images and 3,955 structured XML radiology reports[cite: 150, 152].
- [cite_start]**Target Sections:** The text generation specifically targets the "Findings" and "Impression" sections of the XML reports[cite: 153].
