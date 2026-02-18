# Project Proposal: Minecraft Build Analysis

## Description

Build a reproducible supervised image‑classification pipeline that predicts the style tags of a Minecraft build from a single image.

---

## Goal

Successfully predicts the style tags (Castle, Church, House, Statue etc.) of Minecraft builds based on given images.

---

## Data Collection

### What to collect

Image (screenshot or thumbnail) with labeled build style tags.

### How to collect

Scrape images and metadata from GrabCraft website (https://www.grabcraft.com). The website contains thousands of Minecraft build blueprints.

Each build in GrabCraft is labeled with tags (e.g., Tags: age of empires, castle, medieval, medieval castle), which will be scripped and used as labels. Also, since some builds have more than one images, all images will be downloaded and saved in a separate folder.

---

## Modeling

### Labels

The usage of each tag appeared in the dataset will be calculated. After discarding meaningless tags, the most used ones will be selected as class labels.

### Images

For each build, all images will be sent to the model to improve accuracy.

### Model

Transfer learning with a pretrained CNN backbone such as **ResNet50** or **EfficientNet B0**, fine‑tuned on the GrabCraft images using cross‑entropy loss.

---

## Visualization

- **Class balance and EDA:** bar charts of class counts, histograms of image sizes, and sample image grids per class.  
- **Feature visualization:** t‑SNE or UMAP projection of learned image embeddings colored by class to inspect cluster structure.
- **Interpretability:** Grad‑CAM overlays on sample images for each class showing salient regions.

---

## Test

### Baseline

Stratified split into **70% train, 15% validation, 15% test** by ID to avoid near‑duplicate leakage across splits. Keep the test set untouched until final evaluation. Use validation set for hyperparameter tuning and early stopping.

### Robustness

Evaluate model on images with different resolutions and on augmented variants to measure sensitivity.

---

## Timeline

### Weeks 1–2: Data and setup

Finalize class mapping and write the polite scraper. Download sample dataset and a larger initial crawl for exploration.

### Weeks 3–4: Preprocessing and baseline

Build preprocessing pipeline, create stratified splits, and train baseline transfer‑learning model.

### Weeks 5–6: Model improvements

Adjust various aspects of the training pipeline, including data augmentation, regularization, and optimization strategies, to improve accuracy.

### Weeks 7–8: Final evaluation and writeup

Finalize test evaluation, produce visualizations (confusion matrix, t‑SNE, Grad‑CAM), prepare README, reproducibility instructions, and final report. Package sample data and saved model checkpoint.