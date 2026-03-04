# AI for Sustainability - SRIP 2026
## Delhi Airshed Land-Use Classification Pipeline

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

##  Project Overview

This project implements a complete **Earth Observation pipeline** for analyzing the Delhi Airshed region using satellite imagery (Sentinel-2) and land-cover classification. The Ministry of Environment has commissioned an AI-based audit to identify **land use patterns and pollution sources** through machine learning.

The project addresses three main questions (Q1, Q2, Q3) with specific deliverables for each:

### **Q1: Spatial Reasoning & Data Filtering** (4 Marks)
- Plot Delhi-NCR shapefile with 60×60 km uniform grid
- Filter satellite images by region coordinates
- Generate statistics on data coverage

### **Q2: Label Construction & Dataset Preparation** (6 Marks)
- Extract 128×128 land-cover patches from ESA WorldCover 2021 raster
- Assign labels using dominant land-cover class (mode)
- Map ESA codes to simplified land-use categories
- Perform 60/40 train-test split with class distribution analysis

### **Q3: Model Training & Supervised Evaluation** (5 Marks)
- Train CNN model (ResNet18 or custom CNN)
- Evaluate with accuracy and F1-score
- Display and interpret confusion matrix

---

##  Project Structure

```
AI for Sustainability/
├── src/                          # Source modules
│   ├── spatial_analysis.py       # Q1: Spatial reasoning & filtering
│   ├── label_construction.py     # Q2: Label construction & preprocessing
│   ├── model_training.py         # Q3: Model training & evaluation
│   └── utils.py                  # Utility functions
├── data/
│   ├── raw/                      # Original datasets from Kaggle
│   │   ├── delhi_ncr.shp         # Delhi-NCR shapefile (EPSG:4326)
│   │   ├── delhi_airshed.shp     # Delhi-Airshed shapefile (EPSG:4326)
│   │   ├── land_cover.tif        # ESA WorldCover 2021 raster (10m resolution)
│   │   ├── satellite_images/     # Sentinel-2 RGB patches (128×128, 10m/pixel)
│   │   └── image_metadata.csv    # Image coordinates mapping
│   └── processed/                # Processed data
│       ├── filtered_images.csv   # Images within region
│       └── dataset.csv           # Final labeled dataset
├── models/                       # Trained models
│   └── landuse_classifier.pth    # Saved model weights
├── results/                      # Results and outputs
│   ├── plots/                    # Visualization outputs
│   └── metrics/                  # Evaluation metrics
├── notebooks/                    # Jupyter notebooks for exploration
├── main.py                       # Main pipeline script
├── config.json                   # Configuration parameters
├── requirements.txt              # Python dependencies
├── setup.py                      # Project setup
└── README.md                     # This file
```

---

##  Getting Started

### Prerequisites

- Python 3.7+
- pip or conda
- CUDA 11.8+ (optional, for GPU acceleration)

### Installation

1. **Clone and navigate to project:**
```bash
cd "e:\AI for sustainability"
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or with conda:
```bash
conda create -n airshed python=3.10
conda activate airshed
pip install -r requirements.txt
```

3. **Download dataset:**
```bash
python download_dataset.py
```

This will download the Earth Observation Delhi Airshed dataset from Kaggle (~289 MB).

---

##  Data Specifications

| Parameter | Value |
|-----------|-------|
| **Image Size** | 128×128 pixels |
| **Resolution** | 10 m/pixel (Sentinel-2) |
| **CRS (Input)** | EPSG:4326 (WGS84) |
| **CRS (Gridding)** | EPSG:32644 (UTM Zone 44N) |
| **Label Source** | ESA WorldCover 2021 |
| **Grid Size** | 60×60 km cells |
| **Train-Test Split** | 60% / 40% |

### ESA WorldCover 2021 Classes

| Code | Description | Simplified Class |
|------|-------------|------------------|
| 10 | Tree Cover | Vegetation |
| 20 | Shrubland | Vegetation |
| 30 | Herbaceous Vegetation | Vegetation |
| 40 | Cropland | Cropland |
| 50 | Built-up | Built-up |
| 60 | Bare / Sparse Vegetation | Bare Land |
| 70 | Snow and Ice | Others |
| 80 | Permanent Water Bodies | Water |
| 90 | Herbaceous Wetland | Water |
| 95 | Mangroves | Water |
| 100 | Moss and Lichen | Vegetation |

---

##  Running the Pipeline

### Run Full Pipeline (All Q1, Q2, Q3)

```bash
python main.py
```

### Run Specific Questions

```bash
# Q1 only: Spatial Analysis
python main.py --run-q1

# Q2 only: Label Construction
python main.py --run-q2

# Q3 only: Model Training
python main.py --run-q3
```

### Custom Configuration

Edit `config.json` to modify parameters:

```json
{
  "image_size": 128,
  "resolution": 10,
  "grid_spacing": 60,
  "train_test_split": 0.4,
  "num_epochs": 20,
  "batch_size": 32,
  "learning_rate": 0.001,
  "model_type": "custom",
  "crs_gridding": "EPSG:32644"
}
```

Or pass custom config:
```bash
python main.py --config custom_config.json
```

---

##  Model Architecture

### Option 1: Custom CNN

```
CustomCNN
├── Features
│   ├── Conv2d(3, 64) → BatchNorm2d → ReLU → MaxPool2d
│   ├── Conv2d(64, 128) → BatchNorm2d → ReLU → MaxPool2d
│   ├── Conv2d(128, 256) → BatchNorm2d → ReLU → MaxPool2d
│   ├── Conv2d(256, 256) → BatchNorm2d → ReLU → AdaptiveAvgPool2d
├── Classifier
│   ├── Linear(256, 128) → ReLU → Dropout(0.5)
│   └── Linear(128, num_classes)
```

### Option 2: ResNet18 (Pretrained)

- ResNet18 with ImageNet pretrained weights
- Modified final FC layer for land-use classification
- Transfer learning approach

---

##  Output Files

After running the pipeline, check these outputs:

```
results/
├── plots/
│   ├── q1_spatial_grid.png              # Delhi-NCR with 60×60km grid
│   ├── q2_train_distribution.png        # Training set class distribution
│   ├── q2_test_distribution.png         # Test set class distribution
│   ├── q3_training_history.png          # Loss vs epochs
│   └── q3_confusion_matrix.png          # Confusion matrix heatmap
└── metrics/
    └── evaluation_metrics.json          # Accuracy, F1-score, classification report
```

---

##  Expected Performance

Based on Earth Observation best practices:

- **Accuracy**: 70-85% (depends on class separability)
- **F1-Score (Macro)**: 65-80%
- **F1-Score (Weighted)**: 75-88%

Best performing classes:
-  Built-up (urban areas, distinct spectral signature)
-  Water bodies (high contrast)

Most challenging classes:
-  Herbaceous vegetation vs. Cropland (spectral similarity)
-  Different vegetation types (temporal variations)

---

##  Advanced Usage

### Using GPU

The pipeline automatically detects and uses CUDA if available:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

To force CPU:
```bash
CUDA_VISIBLE_DEVICES="" python main.py
```

### Jupyter Notebooks

For interactive exploration:

```bash
jupyter notebook notebooks/
```

Create a new notebook and import modules:

```python
import sys
sys.path.insert(0, './src')
from spatial_analysis import *
from label_construction import *
from model_training import *
```

### Training with Different Model

```python
# In config.json or command line:
"model_type": "resnet18"  # Instead of "custom"
```

---

##  Key Functions

### Spatial Analysis (`src/spatial_analysis.py`)
- `plot_delhi_ncr_with_grid()` - Visualize spatial grid overlay
- `filter_images_by_region()` - Spatial filtering by coordinates
- `create_spatial_grid_epsg32644()` - Generate UTM grid

### Label Construction (`src/label_construction.py`)
- `extract_landcover_patch()` - Extract 128×128 label patches
- `assign_image_label()` - Assign dominant class label
- `build_training_dataset()` - Create full labeled dataset
- `train_test_split_dataset()` - Stratified train-test split
- `visualize_class_distribution()` - Plot class statistics

### Model Training (`src/model_training.py`)
- `LandUseDataset` - PyTorch Dataset class
- `create_model()` - Initialize CNN or ResNet
- `train_model()` - Full training pipeline
- `compute_metrics()` - Calculate accuracy & F1-score
- `plot_confusion_matrix()` - Confusion matrix visualization

---

##  Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce `batch_size` in config.json

### Issue: Image file not found
**Solution:** Ensure satellite images are in `data/raw/satellite_images/` directory

### Issue: Shapefile read error
**Solution:** Verify shapefile has all components (.shp, .shx, .dbf, .prj)

### Issue: Rasterio error reading land_cover.tif
**Solution:** Ensure GeoTIFF file is properly formatted with spatial reference

---

##  References

1. **ESA WorldCover 2021**: https://www.esa-worldcover.org/
2. **Sentinel-2 Documentation**: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi
3. **GeoPandas**: https://geopandas.org/
4. **PyTorch**: https://pytorch.org/
5. **Rasterio**: https://rasterio.readthedocs.io/

---

##  Project Highlights

-  Complete Earth Observation pipeline
-  Spatial analysis with shapefile operations
-  Raster processing and label extraction
-  CNN training with PyTorch
-  Comprehensive evaluation metrics
-  Reproducible with seeding
-  Configuration-driven parameters
-  Detailed logging and visualization

---