"""
Main pipeline for Delhi Airshed Land-Use Classification
SRIP 2026 - AI for Sustainability
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots

from src.spatial_analysis import (
    plot_delhi_ncr_with_grid,
    filter_images_by_region,
    create_spatial_grid_epsg32644
)
from src.label_construction import (
    build_training_dataset,
    train_test_split_dataset,
    visualize_class_distribution,
    visualize_sample_images_with_labels
)
from src.model_training import (
    LandUseDataset,
    get_data_transforms,
    train_model,
    compute_metrics,
    plot_confusion_matrix,
    plot_training_history
)
from src.utils import setup_project_structure, save_metrics, load_config, print_summary


# ---------------------------------------------------------------------------
# Dataset locations — tries Kaggle cache first, then local archive/ folder
# ---------------------------------------------------------------------------
KAGGLE_DATASET_DIR = (
    r"C:\Users\hp\.cache\kagglehub\datasets"
    r"\rishabhsnip\earth-observation-delhi-airshed\versions\1"
)
ARCHIVE_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'archive')


def resolve_dataset_dir():
    """Return the first dataset directory that actually exists on disk."""
    for candidate in (KAGGLE_DATASET_DIR, ARCHIVE_DATASET_DIR):
        # Check for at least one expected file inside the candidate dir
        if os.path.isdir(candidate) and os.path.exists(
            os.path.join(candidate, 'delhi_ncr_region.geojson')
        ):
            return candidate
    # Last resort — return archive path so error messages are meaningful
    return ARCHIVE_DATASET_DIR


def build_image_metadata(image_directory):
    """
    Parse image filenames in the rgb/ folder to build a metadata DataFrame.
    Each filename is ``{latitude}_{longitude}.png``.

    Returns:
        pd.DataFrame with columns: image_name, latitude, longitude
    """
    records = []
    for fpath in glob.glob(os.path.join(image_directory, '*.png')):
        fname = os.path.basename(fpath)
        name_no_ext = os.path.splitext(fname)[0]  # e.g. "28.2056_76.8558"
        parts = name_no_ext.split('_')
        if len(parts) == 2:
            try:
                lat, lon = float(parts[0]), float(parts[1])
                records.append({
                    'image_name': name_no_ext,
                    'latitude': lat,
                    'longitude': lon,
                })
            except ValueError:
                continue
    df = pd.DataFrame(records)
    print(f"Parsed metadata for {len(df)} images from {image_directory}")
    return df


# ===================================================================
# Q1  –  Spatial Reasoning & Data Filtering
# ===================================================================
def run_q1_spatial_analysis(config, paths):
    print("\n" + "="*70)
    print("Q1: SPATIAL REASONING & DATA FILTERING")
    print("="*70)

    # Q1.1 – Plot Delhi-NCR with 60×60 km grid
    print("\n[Q1.1] Plotting Delhi-NCR region with 60x60 km grid...")
    if os.path.exists(paths['delhi_ncr_shapefile']):
        grid_gdf = plot_delhi_ncr_with_grid(
            paths['delhi_ncr_shapefile'],
            output_path=paths['q1_grid_plot']
        )
        print(f"  Grid plot saved to {paths['q1_grid_plot']}")
    else:
        print(f"  Warning: {paths['delhi_ncr_shapefile']} not found – skipping grid plot")

    # Build image metadata from filenames
    print("\n[Q1.1b] Building image metadata from filenames...")
    image_metadata = build_image_metadata(paths['image_directory'])

    # Q1.2 – Filter images whose centres fall inside Delhi-Airshed
    print("\n[Q1.2] Filtering satellite images by Delhi-Airshed region...")
    if len(image_metadata) > 0 and os.path.exists(paths['delhi_airshed_shapefile']):
        filtered_images = filter_images_by_region(
            image_metadata,                     # DataFrame, not a CSV path
            paths['delhi_airshed_shapefile'],
            output_path=paths['filtered_metadata']
        )
        print(f"  Filtered metadata saved to {paths['filtered_metadata']}")
    else:
        print("  Warning: no metadata or airshed file – returning all images")
        filtered_images = image_metadata

    return filtered_images


# ===================================================================
# Q2  –  Label Construction & Dataset Preparation
# ===================================================================
def run_q2_label_construction(config, paths, filtered_images=None):
    print("\n" + "="*70)
    print("Q2: LABEL CONSTRUCTION & DATASET PREPARATION")
    print("="*70)

    # Q2.1 – Build dataset (extract land-cover patches & assign labels)
    print("\n[Q2.1] Building training dataset from land-cover raster...")
    if filtered_images is not None and len(filtered_images) > 0:
        dataset_df = build_training_dataset(
            paths['image_directory'],
            filtered_images,
            paths['landcover_tif'],
            output_csv=paths['dataset_csv']
        )
    else:
        print("  ERROR: No filtered images available – cannot build dataset.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if len(dataset_df) == 0:
        print("  ERROR: Dataset is empty after label construction.")
        return dataset_df, pd.DataFrame(), pd.DataFrame()

    print(f"  Total labelled samples: {len(dataset_df)}")

    # Q2.2 / Q2.3 – Class mapping summary
    print("\n[Q2.2-Q2.3] ESA WorldCover → simplified categories:")
    print("  Tree Cover / Shrubland / Herbaceous / Moss → Vegetation")
    print("  Cropland → Cropland")
    print("  Built-up → Built-up")
    print("  Permanent Water / Wetland / Mangroves → Water")
    print("  Bare / Sparse → Bare Land")
    print("  Snow & Ice → Others")

    # Q2.4 – 60 / 40 train-test split (stratified)
    print("\n[Q2.4] Performing 60/40 stratified train-test split...")
    train_df, test_df = train_test_split_dataset(
        dataset_df,
        test_size=config['train_test_split'],
        random_state=42
    )

    # Q2.5 – Visualise class distributions
    print("\n[Q2.5] Visualising class distributions...")
    visualize_class_distribution(train_df, set_name='Training Set',
                                save_path=paths['train_distribution_plot'])
    visualize_class_distribution(test_df, set_name='Test Set',
                                save_path=paths['test_distribution_plot'])

    return dataset_df, train_df, test_df


# ===================================================================
# Q3  –  Model Training & Supervised Evaluation
# ===================================================================
def run_q3_model_training(config, paths, train_df, test_df):
    print("\n" + "="*70)
    print("Q3: MODEL TRAINING & SUPERVISED EVALUATION")
    print("="*70)

    if len(train_df) == 0:
        print("  Warning: Empty training set – skipping model training.")
        # Try loading from saved dataset CSV
        if os.path.exists(paths['dataset_csv']):
            print("  Loading dataset from saved CSV...")
            from src.label_construction import train_test_split_dataset as tts
            dataset_df = pd.read_csv(paths['dataset_csv'])
            train_df, test_df = tts(dataset_df, test_size=config['train_test_split'], random_state=42)
        if len(train_df) == 0:
            return None, None, None

    # Q3.1 – Data loaders
    print("\n[Q3.1] Preparing data loaders...")
    train_transform, test_transform = get_data_transforms()

    # Build a consistent label→index mapping from *all* labels in both splits
    all_labels = sorted(set(train_df['label'].unique()) | set(test_df['label'].unique()))
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

    train_dataset = LandUseDataset(train_df, label_to_idx, transform=train_transform)
    test_dataset  = LandUseDataset(test_df,  label_to_idx, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=config['batch_size'],
                              shuffle=False, num_workers=0)

    num_classes = len(label_to_idx)
    class_names = list(label_to_idx.keys())

    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Training batches: {len(train_loader)}  |  Test batches: {len(test_loader)}")

    # Q3.2 – Train CNN
    print("\n[Q3.2] Training CNN model...")
    print(f"  Model: {config['model_type']}  |  Epochs: {config['num_epochs']}  |  LR: {config['learning_rate']}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, test_preds, test_labels, history = train_model(
        train_loader, test_loader, num_classes,
        num_epochs=config['num_epochs'],
        model_type=config['model_type'],
        device=device,
        learning_rate=config['learning_rate']
    )

    # Save model
    os.makedirs(paths['models_dir'], exist_ok=True)
    torch.save(model.state_dict(), paths['model_save_path'])
    print(f"\n  Model saved to {paths['model_save_path']}")

    # Q3.3 – Training history
    print("\n[Q3.3] Plotting training history...")
    plot_training_history(history, save_path=paths['training_history_plot'])

    # Q3.4 / Q3.5 – Metrics
    print("\n[Q3.4-Q3.5] Evaluating with Accuracy & F1-Score...")
    metrics, report = compute_metrics(test_labels, test_preds, class_names)

    print(f"\n  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  F1-Score (Macro):  {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")

    save_metrics(metrics, report, paths['metrics_json'])

    # Q3.6 – Confusion matrix
    print("\n[Q3.6] Displaying Confusion Matrix...")
    cm = plot_confusion_matrix(test_labels, test_preds, class_names,
                               save_path=paths['confusion_matrix_plot'])

    # Interpretation
    print("\n" + "="*70)
    print("RESULTS INTERPRETATION")
    print("="*70)
    print(f"\n  Model achieved {metrics['accuracy']*100:.2f}% accuracy on the test set.")
    print(f"  Weighted F1-Score: {metrics['f1_weighted']:.4f}")

    return model, metrics, cm


# ===================================================================
# Entry point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description='Delhi Airshed Land-Use Classification')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file')
    parser.add_argument('--run-q1', action='store_true', help='Run Q1 only')
    parser.add_argument('--run-q2', action='store_true', help='Run Q2 only')
    parser.add_argument('--run-q3', action='store_true', help='Run Q3 only')
    args = parser.parse_args()

    # Setup
    root_dir = os.path.dirname(os.path.abspath(__file__))
    setup_project_structure(root_dir)

    # Load config
    config = load_config(args.config)

    # ---- Paths (auto-resolve Kaggle cache → local archive) ----
    dataset_dir = resolve_dataset_dir()
    print(f"\n  Using dataset directory: {dataset_dir}")
    paths = {
        # Source data (Kaggle download)
        'delhi_ncr_shapefile':   os.path.join(dataset_dir, 'delhi_ncr_region.geojson'),
        'delhi_airshed_shapefile': os.path.join(dataset_dir, 'delhi_airshed.geojson'),
        'landcover_tif':         os.path.join(dataset_dir, 'worldcover_bbox_delhi_ncr_2021.tif'),
        'image_directory':       os.path.join(dataset_dir, 'rgb'),
        # Processed / output
        'filtered_metadata':       os.path.join(root_dir, 'data/processed/filtered_images.csv'),
        'dataset_csv':             os.path.join(root_dir, 'data/processed/dataset.csv'),
        'q1_grid_plot':            os.path.join(root_dir, 'results/plots/q1_spatial_grid.png'),
        'train_distribution_plot': os.path.join(root_dir, 'results/plots/q2_train_distribution.png'),
        'test_distribution_plot':  os.path.join(root_dir, 'results/plots/q2_test_distribution.png'),
        'training_history_plot':   os.path.join(root_dir, 'results/plots/q3_training_history.png'),
        'confusion_matrix_plot':   os.path.join(root_dir, 'results/plots/q3_confusion_matrix.png'),
        'metrics_json':            os.path.join(root_dir, 'results/metrics/evaluation_metrics.json'),
        'models_dir':              os.path.join(root_dir, 'models'),
        'model_save_path':         os.path.join(root_dir, 'models/landuse_classifier.pth'),
    }

    # Run pipeline
    run_all = not (args.run_q1 or args.run_q2 or args.run_q3)

    print("\n" + "#"*70)
    print("# AI FOR SUSTAINABILITY - SRIP 2026")
    print("# Delhi Airshed Land-Use Classification Pipeline")
    print("#"*70)

    filtered_images = None
    train_df = pd.DataFrame()
    test_df  = pd.DataFrame()

    if run_all or args.run_q1:
        filtered_images = run_q1_spatial_analysis(config, paths)

    if run_all or args.run_q2:
        dataset_df, train_df, test_df = run_q2_label_construction(
            config, paths, filtered_images
        )

    if run_all or args.run_q3:
        model, metrics, cm = run_q3_model_training(config, paths, train_df, test_df)

    # Summary
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETED")
    print("="*70)
    print(f"\n  Results  → {os.path.join(root_dir, 'results/')}")
    print(f"  Models   → {paths['models_dir']}")
    print(f"  Data     → {os.path.join(root_dir, 'data/processed/')}")


if __name__ == '__main__':
    main()
