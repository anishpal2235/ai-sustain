"""
Utility functions for project
"""
import os
import json
import numpy as np
from pathlib import Path


def setup_project_structure(root_dir):
    """Create necessary directories if they don't exist."""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'results/plots',
        'results/metrics',
        'notebooks',
        'src'
    ]
    
    for dir_path in directories:
        full_path = os.path.join(root_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Ensured directory: {full_path}")


def save_metrics(metrics, report, output_path):
    """Save evaluation metrics to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    metrics_serializable = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
        for k, v in metrics.items()
    }
    
    output = {
        'metrics': metrics_serializable,
        'classification_report': report
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Metrics saved to {output_path}")


def load_config(config_path='config.json'):
    """Load configuration from JSON file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Config file {config_path} not found. Using defaults.")
        return get_default_config()


def get_default_config():
    """Get default configuration."""
    return {
        'image_size': 128,
        'resolution': 10,  # meters per pixel
        'grid_spacing': 60,  # kilometers
        'train_test_split': 0.4,  # test size
        'num_epochs': 20,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model_type': 'custom',  # or 'resnet18'
        'crs_gridding': 'EPSG:32644'
    }


def print_summary(config, dataset_stats):
    """Print project summary."""
    print("\n" + "="*70)
    print("PROJECT SUMMARY - AI for Sustainability: Delhi Airshed Analysis")
    print("="*70)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nDataset Statistics:")
    print(f"  Total images: {dataset_stats.get('total_images', 'N/A')}")
    print(f"  Training images: {dataset_stats.get('train_images', 'N/A')}")
    print(f"  Test images: {dataset_stats.get('test_images', 'N/A')}")
    print(f"  Number of classes: {dataset_stats.get('num_classes', 'N/A')}")
    print(f"  Classes: {dataset_stats.get('classes', 'N/A')}")
    print("="*70 + "\n")
