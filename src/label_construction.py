"""
Q2: Label Construction & Dataset Preparation
"""
import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box, Point
from pyproj import Transformer
import pandas as pd
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Re-usable CRS transformers (EPSG:4326 <-> EPSG:32644)
_to_utm   = Transformer.from_crs('EPSG:4326', 'EPSG:32644', always_xy=True)
_to_wgs84 = Transformer.from_crs('EPSG:32644', 'EPSG:4326', always_xy=True)


# ESA WorldCover 2021 class mapping
ESA_CLASS_MAPPING = {
    10: 'Tree Cover',
    20: 'Shrubland',
    30: 'Herbaceous Vegetation',
    40: 'Cropland',
    50: 'Built-up',
    60: 'Bare / Sparse Vegetation',
    70: 'Snow and Ice',
    80: 'Permanent Water Bodies',
    90: 'Herbaceous Wetland',
    95: 'Mangroves',
    100: 'Moss and Lichen'
}

# Simplified land-use categories
SIMPLIFIED_CLASSES = {
    10: 'Vegetation',    # Tree Cover
    20: 'Vegetation',    # Shrubland
    30: 'Vegetation',    # Herbaceous Vegetation
    40: 'Cropland',
    50: 'Built-up',
    60: 'Bare Land',
    70: 'Others',        # Snow and Ice
    80: 'Water',
    90: 'Water',         # Herbaceous Wetland
    95: 'Water',         # Mangroves
    100: 'Vegetation'    # Moss and Lichen
}

def extract_landcover_patch(landcover_tif, latitude, longitude,
                           patch_size=128, resolution=10):
    """
    Extract a patch_size × patch_size land-cover window from land_cover.tif
    centred on (latitude, longitude).

    Steps:
      1. Convert centre to EPSG:32644 (metres).
      2. Compute ±half-extent in metres.
      3. Convert corners back to the raster’s CRS.
      4. Read the window and resize to exactly patch_size × patch_size.

    Args:
        landcover_tif: Path to land_cover.tif raster
        latitude:  Centre latitude  (EPSG:4326)
        longitude: Centre longitude (EPSG:4326)
        patch_size: Size of patch in pixels (128)
        resolution: Pixel resolution in metres (10)

    Returns:
        Numpy array (uint8) of land-cover patch, or None on failure.
    """
    half_m = (patch_size * resolution) / 2.0  # 640 m for default

    # Centre → UTM metres
    cx_utm, cy_utm = _to_utm.transform(longitude, latitude)
    x_min_utm, y_min_utm = cx_utm - half_m, cy_utm - half_m
    x_max_utm, y_max_utm = cx_utm + half_m, cy_utm + half_m

    # Corners back to WGS-84
    x_min_4326, y_min_4326 = _to_wgs84.transform(x_min_utm, y_min_utm)
    x_max_4326, y_max_4326 = _to_wgs84.transform(x_max_utm, y_max_utm)

    try:
        with rasterio.open(landcover_tif) as src:
            # If the raster is not in EPSG:4326, transform to its native CRS
            if src.crs and str(src.crs) != 'EPSG:4326':
                t = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
                x_min_r, y_min_r = t.transform(x_min_4326, y_min_4326)
                x_max_r, y_max_r = t.transform(x_max_4326, y_max_4326)
            else:
                x_min_r, y_min_r = x_min_4326, y_min_4326
                x_max_r, y_max_r = x_max_4326, y_max_4326

            window = from_bounds(x_min_r, y_min_r, x_max_r, y_max_r,
                                 src.transform)
            patch = src.read(1, window=window)

            if patch.shape != (patch_size, patch_size):
                from scipy.ndimage import zoom
                patch = zoom(patch,
                             (patch_size / patch.shape[0],
                              patch_size / patch.shape[1]),
                             order=0)

            return patch.astype(np.uint8)
    except Exception as e:
        print(f"Error extracting patch for ({latitude}, {longitude}): {e}")
        return None


def assign_image_label(landcover_patch, use_simplified=True):
    """
    Assign image label using dominant (mode) land-cover class.
    
    Args:
        landcover_patch: 128×128 numpy array of land-cover classes
        use_simplified: Use simplified categories or ESA classes
        
    Returns:
        Tuple of (label_code, label_name, class_distribution)
    """
    # Get pixel values
    pixel_values = landcover_patch.flatten()
    pixel_values = pixel_values[pixel_values > 0]  # Remove nodata
    
    if len(pixel_values) == 0:
        return None, 'Unknown', {}
    
    # Find mode (dominant class)
    counter = Counter(pixel_values)
    dominant_class = counter.most_common(1)[0][0]
    
    # Get class name
    if use_simplified:
        label_name = SIMPLIFIED_CLASSES.get(dominant_class, 'Others')
    else:
        label_name = ESA_CLASS_MAPPING.get(dominant_class, 'Unknown')
    
    class_dist = dict(counter)
    
    return dominant_class, label_name, class_dist


def build_training_dataset(image_directory, image_metadata, landcover_tif, 
                          output_csv=None):
    """
    Build dataset with images, land-cover patches, and labels.
    
    Args:
        image_directory: Directory containing satellite images (``rgb/``)
        image_metadata: DataFrame with 'image_name', 'latitude', 'longitude'
        landcover_tif: Path to WorldCover .tif raster
        output_csv: Path to save dataset metadata
        
    Returns:
        DataFrame with image paths, labels, and metadata
    """
    dataset = []
    skipped = 0
    
    total = len(image_metadata)
    print(f"  Processing {total} images ...")

    for idx, row in image_metadata.iterrows():
        image_name = row['image_name']
        latitude   = row['latitude']
        longitude  = row['longitude']
        
        image_path = os.path.join(image_directory, f"{image_name}.png")
        
        if not os.path.exists(image_path):
            skipped += 1
            continue
        
        # Extract land-cover patch
        patch = extract_landcover_patch(landcover_tif, latitude, longitude)
        
        if patch is None:
            skipped += 1
            continue
        
        # Assign label
        class_code, class_name, dist = assign_image_label(patch, use_simplified=True)
        
        if class_code is None:
            skipped += 1
            continue
        
        dataset.append({
            'image_name': image_name,
            'image_path': image_path,
            'latitude': latitude,
            'longitude': longitude,
            'label_code': int(class_code),
            'label': class_name,
            'num_pixels': int(patch.size),
            'dominant_pixel_count': int(dist.get(class_code, 0))
        })

        if len(dataset) % 500 == 0:
            print(f"    ... labelled {len(dataset)} images so far")

    print(f"  Labelled: {len(dataset)}  |  Skipped: {skipped}")
    
    dataset_df = pd.DataFrame(dataset)
    
    if output_csv and len(dataset_df) > 0:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        dataset_df.to_csv(output_csv, index=False)
    
    return dataset_df


def train_test_split_dataset(dataset_df, test_size=0.4, random_state=42):
    """
    Perform 60/40 train-test split randomly and stratified by label.
    
    Args:
        dataset_df: Dataset DataFrame with 'label' column
        test_size: Test set proportion (0.4 = 40%)
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        dataset_df,
        test_size=test_size,
        stratify=dataset_df['label'],
        random_state=random_state
    )
    
    print(f"\nDataset Split:")
    print(f"Training set: {len(train_df)} images ({len(train_df)/len(dataset_df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} images ({len(test_df)/len(dataset_df)*100:.1f}%)")
    
    return train_df, test_df


def visualize_class_distribution(dataset_df, set_name='Full Dataset', save_path=None):
    """
    Visualize class distribution in dataset.
    
    Args:
        dataset_df: Dataset DataFrame
        set_name: Name of the dataset (for title)
        save_path: Path to save plot
    """
    class_counts = dataset_df['label'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    class_counts.plot(kind='bar', ax=ax, color='steelblue')
    
    ax.set_title(f'Class Distribution - {set_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Land-Use Class')
    ax.set_ylabel('Number of Images')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return class_counts


def visualize_sample_images_with_labels(train_df, image_directory, 
                                        num_samples=6, save_path=None):
    """
    Display sample images with their labels and patch information.
    """
    num_samples = min(num_samples, len(train_df))
    samples = train_df.sample(n=num_samples, random_state=42)
    
    fig, axes = plt.subplots(2, 3 if num_samples >= 3 else 2, figsize=(14, 8))
    axes = axes.flatten()
    
    for idx, (i, row) in enumerate(samples.iterrows()):
        image_path = row['image_path']
        label = row['label']
        
        try:
            img = Image.open(image_path)
            axes[idx].imshow(img)
        except:
            axes[idx].text(0.5, 0.5, 'Image not found', ha='center', va='center')
        
        axes[idx].set_title(f"Label: {label}\n({row['image_name']})", fontsize=10)
        axes[idx].axis('off')
    
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
