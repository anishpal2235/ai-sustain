"""
Q1: Spatial Reasoning & Data Filtering
"""
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box, Point
import os
import json


def plot_delhi_ncr_with_grid(delhi_ncr_shapefile, grid_spacing_km=60,
                             output_path=None):
    """
    Plot Delhi-NCR shapefile with a uniform grid overlay.

    The grid is constructed in **EPSG:32644** (UTM zone 44N) for metric
    accuracy, then reprojected to EPSG:4326 for display.

    Args:
        delhi_ncr_shapefile: Path to Delhi-NCR shapefile (EPSG:4326).
        grid_spacing_km: Grid cell size in kilometres (default 60).
        output_path: Path to save the plot.

    Returns:
        GeoDataFrame with grid geometries in EPSG:4326.
    """
    # Load shapefile
    delhi_ncr = gpd.read_file(delhi_ncr_shapefile)
    if delhi_ncr.crs is None:
        delhi_ncr = delhi_ncr.set_crs('EPSG:4326')
    print(f"Delhi-NCR Bounds (4326): {delhi_ncr.total_bounds}")

    # Reproject to EPSG:32644 for metric gridding
    delhi_ncr_utm = delhi_ncr.to_crs('EPSG:32644')
    minx, miny, maxx, maxy = delhi_ncr_utm.total_bounds
    print(f"Delhi-NCR Bounds (32644): [{minx:.0f}, {miny:.0f}, {maxx:.0f}, {maxy:.0f}] m")

    # Build grid in metres
    grid_spacing_m = grid_spacing_km * 1000
    grid_cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            grid_cells.append(box(x, y, x + grid_spacing_m, y + grid_spacing_m))
            y += grid_spacing_m
        x += grid_spacing_m

    grid_gdf_utm = gpd.GeoDataFrame(geometry=grid_cells, crs='EPSG:32644')
    grid_gdf = grid_gdf_utm.to_crs('EPSG:4326')  # back to geographic
    print(f"Created {len(grid_cells)} grid cells ({grid_spacing_km}×{grid_spacing_km} km)")

    # Plot in geographic coordinates
    fig, ax = plt.subplots(figsize=(14, 12))
    delhi_ncr.plot(ax=ax, facecolor='lightblue', edgecolor='black', linewidth=2)
    grid_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1, alpha=0.7)
    ax.set_title(f'Delhi-NCR Region with {grid_spacing_km}×{grid_spacing_km} km '
                 'Uniform Grid\n(Grid built in EPSG:32644, displayed in EPSG:4326)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    return grid_gdf


def filter_images_by_region(image_metadata, delhi_airshed_shapefile, output_path=None):
    """
    Filter satellite images whose center coordinates fall inside Delhi-Airshed region.
    
    Args:
        image_metadata: DataFrame (or CSV path) with columns 'image_name', 'latitude', 'longitude'
        delhi_airshed_shapefile: Path to Delhi-Airshed GeoJSON / shapefile
        output_path: Path to save filtered metadata (CSV)
        
    Returns:
        DataFrame with filtered images and metadata
    """
    import pandas as pd

    # Load metadata
    if isinstance(image_metadata, str):
        metadata = pd.read_csv(image_metadata)
    else:
        metadata = image_metadata.copy()

    # Load Delhi-Airshed region
    delhi_airshed = gpd.read_file(delhi_airshed_shapefile)
    # Ensure CRS is EPSG:4326
    if delhi_airshed.crs is None:
        delhi_airshed = delhi_airshed.set_crs('EPSG:4326')

    # Create point geometries from coordinates
    geometry = [Point(xy) for xy in zip(metadata['longitude'], metadata['latitude'])]
    gdf = gpd.GeoDataFrame(metadata, geometry=geometry, crs='EPSG:4326')

    # Spatial join to filter images within region
    filtered = gpd.sjoin(gdf, delhi_airshed, how='inner', predicate='within')

    print(f"Total images before filtering: {len(metadata)}")
    print(f"Total images after filtering:  {len(filtered)}")
    print(f"Images removed:                {len(metadata) - len(filtered)}")

    # Drop spatial-join helper columns and geometry for a clean CSV
    drop_cols = [c for c in filtered.columns if c in ('geometry', 'index_right')]
    result = pd.DataFrame(filtered.drop(columns=drop_cols))

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path, index=False)

    return result


def create_spatial_grid_epsg32644(delhi_airshed_shapefile, grid_spacing_km=60):
    """
    Create spatial grid in EPSG:32644 (UTM Zone 44N) for India.
    
    Args:
        delhi_airshed_shapefile: Path to shapefile
        grid_spacing_km: Grid cell size in kilometers
        
    Returns:
        GeoDataFrame with grid in EPSG:32644
    """
    # Load and reproject to UTM
    delhi_airshed = gpd.read_file(delhi_airshed_shapefile)
    delhi_airshed_utm = delhi_airshed.to_crs('EPSG:32644')
    
    minx, miny, maxx, maxy = delhi_airshed_utm.total_bounds
    
    # Create grid in meters
    grid_size_m = grid_spacing_km * 1000
    
    grid_cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x + grid_size_m, y + grid_size_m)
            grid_cells.append(cell)
            y += grid_size_m
        x += grid_size_m
    
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs='EPSG:32644')
    
    # Reproject back to EPSG:4326 for visualization with images
    grid_gdf_4326 = grid_gdf.to_crs('EPSG:4326')
    
    return grid_gdf, grid_gdf_4326
