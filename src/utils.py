import json
from pathlib import Path

import geopandas as geop
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.mask

from matplotlib import colors

from PIL import Image
from pylandtemp import emissivity , ndvi, single_window

from rasterio.warp import Resampling, calculate_default_transform , reproject
from rasterio.windows import get_data_window , shape, transform
from shapely.geometry import shape


# def clip_to_geojson(band_path, geojson_path, target_dir = None):
# def clip_jpg2_to_geojson(band_path,geojson_path, target_dir=None, driver_in = "JP2OpenJPEG",driver_out="GTiff"):
# def normalize(band,streatch=True):
# def brighten(band, alpha, beta=0):
# def gamma_corr(band, gamma):
# def create_rgb_composite(band_red_path: Path, band_green_path: Path, band_blue_path: Path, target_path: Path, stretch: bool=True, gamma: float = 1,alpha: float=1,beta: float=0, driver: str = "GTiff" ,) -> tuple[np.ndarray,str]:
# def create_rgb_composite2(band_red_path: Path, band_green_path: Path, band_blue_path: Path, targer_path: Path, normal: bool = True, stretch: bool = True, gamma: float = None, alpha: float =None, beta: float =0, driver_in: str ="GTiff", driver_out: str = "GTiff"):
# def resample_image(input_file, output_file, target_resolution):
# def calc_lst(band_4_path:Path, band_5_path:Path, band_10_path:Path, target_path:Path) -> np.ndarray:
# def calc_nvdi(band_4_path: Path, band_5_path:Path, target_path: Path): -> np.ndarray:
# def cal_emissivity(band_4_path: Path, band_5_path: Path, target_path:Path) ->np.ndarray:
# def exaggerate(input_array: np.ndarray, factor: float =2) -> np.ndarray:
# def reproject_geotiff(src_path, target_path, target_crs):
# def create_rgba_color_image(src_path: Path, target_path: Path, colormap="RdBu_r"):
# def clip_to_remove_nodata(input_path: Path, output_path: Path = None) -> None:
# def prepare_split_image(img: np.ndarray, prediction: np.ndarray) -> tuple[Image.Image, Image.Image, Image.Image]:


# GeoJSON coordinate reference system (CRS) , Area of Interest (AOI)

'''
The function clip_to_json is designed to clip (crop) a raster image (e.g., a satellite image) to a given GeoJSON boundary (e.g., a city's boundary or a region of interest). 
It saves the clipped raster and returns the new file path.
'''
def clip_to_json(band_path, geojson_path, target_dir = None):
    if target_dir is None:
        target_dir = band_path.parent
    
    # Load AOI (GeoJSON)
    aoi = gpd.read_file(geojson_path)
    
    with rasterio.open(band_path) as src:
        
        # Convert GeoJSON CRS to match raster CRS
        aoi = aoi.to_crs(src.crs)
        # Extract geometry for masking
        aoi_shape= [shape(aoi.geometry.loc[0])]
        
        # Clip raster to AOI
        out_image, out_transform = rasterio.mask.mask(src, aoi_shape, crop=True)
        out_meta = src.meta
        
    # Update metadata
    out_meta.update(
        {
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        }
    )
    
    # Define new file path
    file_path_new = target_dir / f"{band_path.stem}_clipped{band_path.suffix}"
    
    # Save clipped raster
    with rasterio.open(file_path_new, "w", **out_meta) as dest: 
        dest.write(out_image)
        
    print(f"Saved clipped file to {file_path_new}")
    return str(file_path_new)