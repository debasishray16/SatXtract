import json
from pathlib import Path

import geopandas as gpd
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
import rasterio
import numpy as np
from rasterio.crs import CRS


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
The function clip_to_json is designed to clip (crop) a raster image (e.g., a satellite image) 
to a given GeoJSON boundary (e.g., a city's boundary or a region of interest). 
It saves the clipped raster and returns the new file path.
'''
def clip_to_geojson(band_path, geojson_path, target_dir = None):
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



'''
This function clips a JP2 (JPEG 2000) raster file (e.g., Sentinel-2 satellite images) 
to a specified GeoJSON boundary and saves it as a GeoTIFF (.tiff).

band_path: Path to the input .jp2 raster file (Sentinel-2 image).
geojson_path: Path to the GeoJSON file containing the clipping boundary (AOI).
target_dir (optional): Directory where the clipped .tiff file will be saved. Defaults to the same directory as band_path.
driver_in="JP2OpenJPEG": Specifies the JPEG 2000 driver for reading .jp2 files.
driver_out="GTiff": Specifies the GeoTIFF driver for saving the clipped raster.

'''
def clip_jp2_to_geojson(band_path, geojson_path, target_dir =None, driver_in = "JP2OpenJPEG",driver_out= "GTiff"):
    if target_dir is None:
        target_dir = band_path.parent
    
    # Load AOI (GeoJSON)
    aoi = gpd.read_file(geojson_path)
    
    with rasterio.open(band_path, driver=driver_in) as src:
        
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
            "height":out_image.shape[1],
            "width":out_image.shape[2],
            "transform":out_transform,
            "crs":src.crs,
            "driver":driver_out,
        }
    )
    
    file_path_new = target_dir /f"{band_path.stem}_clipped.tiff"
    # Save the clipped raster to a new GeoTiff file
    with rasterio.open(file_path_new, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"Saved clipped file to {file_path_new}")

    return str(file_path_new)



'''
This function normalizes a raster band (image array) to a range of 0 to 1 (or 0 to 255 if stretching is enabled).
band: A NumPy array representing a raster band (e.g., Sentinel-2 image).
stretch=True (default): If True, scales the output to 0-255 (8-bit image range).
'''
def normalize(band,stretch = True):
    band_min , band_max = (band.min(),band.max())
    
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_band = (band - band_min) / (band_max - band_min)
    
    '''
    Handles Division by Zero Errors:
    If band_max == band_min, this will prevent NaN or Inf values.
    '''
    if stretch:
        normalized_band = normalized_band * 255
    return normalized_band


'''
This function adjusts the brightness of a raster band (image array) using a linear transformation.
-> alpha controls contrast (higher values make the image sharper).
-> beta controls brightness (positive values brighten the image, negative values darken it).
'''
def brighten(band,alpha,beta=0):
    return np.clip(alpha*band+beta,0,255)


'''
This function applies gamma correction to a raster band (image array) using the formula
-> output = band **(1/gamma)
gamma adjusts the brightness/contrast non-linearly.
gamma > 1: Darkens the image.
gamma < 1: Brightens the image.

'''
def gamma_corr(band,gamma):
    return np.power(band, 1/gamma)




'''
This function creates an RGB composite image from three raster bands (Red, Green, and Blue) while applying:
-> Gamma correction
-> Brightness/contrast adjustment
-> Normalization

It then saves the processed RGB image as a GeoTIFF file.

band_red_path, band_green_path, band_blue_path → File paths of Red, Green, and Blue bands (GeoTIFF files).
target_path → Path to save the RGB composite image.
stretch=True → Apply normalization (0-255 scaling).
gamma=1 → Gamma correction factor (<1 brightens, >1 darkens).
alpha=1, beta=0 → Controls brightness & contrast.
driver="GTiff" → Output format (GeoTIFF).
'''
def create_rgb_composite(band_red_path: Path, band_green_path:Path, band_blue_path:Path, target_path: Path, stretch: bool =True, gamma: float = 1, alpha: float =1, beta: float =0, driver: str = "GTiff",) -> tuple[np.ndarray,str]:
    
    # Opens each raster file and reads Band 1 (first layer).
    with rasterio. open(band_blue_path, driver=driver) as src:
        band_blue = src.read(1)
    with rasterio.open(band_green_path, driver=driver) as src:
        band_green = src.read(1)
    with rasterio.open(band_red_path, driver=driver) as src:
        band_red = src.read(1)
        meta = src.meta
    
    
    # Apply gamma correction (default: gamma=1 means no change)
    red_g = gamma_corr(band_red, gamma=gamma)
    blue_g = gamma_corr(band_blue, gamma=gamma)
    green_g = gamma_corr(band_green, gamma=gamma)
    
    
    # Apply brightness correction (default: alpha=1 means no change)
    red_gb = brighten(red_g, alpha=alpha, beta=beta)
    blue_gb = brighten(blue_g, alpha=alpha, beta=beta)
    green_gb = brighten(green_g, alpha=alpha, beta=beta)
    
    
    # Normalize the bands to the range 0-1
    red_gbn = normalize(red_gb, stretch=stretch)
    blue_gbn = normalize(blue_gb, stretch=stretch)
    green_gbn = normalize(green_gb, stretch=stretch)
    
    
    # After gamma , brightness and normalization of image
    # Stack RGB Bands Together
    rgb_composite_gbn = np.dstack((red_gbn, blue_gbn, green_gbn))
    
    # Prepare Output Directory
    target_path.parent.mkdir(parents= True, exist_ok=True)
    
    # Modifies metadata to store a 3-band (RGB) image with 8-bit pixel values (uint8).
    meta.update(
        {
            "count":3, # 3 bands
            "dtype":"uint8",
            "nodata":0,
        }
    )
    
    # Save the RGB composite to a new GeoTiff file
    with rasterio.open(target_path, "w", height=band_red.shape[0],width = band_red.shape[1],) as dst:
        # Move channel information to first axis
        out_array =np.moveaxis(rgb_composite_gbn, source=2, destination=0)
        dst.write(out_array.astype("unit8"))
    
    if (target_path.is_file()):
        print(f"RGB Composite saved to {target_path}")
    
    return rgb_composite_gbn.astype("unit8"),target_path



'''
This function creates an RGB composite image from three raster bands (Red, Green, and Blue) with optional:
-> Gamma correction
-> Brightness/contrast adjustment
-> Normalization
-> Saving in a specified raster format (driver_out)
This will generate an enhanced RGB image from Sentinel-2 bands.
'''
def create_rgb_composite2(band_red_path: Path, band_green_path: Path, band_blue_path: Path, target_path: Path, normal:bool =True, stretch: bool = True, gamma: float = None, alpha: float = None, beta: float = 0, driver_in: str= "GTiff",driver_out: str= "GTiff",):
    
    # Open the raster bands using context manager
    band_red = rasterio.open(band_red_path, driver=driver_in)
    band_red_read  = band_red.read(1)
    
    band_green = rasterio.open(band_green_path, driver=driver_in)
    band_green_read  = band_green.read(1)
    
    band_blue = rasterio.open(band_blue_path, driver=driver_in)
    band_blue_read  = band_blue.read(1)
    
    # Apply gamma correction if provided
    if gamma is not None:
        
        band_red_read = gamma_corr(band_red_read, gamma=gamma)
        band_green_read = gamma_corr(band_green_read, gamma=gamma)
        band_blue_read = gamma_corr(band_blue_read, gamma=gamma)
    
    # Apply brightness/contrast adjustment if provided
    if alpha is not None:
        
        band_red_read = brighten(band_red_read, alpha=alpha, beta=beta)
        band_green_read = brighten(band_green_read, alpha=alpha, beta=beta)
        band_blue_read = brighten(band_blue_read, alpha=alpha, beta=beta)
    
    # Normalize bands if required
    if normal:
        
        band_red_read = normalize(band_red_read, stretch=stretch)
        band_green_read = normalize(band_green_read, stretch=stretch)
        band_blue_read = normalize(band_blue_read, stretch=stretch)
        
    
    # Ensure target directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Update metadata for RGB composite
    rgb = rasterio.open(target_path, "w",driver=driver_out, width = band_red.width, height = band_red.height, count=3 , crs=band_red.crs, transform = band_red.transform, dtype = "uint16",)
    
    rgb.write_band(band_red_read,1)
    rgb.write_band(band_green_read,2)
    rgb.write_band(band_blue_read,3)
    rgb.close()
    
    if target_path.is_file():
        print(f"RGB Composite saved to {target_path}")
    
    return target_path


'''
Your resample_image function is designed to resample a raster image to a specified ground resolution (in meters).
Using Bilinear interpolation method.
'''
def resample_image(input_file, output_file, target_resolution):
    
    with rasterio.open(input_file) as dataset:
        current_resolution = dataset.res[0]
        resampling_factor = current_resolution / target_resolution
        
        width = int(dataset.width / resampling_factor)
        height =  int(dataset.height / resampling_factor)
        
        # Update metadata for new resolution
        profile = dataset.profile
        profile.update(width=width, height=height, transform = dataset.transform * dataset.transform.scale(resampling_factor,resampling_factor),)
        
        with rasterio.open(output_file, "w" ,**profile) as dst:
            for i in range(1, dataset.count + 1):
                # This downsamples/upsamples the raster to the specified resolution using bilinear interpolation.
                data = dataset.read(i, out_shape=(height,width), resampling=Resampling.bilinear)
                dst.write(data,i)
    
    return output_file




'''
Your function calc_lst is computing Land Surface Temperature (LST) from Landsat 8 bands 4, 5, and 10 using the single_window function from pylandtemp.
The single_window algorithm is a Land Surface Temperature (LST) retrieval method used for Landsat 8 data. It is implemented in the pylandtemp library 
and is useful for computing LST from thermal infrared bands (TIR).
'''
def calc_lst(band_4_path: Path, band_5_path: Path, band_10_path: Path, target_path: Path) -> np.ndarray:
    
    with rasterio.open(band_4_path) as src:
        band_4 =  src.read(1)
    with rasterio.open(band_5_path) as src:
        band_5 =  src.read(1)
    with rasterio.open(band_10_path) as src:
        band_10 =  src.read(1)
        
        
        out_meta = src.meta.copy()
    
    # Compute Land Surface Temperature (LST)
    lst_image_array = single_window(band_10,band_4,band_5, unit="celsius")
    
    # For some reason, the values are in Kelvin, so we need to convert them to Celsius
    lst_image_array = lst_image_array - 273.15
    
    out_meta.update(
        {
            "height": lst_image_array.shape[0],
            "width": lst_image_array.shape[1],
            "transform": src.transform,
            "dtype": lst_image_array.dtype,
        }
    )
    
    # Save LST as GeoTIFF
    with rasterio.open(target_path, "w", **out_meta) as dst:
        dst.write(lst_image_array, 1)
    print("Saved Land Surface Temperature (LST) to", target_path)

    return lst_image_array




'''
calc_ndvi function computes the Normalized Difference Vegetation Index (NDVI) 
using Landsat 8 Bands 4 (Red) and 5 (NIR) and saves the result as a GeoTIFF.
'''
def calc_ndvi(band_4_path: Path, band_5_path: Path, target_path: Path) -> np.ndarray:
    
    with rasterio.open(band_4_path) as src:
        band_4 = src.read(1)
    with rasterio.open(band_5_path) as src:
        band_5 = src.read(1)
        
        out_meta = src.meta.copy()
    
    mask = band_4 == 0
    ndvi_image_array = ndvi(band_5,band_4,mask=mask)
    
    out_meta.update(
        {
            "height": ndvi_image_array.shape[0],
            "width": ndvi_image_array.shape[1],
            "transform": src.transform,
            "dtype": ndvi_image_array.dtype,
        }
    )
    
    # Save NDVI as GeoTIFF
    with rasterio.open(target_path, "w", **out_meta) as dst:
        dst.write(ndvi_image_array, 1)
    print("Save NVDI to ", target_path)
    
    return ndvi_image_array


'''
calc_emissivity function calculates land surface emissivity using NDVI and an emissivity estimation method (xiaolei)
'''
def calc_emissivity(band_4_path: Path, band_5_path: Path, target_path: Path) -> np.ndarray:
    
    with rasterio.open(band_4_path) as src:
        band_4 = src.read(1)
    with rasterio.open(band_5_path) as src:
        band_5 = src.read(1)
        
        out_meta = src.meta.copy()
    
    mask = band_4 == 0
    
    ndvi_image_array = ndvi(band_5, band_4, mask=mask)

    emissivity_10_array, emissivity_11_array = emissivity(
        ndvi_image_array, emissivity_method="xiaolei", landsat_band_4=band_4
    )
    
    
    out_meta.update(
        {
            "height": emissivity_10_array.shape[0],
            "width": emissivity_10_array.shape[1],
            "transform": src.transform,
            "dtype": emissivity_10_array.dtype,
        }
    )
    
    # Save emissivity as GeoTIFF
    with rasterio.open(target_path, "w", **out_meta) as dst:
        dst.write(emissivity_10_array, 1)

    print("Saved Emissivity to", target_path)

    return emissivity_10_array



'''
This function exagerrates the value of an array.
The exaggerate function enhances variations in temperature data, specifically emphasizing hotter-than-average regions.
'''
def exaggerate(input_array: np.ndarray, factor: float =2) -> np.ndarray:
    
    # Calculate the mean temperature
    mean_temp = np.mean(input_array)

    valid_mask = (input_array != 0) & (input_array > mean_temp)

    deviation = np.where(valid_mask, input_array - mean_temp, 0)

    # Calculate the deviation from the mean
    # deviation = input_array - mean_temp

    # Apply an exaggeration function to the deviation
    exaggerated_deviation = np.power(deviation, factor)

    # Add the exaggerated deviation back to the mean temperature
    exaggerated_temperature = mean_temp + exaggerated_deviation

    return exaggerated_temperature


'''
Your reproject_geotiff function aims to reproject a GeoTIFF raster to a new coordinate reference system (CRS)
Reproject a GeoTIFF raster to a different coordinate reference system (CRS).
    
    Parameters:
    - src_path (str or Path): Path to the source GeoTIFF file.
    - target_path (str or Path): Path to save the reprojected GeoTIFF.
    - target_crs (int or str): EPSG code or Proj string for the target CRS.

    Returns:
    - numpy.ndarray: The reprojected image array.
'''
def reproject_geotiff(src_path, target_path,target_crs):
    
    with rasterio.open(src_path) as src:
        img = src.read().astype(np.float32)
        
        # Set the nodata values to NaN
        img[img == src.nodata] = np.nan
        
        # Define target CRS (e.g., WGS84)
        target_crs = rasterio.crs.CRS.from_epsg(target_crs)

        # Calculate the default transform for the reprojected image
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        # Create an array to hold the reprojected image
        reprojected_img = np.zeros((img.shape[0], height, width), dtype=img.dtype)
        
        # Reproject the image to match the transformed bounds
        reproject(
            src.read(),
            reprojected_img,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
        )
        
        # Update the metadata for the reprojected image
        reprojected_meta = src.meta.copy()
        reprojected_meta.update(
            {
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        # Set the nodata values to NaN
        reprojected_img[reprojected_img == src.nodata] = np.nan

        # Write the reprojected image to a new GeoTIFF file
        with rasterio.open(target_path, "w", **reprojected_meta) as dst:
            dst.write(reprojected_img)

        return dst, reprojected_img
    
    
    

'''
The function maps raster values to an RGBA color representation and saves the result as:

A PNG image (for visualization).
A GeoTIFF file (with 4 bands for Red, Green, Blue, and Alpha transparency).
A JSON file (containing the spatial bounds of the raster).
It helps in visualizing raster data (e.g., temperature, elevation, NDVI, etc.) by applying a colormap (such as "RdBu_r" for a red-blue gradient).
'''
def create_rgba_color_image(src_path: Path, target_path: Path, colormap="RdBu_r"):
    
    '''
    Function to map raster values to rgba.

    Parameters
    ----------
    src_path : Path
        Path to the source raster file. Assumes that the raster has a single band.
    target_path : Path
        Path to the target raster file. Raster will have 4 bands.
    '''
    
    with rasterio.open(str(src_path)) as src:
        band = src.read(1)
        bounds = src.bounds
        meta = src.meta.copy()

    # Normalize band values to range [0, 1]
    band_norm = (band - np.nanmin(band)) / (np.nanmax(band) - np.nanmin(band))

    # Get minimum and maximum, ignoring NaNs
    vmin = np.nanmin(band_norm)
    vmax = np.nanmax(band_norm)

    # For values less than vmin, assign 0
    band_norm[band_norm < vmin] = 0

    # Linear stretch normalization to range [0, 1]
    band_norm = (band_norm - vmin) / (vmax - vmin)

    # Create a colormap
    cmap = plt.get_cmap(colormap)
    norm = colors.Normalize(vmin=0, vmax=1)

    # Apply the colormap to the normalized band values
    image = cmap(norm(band_norm))

    # Convert the float values in [0,1] to uint8 values in [0,255]
    image = (image * 255).astype(np.uint8)

    # Create paths for the PNG and JSON files
    png_image_path = target_path.parent / (target_path.stem + ".png")
    json_path = target_path.parent / (target_path.stem + "_bounds.json")

    # Save the image as a PNG
    plt.imsave(png_image_path, image)

    # Save the bounds in a JSON file
    bbox = [
        [bounds.bottom, bounds.left],
        [bounds.top, bounds.right],
    ]
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(bbox, file)

    # Update metadata
    meta.update(count=4, dtype=rasterio.uint8)

    # Write the reprojected image to a new GeoTIFF file
    with rasterio.open(target_path, "w", **meta) as dst:
        # Move channel information to first axis
        img_rio = np.moveaxis(image, source=2, destination=0)
        dst.write(img_rio.astype(rasterio.uint8))

    return dst, image



'''
The function removes the NoData (empty) regions from a raster by clipping it to the data window (the smallest bounding box that contains actual data). 
This reduces file size and improves efficiency when processing raster datasets.
'''
def clip_to_remove_nodata(input_path: Path, output_path: Path = None) -> None:
    '''
    Clip a raster to the data window to remove nodata values.
    TODO: Doesn't clip. https://gis.stackexchange.com/a/428982
    '''
    
    if output_path is None:
        output_path = Path(
            input_path.parent, input_path.stem + "_clipped" + input_path.suffix
        )

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        data_window = get_data_window(src.read(masked=True))
        data_transform = transform(data_window, src.transform)
        profile.update(
            transform=data_transform,
            height=data_window.height,
            width=data_window.width,
        )

        data = src.read(window=data_window)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)

    # Save the clipped raster to a new file.
    if output_path.is_file():
        print(f"Clipped file saved to {output_path}")


'''
This function prepares and visualizes segmented predictions
by applying color mapping to different classes in a prediction mask 
and overlaying it onto the original image.

Color Mapping: Assigns RGBA colors to different classification labels (e.g., buildings, trees, water, roads).
Create a Colored Mask: Converts the prediction array into an RGBA image based on the defined color mapping.
Generate an Overlay Image: Merges the original image with the colored mask for better visualization.

'''
def prepare_split_image(img: np.ndarray, prediction: np.ndarray) -> tuple[Image.Image, Image.Image, Image.Image]:
    '''
    Prepare images for display in a split view.
    '''

    # Map the values from 0-4 to RGBA colors (you can choose any colors)
    colors = {
        0: (0, 0, 0, 0),  # Unclassified (transparent)
        1: (239, 131, 84, 200),  # Buildings
        2: (22, 219, 147, 200),  # Trees
        3: (38, 103, 255, 200),  # Water
        4: (224, 202, 60, 200),  # Roads
    }

    # Prepare an empty array for the colored image (height x width x 4 for RGBA)
    colored_image = np.zeros((*prediction.shape, 4), dtype=np.uint8)

    # Apply colors
    for val, color in colors.items():
        colored_image[prediction == val] = color

    # Convert numpy array to PIL image
    original_img = Image.fromarray(img).convert("RGBA")
    segmented_img = Image.fromarray(colored_image)

    # Create an overlay image
    overlay = Image.alpha_composite(original_img, segmented_img)

    return original_img, segmented_img, overlay




