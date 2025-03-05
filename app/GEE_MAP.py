import ee
import geemap

import os
import sys
import time
from pathlib import Path
import gc

import folium
import numpy as np
import rasterio
import streamlit as st
from folium.plugins import Draw
from keras.models import load_model
from PIL import Image
from rasterio.io import MemoryFile
from samgeo import SamGeo, tms_to_geotiff, get_basemaps
from streamlit_folium import st_folium
from streamlit_image_comparison import image_comparison

st.set_page_config(
    page_title="SatXtract",
    page_icon="🛰️",
    layout="wide"
)

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if module_path not in sys.path:
    sys.path.append(module_path)

from predict import get_smooth_prediction_for_file
from utils import prepare_split_image



import ee
import streamlit as st

def initialize_gee(project_id):
    try:
        ee.Initialize()  # Try initializing without authentication
        st.success("Google Earth Engine initialized successfully!")
    except ee.ee_exception.EEException:
        st.warning("Google Earth Engine authentication required.")
        try:
            ee.Authenticate()  # Authenticate only when needed
            ee.Initialize(project=project_id)
            st.success("Google Earth Engine authenticated and initialized successfully!")
        except Exception as e:
            st.error(f"Error during authentication: {e}")

# Call the function
initialize_gee('ee-dd7046')




# Function to get NDVI, Emissivity, and LST from GEE
def get_gee_data(bbox):
    """Fetch satellite data and compute NDVI, Emissivity, and LST."""
    # Define a region of interest (ROI) from the selected bounding box
    roi = ee.Geometry.Polygon([bbox])

    # Load Landsat 8 Collection (Modify for Sentinel if needed)
    landsat = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
        .filterBounds(roi)
        .filterDate("2023-01-01", "2023-12-31")  # Modify date range
        .sort("CLOUD_COVER")
        .first()
    )

    # Extract Red, NIR, and Thermal bands
    red_band = landsat.select("B4")
    nir_band = landsat.select("B5")
    tir_band = landsat.select("B10")  # Thermal band

    # Compute NDVI
    ndvi = nir_band.subtract(red_band).divide(nir_band.add(red_band)).rename("NDVI")

    # Compute Emissivity (Example formula)
    emissivity = ndvi.multiply(0.004).add(0.986).rename("Emissivity")

    # Compute LST (Land Surface Temperature)
    lst = tir_band.divide(ee.Number(1).add(ee.Number(10.8).multiply(tir_band).divide(14380).multiply(emissivity.log()))).rename("LST")

    return ndvi, emissivity, lst




# Function to download and visualize results
def plot_and_save_results(ndvi_array, emissivity_array, lst_array):
    """Generate plots and save as PNG."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ndvi_array, cmap='RdYlGn')
    axes[0].set_title("NDVI")

    axes[1].imshow(emissivity_array, cmap='coolwarm')
    axes[1].set_title("Emissivity")

    axes[2].imshow(lst_array, cmap='hot')
    axes[2].set_title("LST")

    plt.tight_layout()

    # Save as PNG
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(temp_file.name, dpi=300)
    plt.close(fig)

    return temp_file.name


MODELS = {
    "resnet34-epoch-25": {
        "description": "ResNet-34-Epoch-25",
        "file_name": "landcover_resnet34_25_epochs_batch16_freeze.hdf5",
        "backbone": "resnet34",
    },
    "resnet34-epoch-50": {
        "description": "ResNet-34-Epoch-50",
        "file_name": "landcover_resnet34_50_epochs_batch16_freeze.hdf5",
        "backbone": "resnet34",
    },
    "resnet34-epoch-75": {
        "description": "ResNet-34-Epoch-75",
        "file_name": "landcover_resnet34_75_epochs_batch16_freeze.hdf5",
        "backbone": "resnet34",
    },
    "resnet34-epoch-100": {
        "description": "ResNet-34-Epoch-100",
        "file_name": "landcover_resnet34_100_epochs_batch16_freeze.hdf5",
        "backbone": "resnet34",
    },
    
    
    "resnet50-epoch-25": {
        "description": "ResNet-50-Epoch-25",
        "file_name": "landcover_resnet50_25_epochs_batch16_freeze.hdf5",
        "backbone": "resnet50",
    },
    "resnet50-epoch-50": {
        "description": "ResNet-50-Epoch-50",
        "file_name": "landcover_resnet50_50_epochs_batch16_freeze.hdf5",
        "backbone": "resnet50",
    },
    "resnet50-epoch-75": {
        "description": "ResNet-50-Epoch-75",
        "file_name": "landcover_resnet50_75_epochs_batch16_freeze.hdf5",
        "backbone": "resnet50",
    },
    "resnet50-epoch-100": {
        "description": "ResNet-50-Epoch-100",
        "file_name": "landcover_resnet50_100_epochs_batch16_freeze.hdf5",
        "backbone": "resnet50",
    },
    
    
    
    
    "resnet101-epoch-25": {
        "description": "ResNet-101-Epoch-25",
        "file_name": "landcover_resnet101_25_epochs_batch16_freeze.hdf5",
        "backbone": "resnet101",
    },
    
    "resnet101-epoch-50": {
        "description": "ResNet-101-Epoch-50",
        "file_name": "landcover_resnet101_50_epochs_batch16_freeze.hdf5",
        "backbone": "resnet101",
    },
    
    "resnet101-epoch-75": {
        "description": "ResNet-101-Epoch-75",
        "file_name": "landcover_resnet101_75_epochs_batch16_freeze.hdf5",
        "backbone": "resnet50",
    },
    "resnet101-epoch-100": {
        "description": "ResNet-101-Epoch-100",
        "file_name": "landcover_resnet101_100_epochs_batch16_freeze.hdf5",
        "backbone": "resnet101",
    },
}






@st.cache_resource
def load_model_from_file(model_path):
    """ Load a model from a file, ensuring it exists first. """
    
    model_file = Path(model_path)

    if not model_file.exists():
        st.error(f" Model file not found: {model_path}")
        return None  # Prevents the app from crashing

    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f" Error loading model: {e}")
        return None
    


def create_map(location, zoom_start=19):
    """
    Creates a Folium map centered at a specific location.
    
    Args:
        location (list): [latitude, longitude] of the center point.
        zoom_start (int): Zoom level for the map.
    
    Returns:
        folium.Map: A Folium map object.
    """
    m = folium.Map(location = location, zoom_start = zoom_start, control_scale=True, tiles="Esri.WorldImagery") #can use: OpenStreetMap
    Draw(export=True).add_to(m)  # Add drawing tools for user interaction
    return m



def callback():
    st.toast(f"Current zoom: {st.session_state['my_map']['zoom']}")
    st.toast(f"Current center: {st.session_state['my_map']['center']}")



def tab_live_segmentation():
    """
    Streamlit app page to segment images directly from a map area using Google Earth Engine.
    """
    st.title("Live Segmentation with GEE")

    col1, col2 = st.columns([0.2, 0.8])

    locations = {
        "India": [20.5937, 78.9629],
        "United States": [37.0902, -95.7129],
        "United Kingdom": [55.3781, -3.4360],
        "France": [46.6034, 1.8883],
        "Germany": [51.1657, 10.4515],
        "Japan": [36.2048, 138.2529],
        "China": [35.8617, 104.1954],
        "Russia": [61.5240, 105.3188],
        "Australia": [-25.2744, 133.7751],
        "Brazil": [-14.2350, -51.9253],
        "South Africa": [-30.5595, 22.9375],
    }

    with col1:
        selected_location = st.selectbox("Choose a location:", list(locations.keys()))
        location = locations[selected_location]
        st.write("Selected Coordinates:", location)

    with col2:
        import folium

        folium_map = create_map(location)
        output = st_folium(folium_map, width=805, height=600, key="my_map")


        if output["all_drawings"] is not None:
            if len(output["all_drawings"]) > 0 and output["all_drawings"][0]["geometry"]["type"] == "Polygon":
                with st.spinner("Extracting image..."):
                    bbox = output["all_drawings"][0]["geometry"]["coordinates"][0]
                    bbox = [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]]

                    # Get GEE Data
                    ndvi, emissivity, lst = get_gee_data(bbox)

                    # Convert GEE Image to NumPy Arrays
                    ndvi_array = np.array(ndvi.sampleRectangle(region=ee.Geometry.Polygon([bbox])).getInfo()["properties"]["NDVI"])
                    emissivity_array = np.array(emissivity.sampleRectangle(region=ee.Geometry.Polygon([bbox])).getInfo()["properties"]["Emissivity"])
                    lst_array = np.array(lst.sampleRectangle(region=ee.Geometry.Polygon([bbox])).getInfo()["properties"]["LST"])

                    # Generate Image
                    png_file = plot_and_save_results(ndvi_array, emissivity_array, lst_array)

                    # Display and Download PNG
                    image = Image.open(png_file)
                    st.image(image, caption="Generated LST, NDVI, and Emissivity", use_column_width=True)

                    with open(png_file, "rb") as file:
                        st.download_button(label="Download PNG", data=file, file_name="GEE_Analysis.png", mime="image/png")


def main():

    tab_live_segmentation()

    
if __name__ == "__main__":
    main()