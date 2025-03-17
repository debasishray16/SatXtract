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
from osgeo import gdal, osr
from samgeo import SamGeo, tms_to_geotiff, get_basemaps
from streamlit_folium import st_folium
from streamlit_image_comparison import image_comparison

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

import os
os.environ["GDAL_DATA"] = r"C:\Anaconda\envs\osgeo_env\Library\share\gdal"
os.environ["PROJ_LIB"] = r"C:\Anaconda\envs\osgeo_env\Library\share\proj"
from osgeo import gdal, osr

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


CLASS_LABELS = {
    0: ("Not classified", (255, 255, 255)),  # White
    1: ("Buildings", (255, 0, 0)),           # Red
    2: ("Greenery", (0, 255, 0)),               # Blue
    3: ("Water", (0, 0, 255)),            # Green
    4: ("Roads", (128, 128, 128)),           # Gray
}



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



# API Key from Mapbox
MAPBOX_URL = (
    "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}?access_token="
    + st.secrets["MAPBOX_API_KEY"]
)

st.write("Mapbox API Key Loaded Successfully!")


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
    
    
'''
@st.cache_data
def show_prediction(image_path, selected_model):
    """
    Get and show the prediction for a given image.
    """
    # Read image
    with rasterio.open(image_path) as dataset:
        img_array = dataset.read()

    # Move channel information to third axis
    img_array = np.moveaxis(img_array, source=0, destination=2)

    # Load model
    model = load_model_from_file(
        Path("data/models", MODELS[selected_model]["file_name"])
    )

    # Get prediction
    prediction = get_smooth_prediction_for_file(img_array, model, 5, MODELS[selected_model]["backbone"], patch_size=256)

    # Prepare images for visualization
    img, segmented_img, overlay = prepare_split_image(img_array, prediction)

    # Save segmented image
    save_segmented_file(segmented_img, image_path, selected_model)

    gc.collect()
    return img, segmented_img, overlay'''


@st.cache_data
def show_prediction(image_path, selected_model):
    """
    Get and show the prediction for a given image with class labels.
    """
    # Read image
    with rasterio.open(image_path) as dataset:
        img_array = dataset.read()

    # Move channel information to third axis
    img_array = np.moveaxis(img_array, source=0, destination=2)

    # Load model
    model = load_model_from_file(Path("data/models", MODELS[selected_model]["file_name"]))

    # Get prediction
    prediction = get_smooth_prediction_for_file(img_array, model, 5, MODELS[selected_model]["backbone"], patch_size=256)

    # Convert prediction into a color-mapped image
    color_mapped = np.zeros((*prediction.shape, 3), dtype=np.uint8)

    for class_idx, (label, color) in CLASS_LABELS.items():
        color_mapped[prediction == class_idx] = color

    # Convert to PIL Image
    segmented_img = Image.fromarray(color_mapped)

    # Overlay segmentation on original image
    overlay = cv2.addWeighted(img_array.astype(np.uint8), 0.6, color_mapped, 0.4, 0)

    
    
    # Save segmented image
    save_segmented_file(segmented_img, image_path, selected_model)

    gc.collect()
    return img_array, segmented_img, overlay


def display_class_legend():
    """
    Display class labels and colors as a legend in Streamlit.
    """
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    patches = [mpatches.Patch(color=np.array(color) / 255, label=label) for label, color in CLASS_LABELS.values()]
    ax.legend(handles=patches, loc='center', ncol=len(CLASS_LABELS))

    st.pyplot(fig)

@st.cache_data
def save_segmented_file(segmented_img, source_path, selected_model):
    """Save a segmented image to a png file."""
    segmented_png_path = (
        source_path.parent.parent
        / "prediction"
        / f"{source_path.stem}_{selected_model}.png"
    )

    # Make sure image path exists
    Path(segmented_png_path).parent.mkdir(parents=True, exist_ok=True)

    segmented_img.save(segmented_png_path)

'''
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
'''


def create_map(location, zoom_start=18):
    """
    Create a Folium map with a satellite layer and a drawing tool.
    """
    # Create Folium map
    folium_map = folium.Map(location=location, zoom_start=zoom_start, control_scale=True)

    # Add the satellite layer
    folium.TileLayer(MAPBOX_URL, attr="Mapbox").add_to(folium_map)

    # Add drawing tool
    Draw(export=True).add_to(folium_map)

    return folium_map

def callback():
    st.toast(f"Current zoom: {st.session_state['my_map']['zoom']}")
    st.toast(f"Current center: {st.session_state['my_map']['center']}")



def tab_live_segmentation():
    """
    Streamlit app page to segment images directly from a map area.
    """
    st.title("Live Segmentation")

    col1, col2 = st.columns([0.5, 0.5])
    
    locations = {
        # Countries (Coordinates of the geographic center)
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

        # Capitals
        "New Delhi": [28.6139, 77.2090],
        "Washington, D.C.": [38.9072, -77.0369],
        "London": [51.5074, -0.1278],
        "Paris": [48.8566, 2.3522],
        "Berlin": [52.5200, 13.4050],
        "Tokyo": [35.6895, 139.6917],
        "Beijing": [39.9042, 116.4074],
        "Moscow": [55.7558, 37.6173],
        "Canberra": [-35.2809, 149.1300],
        "Brasília": [-15.8267, -47.9218],
        "Pretoria": [-25.7461, 28.1881]
    }

    placeholder = st.empty()

    with col1:
        # Dropdown for selecting a location
        selected_location = st.selectbox("Choose a location:", list(locations.keys()))

        # Retrieve coordinates based on selection
        location = locations[selected_location]

        # Display selected coordinates
        st.write("Selected Coordinates:", location)

                # Create a select box for the model
        selected_model = st.selectbox(
            "Model to use",
            MODELS.keys(),
            format_func=lambda x: MODELS[x]["description"],
            key="model_select_live",
        )
        

    with col2:

        
        # Create Folium map
        folium_map = create_map(location, zoom_start=6)

        # Render map
        output = st_folium(folium_map, width=805, height=805,on_change = callback, key="my_map")

        if output["all_drawings"] is not None:
            # Create image from bounding box
            if (
                len(output["all_drawings"]) > 0
                and output["all_drawings"][0]["geometry"]["type"] == "Polygon"
            ):
                with st.spinner("Extracting image..."):
                    # Get the bounding box of the drawn polygon
                    bbox = output["all_drawings"][0]["geometry"]["coordinates"][0]

                    # Convert for further use [xmin, ymin, xmax, ymax]
                    bbox = [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]]

                    image_path = (
                        "data/predict/app/source/satellite-from-leafmap"
                        f"-{str(time.time()).replace('.', '-')}.tif"
                    )

                    # Make sure image path exists
                    Path(image_path).parent.mkdir(parents=True, exist_ok=True)

                    # Save the selection as a GeoTIFF
                    tms_to_geotiff(
                        output=image_path,
                        bbox=bbox,
                        zoom=18,
                        source=MAPBOX_URL,
                        overwrite=True,
                    )

                # Check if image was created successfully and display it
                
                if Path(image_path).is_file():
                    placeholder.image(image_path, caption="Extracted image", width=700)

                if st.button("Segment", key="segment_button_live"):
                    with st.spinner("Segmenting image..."):
                        img, _, overlay = show_prediction(
                            Path(image_path), selected_model
                        )
                        
                        

                        # Show image comparison in placeholder container
                        with placeholder.container():
                            image_comparison(img1=img, img2=overlay, width=700)

    




def tab_segmentation_from_file():
    """
    Page to segment images from a file.
    """
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        # Create a select box for the model
        selected_model = st.selectbox(
            "Model to use",
            MODELS.keys(),
            format_func=lambda x: MODELS[x]["description"],
            key="model_select_file",
        )

    
    with col2:
        st.title("Segmentation from file")

        with st.spinner("Loading ..."):
            uploaded_file = st.file_uploader(
                "Upload an image file to segment it:", type=["tif", "tiff", "jpg", "jpeg", "png"]
            )

        placeholder = st.empty()

        if uploaded_file is not None:
            try:
                # Open the image from memory
                with MemoryFile(uploaded_file.getvalue()) as memfile:
                    with memfile.open() as dataset:
                        img_array = dataset.read()
                        img_meta = dataset.meta

                # Define file path
                output_dir = Path("data/predict/app/source")
                output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if not exists
                input_file_path = output_dir / uploaded_file.name

                # Convert PNG and JPEG to TIFF format since rasterio only supports TIFF
                if uploaded_file.name.lower().endswith(("png", "jpg", "jpeg")):
                    temp_tif_path = input_file_path.with_suffix(".tif")
                    img = Image.open(uploaded_file)
                    img = img.convert("RGB")
                    img.save(temp_tif_path)
                    input_file_path = temp_tif_path

                # Save the image
                with rasterio.open(input_file_path, "w", **img_meta) as dst:
                    dst.write(img_array)

                # Move channel information to third axis
                img_array = np.moveaxis(img_array, source=0, destination=2)

                # Display the image in the placeholder container
                placeholder.image(img_array, caption="Uploaded Image", use_container_width=True)

                # Show a button to start the segmentation
                if st.button("Segment", key="segment_button_file"):
                    with st.spinner("Segmenting ..."):
                        try:
                            img, segmented_img, overlay = show_prediction(input_file_path, selected_model)

                            with placeholder.container():
                                image_comparison(img1=img, img2=overlay, width=700)

                            # Show the class legend
                            display_class_legend()
                        except Exception as e:
                            st.error(f"Error during segmentation: {e}")

            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")

    '''with col2:
        st.title("Segmentation from file")

        with st.spinner("Loading ..."):
            uploaded_file = st.file_uploader("Upload an image file to segment it:",type=["tif", "tiff", "jpg", "jpeg", "png"],)

        placeholder = st.empty()

        if uploaded_file is not None:
            # Open the image from memory
            with MemoryFile(uploaded_file.getvalue()) as memfile:
                with memfile.open() as dataset:
                    img_array = dataset.read()
                    img_meta = dataset.meta

            # Define file path
            input_file_path = Path(f"data/predict/app/source/{uploaded_file.name}")

            # Write the image to the directory
            with rasterio.open(input_file_path, "w", **img_meta) as dst:
                dst.write(img_array)

            # Move channel information to third axis
            img_array = np.moveaxis(img_array, source=0, destination=2)

            # Display the image in the placeholder container
            placeholder.image(
                img_array, caption="Uploaded Image", use_container_width=True
            )

            
            if st.button("Segment", key="segment_button_file"):
                with st.spinner("Segmenting ..."):
                    img, segmented_img, overlay = show_prediction(input_file_path, selected_model)

                    with placeholder.container():
                        image_comparison(img1=img, img2=overlay, width=700)

                    # Show the class legend
                    display_class_legend()'''
    with col3:
        st.write("")




# @st.cache_data
def tab_show_examples():
    """
    Page to show some example images.
    """
    example_dir = Path("data/predict/examples/source")
    example_dir.mkdir(parents=True, exist_ok=True)  # ✅ Ensure the directory exists

    _, col2, _ = st.columns([1, 3, 1])

    with col2:
        st.title("Examples of segmentations")
        
        if example_dir.exists():
            # Create lists of images and segmentations
            image_paths = list(example_dir.iterdir())
        else:
            st.warning("No example images found. Please add some images to `data/predict/examples/source`.")
            return
        
        

        for model_key, model_values in MODELS.items():
            valid_images = []
            segmentations = []

            # Get images that have a source and a prediction file
            for image_path in image_paths:
                segmentation_path = (
                    image_path.parent.parent
                    / "prediction"
                    / f"{image_path.stem}_{model_key}.png"
                )

                # Skip if there is no segmentation file
                if not segmentation_path.is_file():
                    continue

                # Load image
                image = Image.open(image_path).convert("RGBA")

                # Skip if image is too small
                if image.size[0] < 700:
                    continue

                # Load segmentation
                segmentation = Image.open(segmentation_path).convert("RGBA")

                # Append to lists
                valid_images.append(image)
                segmentations.append(segmentation)

            # Show 6 images, or less if there are less than 6 images
            n_images = 6 if len(valid_images) > 6 else len(valid_images)

            if n_images > 0:
                st.subheader(model_values["description"])

            for i in range(n_images):
                # image = Image.open(valid_images[i]).convert("RGBA")
                # segmentation = Image.open(segmentations[i]).convert("RGBA")
                # overlay = Image.alpha_composite(image, segmentation)
                overlay = Image.alpha_composite(valid_images[i], segmentations[i])

                image_comparison(
                    img1=valid_images[i],
                    img2=overlay,
                    width=800,
                )
















def main():
    tab1, tab2 , tab3 = st.tabs(
        ['Live Segmentation','From Files','Examples']
    )
    
    with tab1:
        tab_live_segmentation()

    with tab2:
        tab_segmentation_from_file()


    with tab3:
        tab_show_examples()
    
if __name__ == "__main__":
    main()