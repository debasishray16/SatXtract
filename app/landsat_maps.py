import json
from pathlib import Path
import folium
import numpy as np
import streamlit as st
from PIL import Image

from streamlit_folium import st_folium




@st.cache
def load_Raster(file_path):
    map_layer_array =  np.array(Image.open(file_path))
    
    with open(file_path.parent / f"{file_path.stem}_bounds.json", "r", encoding="utf-8") as json_file:
        map_layer_bounds = json.load(json_file)

    return map_layer_array, map_layer_bounds


"""
Create overlay layer for map.
"""
def create_overlay(file_path, map_object, name):
    map_layer_array, map_layer_bounds = load_Raster(file_path)

    folium.raster_layers.ImageOverlay(
        image=map_layer_array,
        name=name,
        opacity=0.75,
        bounds=map_layer_bounds,
    ).add_to(map_object)


def create_map(raster_path, names):
    """
    Create map with overlay.
    """
    _ , map_layer_bounds = load_Raster(raster_path)

    # Calculate center of map
    center = [
        (map_layer_bounds[0][0] + map_layer_bounds[1][0]) / 2,
        (map_layer_bounds[0][1] + map_layer_bounds[1][1]) / 2,
    ]

    folium_map = folium.Map(
        location=center,
        tiles="Stamen Terrain",
        zoom_start=13,
    )

    create_overlay(raster_path, folium_map, names[raster_path.stem])
    folium.LayerControl().add_to(folium_map)
    return folium_map



"""
Main function.
"""
def main():

    st.title("Urban Heat Island Detector - Map examples")

    st.write(
        (
            "These maps show values calculated from Landsat satellite"
            "imagery for different locations."
        )
    )

    data_ids = {
        "Japan": "LC08_L1TP_108035_20241227_20250103_02_T1",
        "Bhutan": "LC09_L1TP_137041_20241214_20241214_02_T1",
    }

    location = st.radio(
        "Select a location",
        data_ids.keys(),
        horizontal=True,
    )

    dir_data_root = Path("data")
    dir_raster = dir_data_root / "raster_files"
    file_lst = Path(dir_raster, data_ids[location] + "_lst_repr_colored.png")
    file_ndvi = Path(dir_raster, data_ids[location] + "_ndvi_repr_colored.png")
    file_emissivity = Path(
        dir_raster, data_ids[location] + "_emissivity_repr_colored.png"
    )

    names = {
        file_lst.stem: "Land Surface Temperature",
        file_ndvi.stem: "Normalized Difference Vegetation Index",
        file_emissivity.stem: "Emissivity",
    }

    map_selected = st.radio(
        "Select a map to show",
        (
            "Land Surface temperature",
            "Vegetation Index (NDVI)",
            "Emissivity",
        ),
        horizontal=True,
    )

    if map_selected == "Land Surface temperature":
        folium_map = create_map(file_lst, names)

    if map_selected == "Vegetation Index (NDVI)":
        folium_map = create_map(file_ndvi, names)

    if map_selected == "Emissivity":
        folium_map = create_map(file_emissivity, names)

    st_folium(folium_map, width=1200, height=1200 / 16 * 9)


if __name__ == "__main__":
    main()