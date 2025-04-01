[![GitHub contributors](https://badgen.net/github/contributors/debasishray16/SatXtract)](https://github.com/debasishray16/SatXtract/graphs/contributors) 
[![Merged Pull Request](https://badgen.net/github/merged-prs/debasishray16/SatXtract)](https://github.com/debasishray16/SatXtract/pulls?q=is%3Apr+is%3Amerged)

# Satellite Image Segmentation for LST, Emissivity, and NDVI Estimation Using U-Net and Interpolation Techniques

### Stack Used:



![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![nVIDIA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![PyPi](https://img.shields.io/badge/pypi-%23ececec.svg?style=for-the-badge&logo=pypi&logoColor=1f73b7)
## About

The **Urban Heat Island (UHI)** effect is a significant environmental issue that refers to the increased temperature in urban areas compared to their rural surroundings. This phenomenon is primarily caused by human activities, changes in land use, and built infrastructure. Dark surfaces, such as certain roofing materials, are known to absorb more solar radiation and contribute to elevated temperatures.

The UHI effect exacerbates the impacts of climate change, leading to increased energy consumption, impaired air quality, and adverse health effects such as heat-related illnesses and mortality.

As heat waves are likely to become more frequent, preventing and mitigating the UHI effect is crucial for sustainable urban development and climate change adaptation.

## Goal of project

- The goal of the project was to use satellite imagery – including infrared bands – to detect and visualize UHI effects in urban areas.

- To accomplish that, the plan was to build and train a Deep Learning model to segment building roofs in urban areas from high-resolution satellite imagery.

- With that data, we could identify spots for public interventions like green roofs, change of roof material/color, or installation of solar panels.

## Dataset

In this project , LANDSAT Satellite data is used. **Landsat 8-9 OLI/TIRS C2 L1**

- The **dataset** I used for training is called **Landcover.ai**.
  - It is **publicly available** on [kaggle.com](https://www.kaggle.com/datasets/adrianboguszewski/landcoverai?resource=download) and consists of **40 high resolution labeled images**.
  - I broke those images down to about **40.000 patches of 256x256px** size.
  - From those, I **selected 2.000 patches with relevant information** for training (because most is woodland, which is not of use in this case).
  - The **metric** I used was **Mean IoU (Intersection over Union)**. It is a number from 0-1 which specifies **amount of overlap between prediction & ground truth**. 0 means _no overlap_, 1 means _complete overlap_ (100%).
  - The **best score** I reached on unseen test data: 0.86 (equivalent to 86% of correctly classified pixels).
- Additionally, I wrote Python scripts to process **Landsat 8 satellite imagery** and create geospatial images for the following metrics:
  - **Normalized Difference Vegetation Index (NDVI):** Calculated from Landsat 8 bands 4 (visible red) and 5 (near-infrared)
  - **Emissivity:** Calculated from Landsat 8 bands 4 (visible red) and 5 (near-infrared)
  - **Land Surface Temperature (LST):** Calculated from Landsat 8 bands 4 (visible red), 5 (near-infrared), and 10 (long wavelength infrared)
  - **Building Footprints:** From Segmentation using the Deep Learning model and extracted from Openstreetmap data
  - **Luminance of Building Footprints:** Calculated from Landsat 8 bands 2 (visible blue), 3 (visible green), and 4 (visible red) using building footprints as masks

## Information

Includes LST(Land Surface Temperature), NDVI (Normalized Difference Vegetation Index) and LSE (Land Surface Emissivity) which are used for various applications such as:

- Monitoring Vegetation Health
- Crop Yield Prediction
- Deforestation Studies
- Drought Assessment
- Land Cover Classification
- Climate Change Research
- Urban Heat Island Studies

| Terms                                         | Meaning                                                                                                                                                                                                                                                   |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LST (Land Surface Temperature)                | Remote sensing refers to the temperature of the Earth's surface as measured by satellite sensors. It is a crucial parameter for studying climate change, urban heat islands, and land-cover changes.                                                      |
| NDVI (Normalized Difference Vegetation Index) | Remote sensing index to measure vegetation health and density. It helps differentiate healthy vegetation from bare soil, water bodies, and urban areas.                                                                                                   |
| LSE (Land Surface Emissivity)                 | Efficiency with which the Earth's surface emits thermal infrared radiation. It is a critical parameter for accurate Land Surface Temperature (LST) estimation, as different land covers (vegetation, soil, water, urban areas) emit radiation differently |

```py
branca==0.6.0
folium==0.14.0
GDAL==3.4.1
geopandas==0.13.0
keras==2.13.1
matplotlib==3.7.1
numpy==1.23.5
opencv_python==4.8.1.78
opencv_python_headless==4.7.0.72
pandas==2.0.2
patchify==0.2.3
Pillow==10.2.0
pylandtemp==0.0.1a1
pyproj==3.5.0
rasterio==1.3.7
scikit_learn==1.2.2
scipy==1.11.3
segment_geospatial==0.8.2
segmentation_models==1.0.1
Shapely==2.0.1
split-folders==0.5.1
streamlit==1.27.2
streamlit_folium==0.12.0
streamlit_image_comparison==0.0.4
tensorflow==2.13.0
tqdm==4.65.0

```

## Model Evaluation

| Epoch | Model Location                                              | Notebook Location                                      |
| ----- | ----------------------------------------------------------- | ------------------------------------------------------ |
| 25    | [Model](/models/landcover_resnet50_25_epochs_batch16.hdf5)  | [File](/notebooks/segmentation_step_2_model_test.ipynb) |
| 50    | [Model](/models/landcover_resnet50_50_epochs_batch16.hdf5)  | [File](/notebooks/segmentation_step_2_model_test.ipynb) |
| 75    | [Model](/models/landcover_resnet50_75_epochs_batch16.hdf5)  | [File](/notebooks/segmentation_step_2_model_U_Net_resnet50_75.ipynb)|
| 100   | [Model](/models/landcover_resnet50_100_epochs_batch16.hdf5) | [File](/notebooks/segmentation_step_2_model_test.ipynb) |

### Information

If you want to add data which is over 100 Mb, then you can use **Git LFS Storage**.

1. First add the specific file in your project file.
2. Then, copy the location of the file which is over 100 MB. Then, istall the git lfs by using command:

```sh
git lfs install
```

3. Identify the extension of the desired file.

```sh
# hdf5 file is over 125 mb
git lfs track "*.hdf5"
git add .gitattributes
```

4. Then, add the file location by using:

```sh
git add models/landcover_resnet50_50_epochs_batch16.hdf5
```

5. Push the file by using the following command from your local git repository.

```sh
git commit -m "Add model .hdf5 file"
git push origin main --force
```




```bash
pip install geemap

# import ee
# import geemap


ee.Authenticate()
ee.Initialize()
```







```bash
/c/OSGeo4W/bin/python.exe -m streamlit run C:/Users/debas/UHI_Prediction/Urban-Heat-Island/app/base.py


/c/OSGeo4W/bin/python.exe -m pip install scipy --no-warn-script-location
```


```bash

conda env config vars set GDAL_DATA="C:\Anaconda\envs\osgeo_env\Library\share\gdal"
conda env config vars set PROJ_LIB="C:\Anaconda\envs\osgeo_env\Library\share\proj"
conda activate osgeo_env
# Check if they are set

echo $GDAL_DATA
echo $PROJ_LIB

```
## Contributors

<div align="center">
 <a href="https://github.com/debasishray16/SatXtract/graphs/contributors">
   <img src="https://contrib.rocks/image?repo=debasishray16/SatXtract" />
 </a>
</div>
