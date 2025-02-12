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