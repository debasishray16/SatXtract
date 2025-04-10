from huggingface_hub import HfApi

api = HfApi()

import os

# file_path = "C:/Users/debas/UHI_Prediction/Urban-Heat-Island/models/landcover_resnet34_25_epochs_batch16_freeze.hdf5"
file_path = "C:/Users/debas/UHI_Prediction/Urban-Heat-Island/app/data/models/landcover_resnet18_100_epochs_batch16_freeze.hdf5"

if os.path.exists(file_path):
    print("✅ File found, proceeding with upload...")
    
    api.upload_file(
        path_in_repo="landcover_resnet18_100_epochs_batch16_freeze.hdf5",
        path_or_fileobj=file_path,
        repo_id="debasishray16/satellite_image_segmentation_ResNet_Models",
        repo_type="model"
    )

else:
    print("❌ File not found! Check the path.")
