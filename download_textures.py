"""
This script downloads the PBR textures and Mesh dataset
"""

import os

import gdown
import zipfile


def main():
    # Download the Texture Dataset (45 textures)
    if not os.path.exists("data/textures"):

        texture_dataset_url = (
            "https://drive.google.com/uc?id=1rpk_39XlRivwJ7hmC2Cfmauk6Q9hZEdC"
        )
        gdown.download(texture_dataset_url, "data/textures.zip", quiet=False)
        print("Unzipping the Texture dataset")
        with zipfile.ZipFile("data/textures.zip", "r") as zip_ref:
            zip_ref.extractall("data/textures")
        os.remove("data/textures.zip")
    else:
        print("Texture dataset is already downloaded.")
        print("Remove the existing folder at data/textures to download again.\n")


if __name__ == "__main__":
    main()
