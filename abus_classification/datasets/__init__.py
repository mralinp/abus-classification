import os
from .tumors import Tumors
from .abus import ABUS
from .tdsc import TDSC

path_to_data_directory = "./datasets"

if not os.path.exists(path_to_data_directory):
    os.makedirs(path_to_data_directory)
    print(f"created {path_to_data_directory} directory")
else:
    print("directory exists")

