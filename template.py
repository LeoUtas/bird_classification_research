import os
from pathlib import Path
import logging


# logging the process
logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")


# make a list of files and directories
list_of_files = [
    "input",
    "input/data",
    "input/viz",
    "input/data/real_imgs",
    "output",
    "output/data",
    "output/viz",
    "output/save_models",
    "notebook",
    "notebook/EDA.ipynb",
    "notebook/draft.ipynb",
    "from_ec2",
    "resource/data_org",
    "resource/pretrained_models",
    "resource/common.txt",
    "Temp",
    "logger.py",
    "exception.py",
    "utils_model.py",
    "utils_data.py",
    "requirements.txt",
]


for file_path in list_of_files:
    file_path = Path(file_path)

    # If it's a directory or doesn't have a dot (assuming it's a directory)
    if file_path.is_dir() or "." not in file_path.name:
        if not file_path.exists():
            os.makedirs(file_path, exist_ok=True)
            logging.info(f"Created directory: {file_path}")
        else:
            logging.info(
                f"Directory {file_path} already exists => re-creating ignored."
            )
    else:
        file_dir, file_name = os.path.split(file_path)

        if file_dir != "" and not Path(file_dir).exists():
            os.makedirs(file_dir, exist_ok=True)
            logging.info(f"Created directory: {file_dir} for the file {file_name}")

        # If the file does not exist or its size is 0
        if not file_path.exists() or file_path.stat().st_size == 0:
            with open(file_path, "w") as f:
                pass
            logging.info(f"Created an empty file: {file_name}")
        else:
            logging.info(
                f"File {file_name} already exists and is not empty => re-creating ignored."
            )
