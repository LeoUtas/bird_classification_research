# import required dependencies

import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from exception import CustomException


# ________________ MAKE PATHS ________________ #
def make_data_path(root_name):
    """
    This function constructs and returns the paths for the training, testing, and validation datasets, given a root directory name.

    Parameters:
    - root_name (str): The name of the root directory under which the datasets reside.

    Returns:
    - train_path (pathlib.Path): Path object pointing to the training dataset directory.
    - test_path (pathlib.Path): Path object pointing to the testing dataset directory.
    - val_path (pathlib.Path): Path object pointing to the validation dataset directory.

    Notes:
    - The function assumes a specific directory structure:
      root_name/input/data/{train, test, valid}
    - The paths are constructed relative to the grandparent directory of the current working directory.

    """

    try:
        current_path = os.getcwd()

        train_path = Path(
            os.path.join(
                (os.path.dirname(os.path.dirname(current_path))),
                root_name,
                "input",
                "data",
                "train",
            )
        )

        test_path = Path(
            os.path.join(
                (os.path.dirname(os.path.dirname(current_path))),
                root_name,
                "input",
                "data",
                "test",
            )
        )

        val_path = Path(
            os.path.join(
                (os.path.dirname(os.path.dirname(current_path))),
                root_name,
                "input",
                "data",
                "val",
            )
        )

        return train_path, test_path, val_path

    except Exception as e:
        raise CustomException(e, sys)


# ________________ VISUALIZE IMAGES ________________ #
def visualize_images(path, nrows=4, ncols=4, figsize=(12, 12)):
    """
    This function visualizes randomly selected bird images from a specified directory, displaying both their common and scientific names.

    Parameters:
    - path (Path): A pathlib.Path object pointing to the directory containing bird species folders.
    - nrows (int, optional, default=4): Number of rows for the subplots grid.
    - ncols (int, optional, default=4): Number of columns for the subplots grid.
    - figsize (tuple, optional, default=(12, 12)): Figure size.

    Returns:
    - fig (matplotlib.figure.Figure): The created figure object.

    Notes:
    - Each bird species directory should contain .jpg images of that species.
    - There should be a CSV file named "birds.csv" in the same directory as the `path`.
      This CSV should have columns 'labels' (common bird names) and 'scientific name'
      (scientific names of the birds).

    """

    try:
        # prepare for visualizing the images
        image_files = [
            f for bird_species in path.iterdir() for f in bird_species.glob("*.jpg")
        ]
        image_df = pd.DataFrame(
            {
                "file_path": image_files,
                "Label": [
                    f.parent.name for f in image_files
                ],  # using the parent folder name as label
            }
        )

        # read in the CSV file
        birds_df = pd.read_csv(os.path.join(os.path.dirname(path), "birds.csv"))
        labels_to_scientific_name = dict(
            zip(birds_df["labels"], birds_df["scientific name"])
        )

        # visualize 16 random images
        random_index = np.random.randint(0, len(image_df), nrows * ncols)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            subplot_kw={"xticks": [], "yticks": []},
        )

        for i, ax in enumerate(axes.flat):
            ax.imshow(plt.imread(image_df.file_path.iloc[random_index[i]]))

            # fetch the scientific name using the label and the created dictionary
            label = image_df.Label.iloc[random_index[i]]
            scientific_name = labels_to_scientific_name.get(label, "Unknown")
            ax.set_title(
                f"{label}\n({scientific_name})"
            )  # display both common and scientific names

        plt.tight_layout()
        # plt.show()

        return fig

    except Exception as e:
        raise CustomException(e, sys)


# ________________ PREPARE THE DATA ________________ #
def make_data_ready(DataGenerator, train_path, test_path, val_path):
    """

    This function prepares training, testing, and validation data using the specified data generator.

    Parameters:
    - DataGenerator (tf.keras.preprocessing.image.ImageDataGenerator): The image data generator to use for data augmentation and preprocessing.
    - train_path (str or pathlib.Path): Path to the directory containing training data.
    - test_path (str or pathlib.Path): Path to the directory containing testing data.
    - val_path (str or pathlib.Path): Path to the directory containing validation data.

    Returns:
    - train_data (tf.data.Dataset): The prepared training dataset.
    - test_data (tf.data.Dataset): The prepared testing dataset.
    - valid_data (tf.data.Dataset): The prepared validation dataset.

    Notes:
    - The directories specified by train_path, test_path, and val_path should have sub-directories for each class,
      where the sub-directory name is the class name, and it contains images of that class.
    - The function assumes that images should be resized to 224x224 and batches of 32 images are used.
      The class mode is set to 'categorical' for one-hot encoded labels.

    """

    try:
        train_data = DataGenerator.flow_from_directory(
            train_path, target_size=(224, 224), batch_size=32, class_mode="categorical"
        )

        test_data = DataGenerator.flow_from_directory(
            test_path, target_size=(224, 224), batch_size=32, class_mode="categorical"
        )

        valid_data = DataGenerator.flow_from_directory(
            val_path, target_size=(224, 224), batch_size=32, class_mode="categorical"
        )

        return train_data, test_data, valid_data

    except Exception as e:
        raise CustomException(e, sys)


# ________________ GET CLASSES FROM DATA ________________ #
def get_classes(file_path, file_name):
    """
    This function is to read a CSV file named "birds.csv", then to parse and return a dictionary that maps class IDs to their respective labels and scientific names.

    Parameters:
    - file_path (str): The path where the "birds.csv" file is located.

    Returns:
    - dict: A dictionary where each key is a class ID (int), and each value is another      dictionary containing two key-value pairs:
        - "label" (str): The common name of the bird class.
        - "scientific_name" (str): The scientific name of the bird class.

    """

    try:
        # read in the CSV file
        birds_df = pd.read_csv(os.path.join(file_path, file_name))

        # remove all information related to the label "LOONEY BIRDS"
        birds_df = birds_df[birds_df["labels"] != "LOONEY BIRDS"]

        # get unique labels and sort them
        unique_labels = sorted(birds_df["labels"].unique())

        # create a dictionary to hold index, label, and scientific_name
        class_dict = {
            index: {
                "label": label,
                "scientific_name": birds_df[birds_df["labels"] == label][
                    "scientific name"
                ].iloc[0],
            }
            for index, label in enumerate(unique_labels)
        }

        return class_dict

    except Exception as e:
        raise CustomException(e, sys)
