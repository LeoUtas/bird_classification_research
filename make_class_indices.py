import os, json, sys

# ________________ HANDLE THE PATH THING ________________ #
# get the absolute path of the script's directory
script_path = os.path.dirname(os.path.abspath(__file__))
# get the parent directory of the script's directory
parent_path = os.path.dirname(script_path)
sys.path.append(parent_path)

from exception import CustomException
from utils_data import *


class Class_indices:
    """

    This is to handle the classes in the bird classification data. There are 524 classes of bird (i.e., labelling their common and scientific names). Each image is listed as a record (i.e., a row) in a CSV file containing the information of class id (i.e., class index), common name, scientific name, etc.

    """

    def __init__(self, csv_file_name, json_file_name):
        self.csv_file_name = csv_file_name
        self.json_file_name = json_file_name

        path_to_save_json = os.path.join(
            script_path, os.path.join("input", "data", self.json_file_name)
        )

        try:
            # check if the class_indices.json already exist
            if os.path.exists(path_to_save_json):
                # if the JSON exist then load class_indices from JSON file
                with open(path_to_save_json, "r") as json_file:
                    # load the class_indices.json with str(keys) by default
                    class_indices_str_keys = json.load(json_file)
                    # convert the str(keys) back to integers
                    self.class_indices = {
                        int(k): v for k, v in class_indices_str_keys.items()
                    }

            else:  # if the classes_indices.json doesn't exist (i.e., first time code run)
                # then, make class_indices from the .csv file and save to .json
                self.class_indices = self.make_class_indices()
                with open(path_to_save_json, "w") as json_file:
                    json.dump(self.class_indices, json_file)

        except Exception as e:
            raise CustomException(e, sys)

    def make_class_indices(self):
        try:
            file_path = os.path.join(script_path, "input", "data")
            class_indices = get_classes(file_path, self.csv_file_name)
            return class_indices

        except Exception as e:
            raise CustomException(e, sys)


# test execute the script
if __name__ == "__main__":
    csv_file_name = "birds.csv"
    json_file_name = "class_indices.json"
    class_indices_handler = Class_indices(csv_file_name, json_file_name)
    class_indices = class_indices_handler.class_indices
