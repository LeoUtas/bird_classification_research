import sys, os
from time import time

# ________________ HANDLE THE PATH THING ________________ #
# get the absolute path of the script's directory
script_path = os.path.dirname(os.path.abspath(__file__))
# get the parent directory of the script's directory
parent_path = os.path.dirname(script_path)
sys.path.append(parent_path)

from utils_YOLOv8 import *

start_time = time()


# ________________ MAKE MODEL CONFIGURATION ________________ #
# ****** --------- ****** #
test_name = "test1"
note = ""
# ****** --------- ****** #


model_yaml = "yolov8n-cls.yaml"
model_pt = "yolov8n-cls.pt"
# model_yaml = "yolov8x-cls.yaml"
# model_pt = "yolov8x-cls.pt"


data = os.path.join(parent_path, "input", "data")
project = os.path.join(parent_path, "models", "YOLOv8", test_name)

# custom configuration
epochs = 50
patience = 15
batch = 16  # -1 autobatch
image_size = 224
device = 0
workers = 2
pretrained = False
optimizer = "auto"
verbose = True
lr0 = 0.01
weight_decay = 0.0005
dropout = 0.0


model = YOLOv8(
    test_name=test_name,
    model_yaml=model_yaml,
    model_pt=model_pt,
    data=data,
    project=project,
    note=note,
    # custom configuration
    epochs=epochs,
    patience=patience,
    batch=batch,
    image_size=image_size,
    device=device,
    workers=workers,
    pretrained=pretrained,
    optimizer=optimizer,
    verbose=verbose,
    lr0=lr0,
    weight_decay=weight_decay,
    dropout=dropout,
)


# _ TRAIN _ #
model.train()


# # ________________ VALIDATE ________________ #
# path_to_val_model = os.path.join(
#     parent_path, "models", "YOLOv8", test_name, "train", "weights", "last.pt"
# )
# model.validate(path_to_val_model=path_to_val_model)


execution_time = round((time() - start_time) / 60, 2)
print(f"Execution time: {execution_time} mins")


# ________________ VALIDATE ON TEST DATA ________________ #
path_to_chosen_model = os.path.join(
    script_path, "YOLOv8", test_name, "train", "weights", "last.pt"
)
path_to_testdata = os.path.join(parent_path, "input", "data", "test")
path_to_json = os.path.join(parent_path, "input", "data", "class_indices.json")

predictions, test_accuracy = make_evaluation_on_testdata(
    path_to_chosen_model, path_to_testdata, path_to_json
)


# _______ RECORD TEST RESULTS _______ #
metrics = {
    f"YOLOv8_{test_name}": {
        "test_accuracy": test_accuracy,
        "execution_time": execution_time,
    },
}

path_to_save_json = os.path.join(
    parent_path, "models", "YOLOv8", test_name, f"metrics_{test_name}.json"
)
with open(path_to_save_json, "w") as json_file:
    json.dump(metrics, json_file)
