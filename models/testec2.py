import sys, os
import json

# ________________ HANDLE THE PATH THING ________________ #
# get the absolute path of the script's directory
script_path = os.path.dirname(os.path.abspath(__file__))
# get the parent directory of the script's directory
parent_path = os.path.dirname(script_path)
sys.path.append(parent_path)


from utils_data import *
from utils_model import *

# ________________ SAVE MODEL PERFORMANCE LOG TO .JSON ________________ #
# metrics = "Test is ok"
# test_name = "test_path_on_ec2"
# file_path = "/home/ubuntu/Projects/Bird_classification/Research/output/data"
# # file_path = "/home/hoangng/Projects/Bird_Classification/Research/output/data"
# file_name = "log_" + test_name + ".json"
# save_metric(metrics, file_name, file_path)


test_name = "X"
file_path = "/home/ubuntu/Projects/Bird_classification/Research/output/data"
file_name = "model_performance_log_" + test_name + ".json"

with open(
    "/home/ubuntu/Projects/Bird_classification/Research/output/data/model_performance_log_mobilenetv2_finetune0.json",
    "r",
) as f:
    metrics = json.load(f)

nrows = 2
ncols = 2
width = 10
height = nrows * width / ncols  # to shape each plot element in square shape

figure = visualize_metric(metrics, nrows=nrows, ncols=ncols, figsize=(width, height))

file_path = "/home/ubuntu/Projects/Bird_classification/Research/output/viz"
file_name = "model_performance_plot_" + test_name + ".jpg"
save_plot(figure, file_name, file_path)
