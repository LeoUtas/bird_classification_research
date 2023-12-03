from utils_data import *
from utils_model import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ________________ GET THE DATA READY ________________ #
root_name = os.path.join("Bird_classification", "Research")
train_path, test_path, val_path = make_data_path(root_name=root_name)

DataGenerator = ImageDataGenerator(rescale=1.0 / 255)
train_data, test_data, val_data = make_data_ready(
    DataGenerator, train_path, test_path, val_path
)

# ________________ LOAD THE CHOSEN MODEL ________________ #
chosen_model = tf.keras.models.load_model(
    "models/save_best_models/best_model_mobilenet_test6.h5"
)

number_cases = 16

# ________________ FIND BEST & WORST CASES IN CORRECT PREDICTIONS ________________ #
best_cases_info, worst_cases_info = find_best_worst_cases_in_correct_predictions(
    chosen_model, test_data, number_cases
)

best_cases_viz = visualize_found_cases(
    cases_info=best_cases_info,
    test_data=test_data,
    plot_title="Best cases in correct predictions",
    figsize=(12, 12),
    nrows=4,
    ncols=4,
    fontsize=12,
)

file_name = "best_cases_viz.jpg"
file_path = os.path.join("output", "viz")
save_plot(best_cases_viz, file_name, file_path)


worst_cases_viz = visualize_found_cases(
    worst_cases_info,
    test_data,
    plot_title="Worst cases in correct predictions",
    figsize=(12, 12),
    nrows=4,
    ncols=4,
    fontsize=12,
)

file_name = "worst_cases_viz.jpg"
file_path = os.path.join("output", "viz")
save_plot(worst_cases_viz, file_name, file_path)

if __name__ == "__main__":
    print("all done")
