import sys, os

# ________________ HANDLE THE PATH THING ________________ #
# get the absolute path of the script's directory
script_path = os.path.dirname(os.path.abspath(__file__))
# get the parent directory of the script's directory
parent_path = os.path.dirname(script_path)
sys.path.append(parent_path)


from utils_data import *
from utils_model import *
from logger import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from time import time


# ________________ GET THE DATA READY ________________ #
train_path, test_path, val_path = make_data_path(root_name=parent_path)
DataGenerator = ImageDataGenerator(rescale=1.0 / 255)
train_data, test_data, val_data = make_data_ready(
    DataGenerator, train_path, test_path, val_path
)

logging.info("the preparation of train_data, test_data, val_data was done")

# ________________ CONFIG THE BASE MODELS ________________ #
weights = "imagenet"
input_shape = (224, 224, 3)
num_class = 524

include_top = False
base_trainable = False
pooling = "max"
learning_rate = 1
epochs = 1

model_funcs = [
    tf.keras.applications.inception_v3.InceptionV3,
    tf.keras.applications.resnet50.ResNet50,
    tf.keras.applications.mobilenet.MobileNet,
    tf.keras.applications.mobilenet_v2.MobileNetV2,
    tf.keras.applications.efficientnet.EfficientNetB0,
    tf.keras.applications.efficientnet_v2.EfficientNetV2B0,
]

model_0s = {}

for model_func in model_funcs:
    model_name = model_func.__name__

    model_0s[model_name] = configure_model_base(
        model_func=model_func,
        weights=weights,
        include_top=include_top,
        base_trainable=base_trainable,
        input_shape=input_shape,
        pooling=pooling,
        num_class=num_class,
        learning_rate=learning_rate,
    )

    logging.info(f"the configuration of : {model_name} was done")


history = {}
metrics = {}


# ________________ TEST TRAIN THE BASE MODELS ________________ #
for model_name, model_0 in model_0s.items():
    # train the models

    start_time = time()

    history[model_name] = model_0.fit(
        train_data,
        epochs=epochs,
        steps_per_epoch=len(train_data),
        validation_data=val_data,
        validation_steps=int(0.25 * len(val_data)),
    )

    end_time = time()
    execution_time = end_time - start_time

    # save metrics for the models
    metrics[model_name] = {
        "train_loss": history[model_name].history["loss"],
        "train_accuracy": history[model_name].history["accuracy"],
        "val_loss": history[model_name].history["val_loss"],
        "val_accuracy": history[model_name].history["val_accuracy"],
        "execution_time": execution_time,
    }

    # evaluate the model on test data and save the results
    test_loss, test_accuracy = model_0.evaluate(test_data)
    metrics[model_name]["test_loss"] = test_loss
    metrics[model_name]["test_accuracy"] = test_accuracy

    logging.info(f"training: {model_name} was done in {execution_time/3600} hours")


# ________________ SAVE MODEL PERFORMANCE LOG TO .JSON ________________ #
test_name = "model_base_test0"
file_path = "/home/ubuntu/Projects/Bird_classification/Research/output/data"
file_name = "model_performance_log_" + test_name + ".json"
save_metric(metrics, file_name, file_path)


# ________________ SAVE MODEL PERFORMANCE PLOT TO .jpg ________________ #
nrows = len(model_funcs)
ncols = 2
width = 10
height = nrows * width / ncols  # to shape each plot element in square shape

figure = visualize_metric(metrics, nrows=nrows, ncols=ncols, figsize=(width, height))
file_path = "/home/ubuntu/Projects/Bird_classification/Research/output/viz"
file_name = "model_performance_plot_" + test_name + ".jpg"
save_plot(figure, file_name, file_path)


notify_training_completion(test_name)
