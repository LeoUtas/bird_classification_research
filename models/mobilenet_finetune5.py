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
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
from time import time


# ________________ GET THE DATA READY WITH NO DATA AUGMENTATION ________________ #
train_path, test_path, val_path = make_data_path(root_name=parent_path)
DataGenerator = ImageDataGenerator(rescale=1.0 / 255)
train_data, test_data, val_data = make_data_ready(
    DataGenerator, train_path, test_path, val_path
)

logging.info("the preparation of train_data, test_data, val_data was done")


# ________________ CONFIG THE BASE MODELS ________________ #
learning_rate = 0.001
epochs = 35
patience = 5
factor = 0.6


# _______ MOBILENET TEST14: RE-TRAIN THE LAST 15 LAYERS _______ #
start_time = time()

model_base = tf.keras.applications.mobilenet.MobileNet(
    include_top=False, weights="imagenet"
)
model_base.trainable = False

# assign the last 5 layers to be trainable
last_15_layers = model_base.layers[-15:]
for layer in last_15_layers:
    layer.trainable = True

inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input-layer")
X = model_base(inputs)
X = tf.keras.layers.GlobalAveragePooling2D(name="global_max_pooling_layer")(X)
outputs = tf.keras.layers.Dense(524, activation="softmax", name="output-layer")(X)
model_0 = tf.keras.Model(inputs, outputs)


# recompile the model to apply fine tuning changes
model_0.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.1),
    metrics=["accuracy"],
)

# add callbacks mechanism to the model fitting
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=patience, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=factor, patience=2, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "save_best_models/best_model_mobilenet_test14.h5",  # path to save the best model
        monitor="val_accuracy",
        save_best_only=True,  # only save the best model based on `monitor`
        verbose=1,  # print a message whenever a new best model is saved
        mode="max",  # since we're monitoring accuracy, we want to maximize it
    ),
]

# refit model and record metrics
history = model_0.fit(
    train_data,
    epochs=epochs,
    steps_per_epoch=len(train_data),
    validation_data=val_data,
    validation_steps=int(0.25 * len(val_data)),
    verbose=1,
    callbacks=callbacks,
)

mobilenet_test14_train_accuracy = history.history["accuracy"]
mobilenet_test14_val_accuracy = history.history["val_accuracy"]
mobilenet_test14_train_loss = history.history["loss"]
mobilenet_test14_val_loss = history.history["val_loss"]

# load the saved best model
best_model_0 = tf.keras.models.load_model(
    "save_best_models/best_model_mobilenet_test14.h5"
)

# evaluate the test data using the loaded best model
mobilenet_test14_test_loss, mobilenet_test14_test_accuracy = best_model_0.evaluate(
    test_data
)

end_time = time()
execution_time_test14 = end_time - start_time

logging.info("mobilenet test14 was done")
# --------------------------------------------------------------------------------------- #


# _______ MOBILENET TEST15: RE-TRAIN THE LAST 10 LAYERS _______ #
start_time = time()

model_base = tf.keras.applications.mobilenet.MobileNet(
    include_top=False, weights="imagenet"
)
model_base.trainable = False

# assign the last 10 layers to be trainable
last_10_layers = model_base.layers[-10:]
for layer in last_10_layers:
    layer.trainable = True

inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input-layer")
X = model_base(inputs)
X = tf.keras.layers.GlobalAveragePooling2D(name="global_max_pooling_layer")(X)
outputs = tf.keras.layers.Dense(524, activation="softmax", name="output-layer")(X)
model_0 = tf.keras.Model(inputs, outputs)


# recompile the model to apply fine tuning changes
model_0.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.1),
    metrics=["accuracy"],
)

# add callbacks mechanism to the model fitting
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=patience, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=factor, patience=2, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "save_best_models/best_model_mobilenet_test15.h5",  # path to save the best model
        monitor="val_accuracy",
        save_best_only=True,  # only save the best model based on `monitor`
        verbose=1,  # print a message whenever a new best model is saved
        mode="max",  # since we're monitoring accuracy, we want to maximize it
    ),
]

# refit model and record metrics
history = model_0.fit(
    train_data,
    epochs=epochs,
    steps_per_epoch=len(train_data),
    validation_data=val_data,
    validation_steps=int(0.25 * len(val_data)),
    verbose=1,
    callbacks=callbacks,
)

mobilenet_test15_train_accuracy = history.history["accuracy"]
mobilenet_test15_val_accuracy = history.history["val_accuracy"]
mobilenet_test15_train_loss = history.history["loss"]
mobilenet_test15_val_loss = history.history["val_loss"]

# load the saved best model
best_model_0 = tf.keras.models.load_model(
    "save_best_models/best_model_mobilenet_test15.h5"
)

# evaluate the test data using the loaded best model
mobilenet_test15_test_loss, mobilenet_test15_test_accuracy = best_model_0.evaluate(
    test_data
)

end_time = time()
execution_time_test15 = end_time - start_time

logging.info("mobilenet test15 was done")
# --------------------------------------------------------------------------------------- #

# _______ RECORD ALL RESULTS _______ #
metrics = {
    "MobileNet_test14": {
        "train_accuracy": mobilenet_test14_train_accuracy,
        "val_accuracy": mobilenet_test14_val_accuracy,
        "train_loss": mobilenet_test14_train_loss,
        "val_loss": mobilenet_test14_val_loss,
        "test_accuracy": mobilenet_test14_test_accuracy,
        "test_loss": mobilenet_test14_test_loss,
        "execution_time": execution_time_test14,
    },
    "MobileNet_test15": {
        "train_accuracy": mobilenet_test15_train_accuracy,
        "val_accuracy": mobilenet_test15_val_accuracy,
        "train_loss": mobilenet_test15_train_loss,
        "val_loss": mobilenet_test15_val_loss,
        "test_accuracy": mobilenet_test15_test_accuracy,
        "test_loss": mobilenet_test15_test_loss,
        "execution_time": execution_time_test15,
    },
}


# ________________ SAVE MODEL PERFORMANCE LOG TO .JSON ________________ #
test_name = "mobilenet_finetune5"
# file_path = "/home/ubuntu/Projects/Bird_classification/Research/output/data"
file_path = "/home/hoangng/Projects/Bird_classification/Research/output/data"
file_name = "model_performance_log_" + test_name + ".json"
save_metric(metrics, file_name, file_path)


# ________________ SAVE MODEL PERFORMANCE PLOT TO .jpg ________________ #
nrows = len(metrics)
ncols = 2
width = 10
height = nrows * width / ncols  # to shape each plot element in square shape

figure = visualize_metric(metrics, nrows=nrows, ncols=ncols, figsize=(width, height))

# file_path = "/home/ubuntu/Projects/Bird_classification/Research/output/viz"
file_path = "/home/hoangng/Projects/Bird_classification/Research/output/viz"
file_name = "model_performance_plot_" + test_name + ".jpg"
save_plot(figure, file_name, file_path)

# notify_training_completion(test_name)
