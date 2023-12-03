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
epochs = 15
patience = 5
factor = 0.6


# _______ MOBILENET TEST10: USE DROPOUT _______ #
start_time = time()

# load the saved model
model_0 = tf.keras.models.load_model("save_best_models/best_model_mobilenet_test6.h5")


# find the position of the dense layer
dense_layer_position = None
for i, layer in enumerate(model_0.layers):
    if isinstance(layer, Dense):
        dense_layer_position = i
        break

# if the dense layer is found, reconstruct the model with dropout before the dense layer
if dense_layer_position is not None:
    model_with_dropout = Sequential()
    for i, layer in enumerate(model_0.layers):
        if i == dense_layer_position:
            model_with_dropout.add(Dropout(0.4))  # add a dropout layer .2
        model_with_dropout.add(layer)
else:
    print("Dense layer not found!")
    model_with_dropout = model_0  # use the original model if no dense layer is found

model_0 = model_with_dropout


# recompile the model to apply fine tuning changes
model_0.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.1),
    metrics=["accuracy"],
)


# add callbacks mechanism to the model fitting
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "save_best_models/best_model_mobilenet_test10.h5",  # path to save the best model
        monitor="val_accuracy",
        save_best_only=True,  # only save the best model based on `monitor`
        verbose=1,  # print a message whenever a new best model is saved
        mode="max",  # since we're monitoring accuracy, we want to maximize it
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=patience, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=factor, patience=2, verbose=1
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

mobilenet_test10_train_accuracy = history.history["accuracy"]
mobilenet_test10_val_accuracy = history.history["val_accuracy"]
mobilenet_test10_train_loss = history.history["loss"]
mobilenet_test10_val_loss = history.history["val_loss"]

# load the saved best model
best_model_0 = tf.keras.models.load_model(
    "save_best_models/best_model_mobilenet_test10.h5"
)

# evaluate the test data using the loaded best model
mobilenet_test10_test_loss, mobilenet_test10_test_accuracy = best_model_0.evaluate(
    test_data
)

end_time = time()
execution_time_test10 = end_time - start_time

logging.info("mobilenet test10 was done")
# --------------------------------------------------------------------------------------- #


# _______ MOBILENET TEST11: RE-TRAIN THE LAST 5 LAYERS _______ #
start_time = time()

# load the saved model
model_0 = tf.keras.models.load_model("save_best_models/best_model_mobilenet_test6.h5")

# assign the last 10 layers to be trainable
last_5_layers = model_0.layers[-5:]
for layer in last_5_layers:
    layer.trainable = True

# # create a new model with dropout added before the last 5 layers
# model_0 = tf.keras.models.Sequential(model_0.layers[:-5])
# model_0.add(Dropout(0.2))  # add a dropout layer

# # add the last 5 layers back to the new model
# for layer in last_5_layers:
#     model_0.add(layer)

# recompile the model to apply fine tuning changes
model_0.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.1),
    metrics=["accuracy"],
)

# add callbacks mechanism to the model fitting
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "save_best_models/best_model_mobilenet_test11.h5",  # path to save the best model
        monitor="val_accuracy",
        save_best_only=True,  # only save the best model based on `monitor`
        verbose=1,  # print a message whenever a new best model is saved
        mode="max",  # since we're monitoring accuracy, we want to maximize it
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=patience, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=factor, patience=2, verbose=1
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

mobilenet_test11_train_accuracy = history.history["accuracy"]
mobilenet_test11_val_accuracy = history.history["val_accuracy"]
mobilenet_test11_train_loss = history.history["loss"]
mobilenet_test11_val_loss = history.history["val_loss"]

# load the saved best model
best_model_0 = tf.keras.models.load_model(
    "save_best_models/best_model_mobilenet_test11.h5"
)

# evaluate the test data using the loaded best model
mobilenet_test11_test_loss, mobilenet_test11_test_accuracy = best_model_0.evaluate(
    test_data
)

end_time = time()
execution_time_test11 = end_time - start_time

logging.info("mobilenet test11 was done")
# --------------------------------------------------------------------------------------- #


# _______ MOBILENET TEST12: USE CALLBACKS _______ #
start_time = time()

# load the saved model
model_0 = tf.keras.models.load_model("save_best_models/best_model_mobilenet_test6.h5")

# recompile the model to apply fine tuning changes
model_0.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.1),
    metrics=["accuracy"],
)

# add callbacks mechanism to the model fitting
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "save_best_models/best_model_mobilenet_test12.h5",  # path to save the best model
        monitor="val_accuracy",
        save_best_only=True,  # only save the best model based on `monitor`
        verbose=1,  # print a message whenever a new best model is saved
        mode="max",  # since we're monitoring accuracy, we want to maximize it
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=patience, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=factor, patience=2, verbose=1
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

mobilenet_test12_train_accuracy = history.history["accuracy"]
mobilenet_test12_val_accuracy = history.history["val_accuracy"]
mobilenet_test12_train_loss = history.history["loss"]
mobilenet_test12_val_loss = history.history["val_loss"]

# load the saved best model
best_model_0 = tf.keras.models.load_model(
    "save_best_models/best_model_mobilenet_test12.h5"
)

# evaluate the test data using the loaded best model
mobilenet_test12_test_loss, mobilenet_test12_test_accuracy = best_model_0.evaluate(
    test_data
)

end_time = time()
execution_time_test12 = end_time - start_time

logging.info("mobilenet test12 was done")
# --------------------------------------------------------------------------------------- #


# _______ RECORD ALL RESULTS _______ #
metrics = {
    "MobileNet_test10": {
        "train_accuracy": mobilenet_test10_train_accuracy,
        "val_accuracy": mobilenet_test10_val_accuracy,
        "train_loss": mobilenet_test10_train_loss,
        "val_loss": mobilenet_test10_val_loss,
        "test_accuracy": mobilenet_test10_test_accuracy,
        "test_loss": mobilenet_test10_test_loss,
        "execution_time": execution_time_test10,
    },
    "MobileNet_test11": {
        "train_accuracy": mobilenet_test11_train_accuracy,
        "val_accuracy": mobilenet_test11_val_accuracy,
        "train_loss": mobilenet_test11_train_loss,
        "val_loss": mobilenet_test11_val_loss,
        "test_accuracy": mobilenet_test11_test_accuracy,
        "test_loss": mobilenet_test11_test_loss,
        "execution_time": execution_time_test11,
    },
    "MobileNet_test12": {
        "train_accuracy": mobilenet_test12_train_accuracy,
        "val_accuracy": mobilenet_test12_val_accuracy,
        "train_loss": mobilenet_test12_train_loss,
        "val_loss": mobilenet_test12_val_loss,
        "test_accuracy": mobilenet_test12_test_accuracy,
        "test_loss": mobilenet_test12_test_loss,
        "execution_time": execution_time_test12,
    },
}


# ________________ SAVE MODEL PERFORMANCE LOG TO .JSON ________________ #
test_name = "mobilenet_finetune4"
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
