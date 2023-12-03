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


# ________________ GET THE DATA READY WITH NO DATA AUGMENTATION ________________ #
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
learning_rate = 0.001
epochs = 35


# _______ MOBILENET TEST5: NO DATA AUGMENTATION & NO RE-TRAIN LAYER & CALLBACKS _______ #
factor = 0.7
patience = 5


model_base = tf.keras.applications.mobilenet.MobileNet(
    include_top=False, weights=weights
)
model_base.trainable = False
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
        "save_best_models/best_model_mobilenet_test5.h5",  # path to save the best model
        monitor="val_accuracy",
        save_best_only=True,  # only save the best model based on `monitor`
        verbose=1,  # print a message whenever a new best model is saved
        mode="max",  # since we're monitoring accuracy, we want to maximize it
    ),
]

start_time = time()

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

mobilenet_test5_train_accuracy = history.history["accuracy"]
mobilenet_test5_val_accuracy = history.history["val_accuracy"]
mobilenet_test5_train_loss = history.history["loss"]
mobilenet_test5_val_loss = history.history["val_loss"]

# load the saved best model
best_model_0 = tf.keras.models.load_model(
    "save_best_models/best_model_mobilenet_test5.h5"
)

# evaluate the test data using the loaded best model
mobilenet_test5_test_loss, mobilenet_test5_test_accuracy = best_model_0.evaluate(
    test_data
)

end_time = time()
execution_time_test5 = end_time - start_time

logging.info("mobilenet test5 was done")
# --------------------------------------------------------------------------------------- #


# _______ MOBILENET TEST6: NO DATA AUGMENTATION & NO RE-TRAIN LAYER & CALLBACKS _______ #
factor = 0.6
patience = 6


model_base = tf.keras.applications.mobilenet.MobileNet(
    include_top=False, weights=weights
)
model_base.trainable = False
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
        "save_best_models/best_model_mobilenet_test6.h5",  # path to save the best model
        monitor="val_accuracy",
        save_best_only=True,  # only save the best model based on `monitor`
        verbose=1,  # print a message whenever a new best model is saved
        mode="max",  # since we're monitoring accuracy, we want to maximize it
    ),
]

start_time = time()

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

mobilenet_test6_train_accuracy = history.history["accuracy"]
mobilenet_test6_val_accuracy = history.history["val_accuracy"]
mobilenet_test6_train_loss = history.history["loss"]
mobilenet_test6_val_loss = history.history["val_loss"]

# load the saved best model
best_model_0 = tf.keras.models.load_model(
    "save_best_models/best_model_mobilenet_test6.h5"
)

# evaluate the test data using the loaded best model
mobilenet_test6_test_loss, mobilenet_test6_test_accuracy = best_model_0.evaluate(
    test_data
)

end_time = time()
execution_time_test6 = end_time - start_time

logging.info("mobilenet test6 was done")
# --------------------------------------------------------------------------------------- #

# _______ RECORD ALL RESULTS _______ #
metrics = {
    "MobileNet_test5": {
        "train_accuracy": mobilenet_test5_train_accuracy,
        "val_accuracy": mobilenet_test5_val_accuracy,
        "train_loss": mobilenet_test5_train_loss,
        "val_loss": mobilenet_test5_val_loss,
        "test_accuracy": mobilenet_test5_test_accuracy,
        "test_loss": mobilenet_test5_test_loss,
        "execution_time": execution_time_test5,
    },
    "MobileNet_test6": {
        "train_accuracy": mobilenet_test6_train_accuracy,
        "val_accuracy": mobilenet_test6_val_accuracy,
        "train_loss": mobilenet_test6_train_loss,
        "val_loss": mobilenet_test6_val_loss,
        "test_accuracy": mobilenet_test6_test_accuracy,
        "test_loss": mobilenet_test6_test_loss,
        "execution_time": execution_time_test6,
    },
}


# ________________ SAVE MODEL PERFORMANCE LOG TO .JSON ________________ #
test_name = "mobilenet_finetune2"
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
file_path = "/home/hoangng/Projects/Bird_classification/Research/output/data"
file_name = "model_performance_plot_" + test_name + ".jpg"
save_plot(figure, file_name, file_path)

notify_training_completion(test_name)