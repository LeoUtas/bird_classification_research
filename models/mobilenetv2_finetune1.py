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


# _______ MOBILENET TEST1: NO DATA AUGMENTATION & NO RE-TRAIN LAYER & CALLBACKS _______ #
factor = 0.7
patience = 5


model_base = tf.keras.applications.mobilenet_v2.MobileNetV2(
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

mobilenetv2_test1_train_accuracy = history.history["accuracy"]
mobilenetv2_test1_val_accuracy = history.history["val_accuracy"]
mobilenetv2_test1_train_loss = history.history["loss"]
mobilenetv2_test1_val_loss = history.history["val_loss"]
mobilenetv2_test1_test_loss, mobilenetv2_test1_test_accuracy = model_0.evaluate(
    test_data
)

end_time = time()
execution_time_test1 = end_time - start_time

logging.info("mobilenet test1 was done")


# _______ MOBILENET TEST4: NO DATA AUGMENTATION & NO RE-TRAIN LAYER & USE CALLBACKS _______ #
factor = 0.8
patience = 5

model_base = tf.keras.applications.mobilenet_v2.MobileNetV2(
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

mobilenetv2_test4_train_accuracy = history.history["accuracy"]
mobilenetv2_test4_val_accuracy = history.history["val_accuracy"]
mobilenetv2_test4_train_loss = history.history["loss"]
mobilenetv2_test4_val_loss = history.history["val_loss"]
mobilenetv2_test4_test_loss, mobilenetv2_test4_test_accuracy = model_0.evaluate(
    test_data
)

end_time = time()
execution_time_test4 = end_time - start_time

logging.info("mobilenet test4 was done")

# # ________________ GET THE DATA READY WITH DATA AUGMENTATION ________________ #
# train_path, test_path, val_path = make_data_path(root_name=parent_path)
# DataGenerator = ImageDataGenerator(
#     rescale=1.0 / 255,
#     horizontal_flip=True,
#     rotation_range=10,  # degrees
#     zoom_range=0.1,
#     # brightness_range=(0.9, 1.1),
# )
# train_data, test_data, val_data = make_data_ready(
#     DataGenerator, train_path, test_path, val_path
# )

# logging.info(
#     "the preparation of train_data, test_data, val_data with augmentation was done"
# )


# # _______ MOBILENET TEST3: DATA AUGMENTATION & NO RE-TRAIN LAYER & CALLBACKS _______ #
# model_base = tf.keras.applications.mobilenet_v2.MobileNetV2(
#     include_top=False, weights=weights
# )
# model_base.trainable = False
# inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input-layer")
# X = model_base(inputs)
# X = tf.keras.layers.GlobalAveragePooling2D(name="global_max_pooling_layer")(X)
# outputs = tf.keras.layers.Dense(524, activation="softmax", name="output-layer")(X)
# model_0 = tf.keras.Model(inputs, outputs)

# # recompile the model to apply fine tuning changes
# model_0.compile(
#     loss="categorical_crossentropy",
#     optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.1),
#     metrics=["accuracy"],
# )

# # add callbacks mechanism to the model fitting
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(
#         monitor="val_accuracy", patience=5, restore_best_weights=True
#     ),
#     tf.keras.callbacks.ReduceLROnPlateau(
#         monitor="val_loss", factor=0.7, patience=2, verbose=1
#     ),
# ]

# start_time = time()

# # refit model and record metrics
# history = model_0.fit(
#     train_data,
#     epochs=epochs,
#     steps_per_epoch=len(train_data),
#     validation_data=val_data,
#     validation_steps=int(0.25 * len(val_data)),
#     verbose=1,
#     callbacks=callbacks,
# )

# mobilenetv2_test3_train_accuracy = history.history["accuracy"]
# mobilenetv2_test3_val_accuracy = history.history["val_accuracy"]
# mobilenetv2_test3_train_loss = history.history["loss"]
# mobilenetv2_test3_val_loss = history.history["val_loss"]
# mobilenetv2_test3_test_loss, mobilenet_test3_test_accuracy = model_0.evaluate(test_data)

# end_time = time()
# execution_time_test3 = end_time - start_time

# logging.info("mobilenet test3 was done")


# _______ RECORD ALL RESULTS _______ #
metrics = {
    "MobileNetV2_test1": {
        "train_accuracy": mobilenetv2_test1_train_accuracy,
        "val_accuracy": mobilenetv2_test1_val_accuracy,
        "train_loss": mobilenetv2_test1_train_loss,
        "val_loss": mobilenetv2_test1_val_loss,
        "test_accuracy": mobilenetv2_test1_test_accuracy,
        "test_loss": mobilenetv2_test1_test_loss,
        "execution_time": execution_time_test1,
    },
    # "MobileNetV2_test3": {
    #     "train_accuracy": mobilenet_test3_train_accuracy,
    #     "val_accuracy": mobilenet_test3_val_accuracy,
    #     "train_loss": mobilenet_test3_train_loss,
    #     "val_loss": mobilenet_test3_val_loss,
    #     "test_accuracy": mobilenet_test3_test_accuracy,
    #     "test_loss": mobilenet_test3_test_loss,
    #     "execution_time": execution_time_test3,
    # },
    "MobileNetV2_test4": {
        "train_accuracy": mobilenetv2_test4_train_accuracy,
        "val_accuracy": mobilenetv2_test4_val_accuracy,
        "train_loss": mobilenetv2_test4_train_loss,
        "val_loss": mobilenetv2_test4_val_loss,
        "test_accuracy": mobilenetv2_test4_test_accuracy,
        "test_loss": mobilenetv2_test4_test_loss,
        "execution_time": execution_time_test4,
    },
}


# ________________ SAVE MODEL PERFORMANCE LOG TO .JSON ________________ #
test_name = "mobilenetv2_finetune1"
file_path = "/home/ubuntu/Projects/Bird_classification/Research/output/data"
file_name = "model_performance_log_" + test_name + ".json"
save_metric(metrics, file_name, file_path)


# ________________ SAVE MODEL PERFORMANCE PLOT TO .jpg ________________ #
nrows = len(metrics)
ncols = 2
width = 10
height = nrows * width / ncols  # to shape each plot element in square shape

figure = visualize_metric(metrics, nrows=nrows, ncols=ncols, figsize=(width, height))

file_path = "/home/ubuntu/Projects/Bird_classification/Research/output/viz"
file_name = "model_performance_plot_" + test_name + ".jpg"
save_plot(figure, file_name, file_path)

notify_training_completion(test_name)
