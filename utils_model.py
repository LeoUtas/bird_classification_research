import tensorflow as tf
import matplotlib.pyplot as plt
import json, os, sys, boto3, subprocess
from exception import CustomException
import numpy as np


# ________________ CONFIG THE BASE MODEL ________________ #
def configure_model_base(
    model_func,
    weights,
    include_top,
    base_trainable,
    input_shape,
    pooling,
    num_class,
    learning_rate,
):
    """

    This function configures and compiles a base model for training and evaluation based on the given parameters.

    Parameters:
    - model_func (function): A function to call and initialize the base model. Examples: ResNet50, VGG16, etc.
    - weights (str or None): One of `None` (random initialization), or a path to pre-trained weights.
    - include_top (bool): Whether to include the top fully connected layer.
    - base_trainable (bool): If `True`, the base model weights will be trainable; if `False`, they will be frozen.
    - input_shape (tuple): Specifies the shape of the input tensor (e.g., (224, 224, 3) for a color image).
    - pooling (str): The pooling type. Should be one of 'max' for global max pooling or 'avg' for global average pooling.
    - num_class (int): The number of output classes for the model.
    - learning_rate (float): The learning rate for the Adam optimizer.

    Returns:
    - model_0 (tf.keras.Model): The configured and compiled Keras model.

    """

    try:
        base_model = model_func(
            weights=weights,
            include_top=include_top,
        )

        if not base_trainable:
            base_model.trainable = False
        else:
            base_model.trainable = True

        inputs = tf.keras.layers.Input(shape=input_shape, name="input-layer")

        X = base_model(inputs)

        if pooling == "max":
            X = tf.keras.layers.GlobalMaxPooling2D(name="global_max_pooling_layer")(X)
        elif pooling == "avg":
            X = tf.keras.layers.GlobalAveragePooling2D(
                name="global_average_pooling_layer"
            )(X)

        outputs = tf.keras.layers.Dense(
            num_class, activation="softmax", name="output-layer"
        )(X)

        model_0 = tf.keras.Model(inputs, outputs)

        model_0.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["accuracy"],
        )

        return model_0

    except Exception as e:
        raise CustomException(e, sys)


# ________________ SAVE MODEL PERFORMANCE METRICS ________________ #
def save_metric(metrics, file_name, file_path):
    """

    This function saves model performance metrics to a JSON file.

    Parameters:
    - metrics (dict): A dictionary containing the metrics data of tested models.
    - file_name (str): The name of the output JSON file where metrics will be saved.
    - file_path (str): The path to the directory where the JSON file should be saved.

    Note:
    If the directory specified in file_path does not exist, it will be created.

    """

    try:
        # ensure the directory structure exists
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        full_path = os.path.join(file_path, file_name)

        # serialize and save the dictionary
        with open(full_path, "w") as f:
            json.dump(metrics, f, indent=4)

    except Exception as e:
        raise CustomException(e, sys)


# ________________ LOAD A JSON FILE ________________ #
def load_dict_from_json(file_name, file_path):
    """

    This function loads a dictionary from a JSON file given its name and path.

    Parameters:
    - file_name (str): The name of the JSON file to be loaded.
    - file_path (str): The path to the directory where the JSON file is located.

    Returns:
    - data (dict): The dictionary loaded from the JSON file.

    """

    try:
        full_path = os.path.join(file_path, file_name)

        with open(full_path, "r") as f:
            data = json.load(f)
        return data

    except Exception as e:
        raise CustomException(e, sys)


# ________________ VISUALIZE THE METRICS ________________ #
def visualize_metric(
    metrics,
    nrows=4,
    ncols=2,
    figsize=None,
    fontsize=16,
    train_acc_color="blue",
    val_acc_color="red",
    train_loss_color="blue",
    val_loss_color="red",
    test_acc_color="darkgreen",
    test_acc_fontsize=10,
):
    """

    This function visualizes the accuracy and loss metrics of tested models.

    Parameters:
    - metrics (dict): A dictionary containing the metrics data of tested models.
                      Expected keys for each model: 'train_accuracy', 'val_accuracy',
                      'test_accuracy', 'train_loss', 'val_loss', and 'execution_time'.
                      
    - nrows (int, optional, default=4): Number of rows for the subplots grid.
    - ncols (int, optional, default=2): Number of columns for the subplots grid.
    - figsize (tuple, optional): Figure size. Default is calculated based on the number of metrics.
    - fontsize (int, optional, default=16): Font size for the super title of the figure.
    - train_acc_color (str, optional, default="blue"): Color for the training accuracy plot.
    - val_acc_color (str, optional, default="red"): Color for the validation accuracy plot.
    - train_loss_color (str, optional, default="blue"): Color for the training loss plot.
    - val_loss_color (str, optional, default="red"): Color for the validation loss plot.
    - test_acc_color (str, optional, default="darkgreen"): Color for the test accuracy text displayed within the plot.
    - test_acc_fontsize (int, optional, default=10): Font size for the test accuracy text.

    Returns:
    - fig (matplotlib.figure.Figure): The created figure object.

    """

    try:
        if figsize is None:
            figsize = (20, 5 * len(metrics))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, squeeze=False
        )  # ensure axes is always a 2D array
        fig.suptitle("Model Performance Metrics", fontsize=fontsize * 1.2, y=1.05)

        model_names = list(metrics.keys())

        for i, model_name in enumerate(model_names):
            # Adjusted indexing for 2D axes array
            ax_idx = (i, 0)  # Accuracy will always be on the left

            # PLOT ACCURACY
            ax = axes[ax_idx[0]][0]

            ax.plot(
                metrics[model_name]["train_accuracy"],
                label="Train Accuracy",
                color=train_acc_color,
            )
            ax.plot(
                metrics[model_name]["val_accuracy"],
                label="Validation Accuracy",
                color=val_acc_color,
            )

            test_accuracy = metrics[model_name]["test_accuracy"]
            ax.text(
                0.4,
                0.15,
                f"Test Accuracy: {test_accuracy:.2%}",
                transform=ax.transAxes,
                color=test_acc_color,
                fontsize=test_acc_fontsize,
                ha="left",
            )

            execution_time = metrics[model_name]["execution_time"]
            ax.text(
                0.4,
                0.1,
                f"Training time: {execution_time/60:.2} minutes",
                transform=ax.transAxes,
                color=test_acc_color,
                fontsize=test_acc_fontsize,
                ha="left",
            )

            ax.set_title(f"{model_name} - Accuracy")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Accuracy")
            ax.legend(loc="upper left")

            # PLOT LOSS
            ax = axes[ax_idx[0]][1]

            ax.plot(
                metrics[model_name]["train_loss"],
                label="Train Loss",
                color=train_loss_color,
            )
            ax.plot(
                metrics[model_name]["val_loss"],
                label="Validation Loss",
                color=val_loss_color,
            )

            ax.set_title(f"{model_name} - Loss")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.legend(loc="upper left")

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # make space for the super title

        return fig

    except Exception as e:
        raise CustomException(e, sys)


# ________________ SAVE THE PLOTS ________________ #
def save_plot(fig, file_name, file_path):
    """

    This function is to save the given figure to a .jpg file.

    Parameters:
    - fig: The figure object to be saved.
    - file_name (str, optional): The name of the output file".
    - file_path (str, optional): The path where the output file should be saved".

    """

    try:
        # ensure the directory structure exists
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        full_path = os.path.join(file_path, file_name)

        fig.savefig(full_path, dpi=300, bbox_inches="tight", format="jpg")

    except Exception as e:
        raise CustomException(e, sys)


# ________________ SAVE TRAINED MODELS ________________ #
def save_model(model, model_name, parent_path):
    """

    This function saves trained model/s to a .keras file/s.

    Parameters:
    - model: The trained model instance to be saved.
    - model_name (str): The desired name for the saved model, without the .keras extension.
    - parent_path (str): The directory where the 'output/save_models' folder is located.

    """

    try:
        save_model_path = os.path.join(
            parent_path + "/output/save_models", model_name + ".keras"
        )

        # os.makedirs(save_model_path, exist_ok=True)  # to make sure directory exists
        model.save(save_model_path)

    except Exception as e:
        raise CustomException(e, sys)


# ________________ SEND NOTIFICATION WHEN TRAINING IS DONE ________________ #
def notify_training_completion(test_name):
    """
    This function sends a notification to aspecified AWS SNS topic when training is complete.

    Parameters:
    None

    Returns:
    None

    """

    client = boto3.client(
        "sns", region_name="ca-central-1"
    )  # replace 'your-region' with your AWS region, e.g., 'us-west-1'
    topic_arn = "arn:aws:sns:ca-central-1:415937355922:training_notification"  # replace with your SNS topic ARN

    # Publish a message
    client.publish(
        TopicArn=topic_arn,
        Message=f"Mate, {test_name} has been completed.",
        Subject=f"EC2 {test_name} done",
    )


# ________________ HANDY FUNCTION TO GET THE FILE NAME ________________ #
def get_name():
    """

    This function gets the name of the current file and remove the extention part of the name

    Parameters:
    None

    Returns:
    The file name without the extension

    """

    file_name = os.path.basename(__file__)
    file_name, _ = os.path.splitext(file_name)
    return file_name


# ______ SEARCH FOR BEST | WORST CASES AMONGST CORRECT PREDICTIONS ______ #
def find_best_worst_cases_in_correct_predictions(chosen_model, test_data, number_cases):
    """

    This function is to find the worst cases in correct predictions (i.e., the least condident level amongst correct predictions)

    Parameters:
    chosen_model: take in the chosen model for the test
    test_data: take in the test data for prediction
    number_cases: take in the expected number of cases to track
    best_or_worst: bool take in "best" or "worst to determine what cases to find

    Returns:
    either worst_cases_info | best_cases_info

    worst_cases_info: an array containing the batch number, the image number within the batch and probability of the worst cases in correct predictions

    best_cases_info: an array containing the batch number, the image number within the batch and probability of the best cases in correct predictions

    """

    try:
        # ________________ MAKE PREDICTION ON TEST DATA ________________ #
        # initialize lists to store predictions and labels
        predictions = []  # to store predictions
        true_labels = []  # to store true labels

        for i in range(len(test_data)):
            batch_index = [i]
            print(f"The batch index is {batch_index}")

            # iterate through the current batch
            for j in range(len(test_data[i][0])):
                print()
                print(f"The image is of the image {j} in batch {i}")
                print(f"The shape of the image is {test_data[i][0][j].shape}")

                input_image = np.expand_dims(test_data[i][0][j], axis=0)

                image_prediction = chosen_model.predict(input_image)
                print(f"The prediction is of the image {j} in batch {i}")

                print(" ----------- END TEST OF ONE IMAGE ----------- ")
                print()

                # append the prediction and true label
                predictions.append(image_prediction)
                true_labels.append(np.argmax(test_data[i][1][j]))

        # convert predictions and true labels to NumPy arrays for easier manipulation
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

    except Exception as e:
        raise CustomException(e, sys)

    try:
        # ________________ GROUP CORRECT AND INCORRECT PREDICTIONS ________________ #
        # initialize lists to store correct and incorrect predictions
        correct_predictions = []  # to store correct predictions
        incorrect_predictions = []  # to store incorrect predictions

        for i in range(len(true_labels)):
            true_label = true_labels[i]
            predicted_probabilities = predictions[i]

            # find the index of the one with the highest probability (i.e., to map with predicted label in prediction process)
            predicted_label = np.argmax(predicted_probabilities)

            if true_label == predicted_label:  # i.e., the prediction is correct
                correct_predictions.append((predicted_probabilities, true_label))
            else:  # the prediction is incorrect
                incorrect_predictions.append((predicted_probabilities, true_label))

    except Exception as e:
        raise CustomException(e, sys)

    try:
        # ______ SEARCH FOR THE BEST OF ALL AND THE WORST AMONGST CORRECT PREDICTIONS ______ #
        # initialize variables to keep track of the worst probabilities and their indices in correct_predictions
        number_worst_cases = number_cases  # number of worst cases to track
        worst_probabilities = [np.inf] * number_worst_cases
        worst_probability_indices = [-1] * number_worst_cases

        for i, (predicted_probabilities, _) in enumerate(correct_predictions):
            best_probability = np.max(predicted_probabilities)  # best probability

            # check if this is one of the worst probabilities among the best probabilities
            for j in range(number_worst_cases):
                if best_probability < worst_probabilities[j]:
                    worst_probabilities[j] = best_probability
                    worst_probability_indices[j] = i
                    break

        # calculate the batch and image indices for the worst cases
        worst_cases_info = []

        for worst_probability_index in worst_probability_indices:
            if worst_probability_index != -1:
                batch_index = worst_probability_index // len(
                    test_data[0][0]
                )  # calculate the batch index
                image_index_within_batch = worst_probability_index % len(
                    test_data[0][0]
                )  # calculate the image index within the batch
                worst_cases_info.append(
                    (
                        worst_probabilities[
                            worst_probability_indices.index(worst_probability_index)
                        ],
                        batch_index,
                        image_index_within_batch,
                    )
                )

        # sort the worst cases by probability in ascending order
        worst_cases_info.sort(key=lambda x: x[0])

        # print information about the worst cases, including their worst probabilities, batch indices, and image indices within the batch
        for idx, (
            worst_probability,
            batch_index,
            image_index_within_batch,
        ) in enumerate(worst_cases_info):
            print()
            print(
                f"Worst probability {idx + 1} in correct prediction:",
                worst_probability,
            )
            print(f"Batch index of worst probability {idx + 1}:", batch_index)
            print(f"Image index within its batch:", image_index_within_batch)
            print()

        # ______ SEARCH FOR THE BEST OF ALL AND THE WORST AMONGST CORRECT PREDICTIONS ______ #
        # initialize variables to keep track of the best probabilities and their indices in correct_predictions
        number_best_cases = number_cases  # number of best cases to track
        best_probabilities = [-np.inf] * number_best_cases
        best_probability_indices = [-1] * number_best_cases

        # Iterate through the correct predictions and track the best probabilities
        for i, (predicted_probabilities, _) in enumerate(correct_predictions):
            best_probability = np.max(predicted_probabilities)  # best probability

            # Check if this is one of the best probabilities among the correct predictions
            for j in range(number_best_cases):
                if best_probability > best_probabilities[j]:
                    best_probabilities.insert(j, best_probability)
                    best_probability_indices.insert(j, i)
                    best_probabilities.pop()
                    best_probability_indices.pop()
                    break

        # calculate the batch and image indices for the best cases
        best_cases_info = []

        for best_probability_index in best_probability_indices:
            if best_probability_index != -1:
                batch_index = best_probability_index // len(
                    test_data[0][0]
                )  # calculate the batch index
                image_index_within_batch = best_probability_index % len(
                    test_data[0][0]
                )  # calculate the image index within the batch
                best_cases_info.append(
                    (
                        best_probabilities[
                            best_probability_indices.index(best_probability_index)
                        ],
                        batch_index,
                        image_index_within_batch,
                    )
                )

        # sort the best cases by probability in descending order
        best_cases_info.sort(reverse=True, key=lambda x: x[0])

        # print information about the 9 best cases, including their best probabilities, batch indices, and image indices within the batch
        for idx, (
            best_probability,
            batch_index,
            image_index_within_batch,
        ) in enumerate(best_cases_info):
            print(
                f"Best probability {idx + 1} in correct prediction:",
                best_probability,
            )
            print(f"Batch index of best probability {idx + 1}:", batch_index)
            print(f"Image index within its batch:", image_index_within_batch)
            print()

        return best_cases_info, worst_cases_info

    except Exception as e:
        raise CustomException(e, sys)


# ________________ VISUALIZE THE CASES ________________ #
def visualize_found_cases(
    cases_info, test_data, plot_title, figsize=(12, 12), nrows=4, ncols=4, fontsize=12
):
    """

    This function is to visualize the images found in find_best_cases_in_correct_predictions() and find_worst_cases_in_correct_predictions()

    """
    try:
        # create subplots
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, squeeze=False
        )  # ensure axes is always a 2D array
        fig.suptitle(plot_title, fontsize=fontsize * 1.2, y=1.05)

        # iterate through the best cases and display them in subplots
        for idx, (best_probability, batch_index, image_index_within_batch) in enumerate(
            cases_info
        ):
            row = idx // ncols  # calculate the row index
            col = idx % ncols  # calculate the column index

            ax = axes[row, col]  # Get the current subplot

            # display the image
            ax.imshow(test_data[batch_index][0][image_index_within_batch])
            ax.set_title(
                f"Batch {batch_index}, Image {image_index_within_batch}\nProbability: {best_probability:.9f}\n"
            )
            ax.axis("off")  # turn off axes

        # ensure tight layout and show the plot
        plt.tight_layout()
        # plt.show()
        return fig

    except Exception as e:
        raise CustomException(e, sys)
