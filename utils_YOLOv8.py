import os, sys, json, cv2, shutil
import pandas as pd
from exception import CustomException
from ultralytics import YOLO
from time import time


# ________________ MAKE MODEL CONFIGURATION ________________ #
class YOLOv8:
    def __init__(
        self, test_name, model_yaml, model_pt, data, project, note="", **kwargs
    ):
        # Default configuration values
        self.config = {
            "test_name": test_name,
            "model_yaml": model_yaml,
            "model_pt": model_pt,
            "data": data,
            "project": project,
            "epochs": 100,
            "patience": 50,
            "batch": 16,
            "imgsz": 640,  # image size
            "device": 0,
            "workers": 8,
            "pretrained": True,
            "optimizer": "auto",
            "verbose": True,
            "lr0": 0.01,
            "weight_decay": 0.0005,
            "dropout": 0.0,
            "note": note,
        }

        # Apply any overrides provided upon initialization
        self.config.update(kwargs)

        # Create the project directory if it doesn't exist
        os.makedirs(self.config["project"], exist_ok=True)

    def update_config(self, **kwargs):
        # Update the configuration with new values
        self.config.update(kwargs)

    def train(self):
        try:
            # Instantiate the model using the provided YAML and PT files
            model = YOLO(self.config["model_yaml"])
            model = YOLO(self.config["model_pt"])

            # Start the training process with the provided configuration
            results = model.train(
                data=self.config["data"],
                project=self.config["project"],
                epochs=self.config["epochs"],
                patience=self.config["patience"],
                batch=self.config["batch"],
                imgsz=self.config["imgsz"],
                device=self.config["device"],
                workers=self.config["workers"],
                pretrained=self.config["pretrained"],
                optimizer=self.config["optimizer"],
                verbose=self.config["verbose"],
                lr0=self.config["lr0"],
                weight_decay=self.config["weight_decay"],
                dropout=self.config["dropout"],
            )

            return results

        except Exception as e:
            raise CustomException(e, sys)

    def validate(self, path_to_val_model):
        try:
            model = YOLO(path_to_val_model)

            metrics = model.val(
                data=self.config["data"],
                project=self.config["project"],
            )

            return metrics

        except Exception as e:
            raise CustomException(e, sys)


# ___________ MAKE PREDICTION FOR IMAGES TO COMPUTE TEST ACCURACY ___________ #
def make_evaluation_on_testdata(path_to_chosen_model, path_to_testdata, path_to_json):
    try:
        start_time = time()

        model = YOLO(path_to_chosen_model)
        # Read the JSON file to get the labels
        with open(path_to_json, "r") as file:
            class_indices = json.load(file)

        # Dictionary to store predictions
        predictions = {}
        correct_count = 0
        total_images = 0

        # Iterate over each label
        for key, value in class_indices.items():
            label = value["label"]
            path_to_bird_class = os.path.join(path_to_testdata, label)
            # print(path_to_bird_class)

            # Check if the subfolder exists
            if os.path.isdir(path_to_bird_class):
                predictions[label] = []

                # Process each image in the bird class folder
                for image in os.listdir(path_to_bird_class):
                    if image.lower().endswith((".jpg", ".jpeg", ".png")):
                        total_images += 1
                        image_path = os.path.join(path_to_bird_class, image)

                        prediction = model.predict(image_path, device="cpu")
                        pred_index = prediction[0].probs.top1
                        pred_conf = prediction[0].probs.top1conf
                        pred_name = prediction[0].names[pred_index]
                        if pred_name == label:
                            correct_count += 1

                        predictions[label].append((image, pred_conf))

        accuracy = correct_count / total_images

        execution_time = time() - start_time
        print(f"Execution time: {round(execution_time/60, 2)} mins")
        print(f"Total number of test images: {total_images}")

        return predictions, accuracy

    except Exception as e:
        # CustomException should be defined elsewhere in your code
        raise CustomException(e, sys)


def make_metrics_json(test_name, path_to_csv, path_to_save_json):
    try:
        # Load the data
        data = pd.read_csv(path_to_csv)

        # Creating the dictionary
        metrics = {}
        for index, row in data.iterrows():
            metrics = {
                "train_accuracy": row["  metrics/accuracy_top1"],
                "train_loss": row["             train/loss"],
            }

        # Constructing the full path for the JSON file
        json_filename = f"{test_name}_metrics.json"
        full_path = os.path.join(path_to_save_json, json_filename)

        # Saving the dictionary as a JSON file
        with open(full_path, "w") as json_file:
            json.dump(metrics, json_file)

    except Exception as e:
        raise CustomException(e, sys)


# ___________ MAKE PREDICTION FOR 1 IMAGE TO TEST ACCURACY ___________ #
class YOLOv8_classifier:
    """
    This class is the core part of the project. It loads a chosen model of YOLOv8, class indices and retrieve uploaded images to use in prediction generating a series of probabilities for class indices, the class index, with the highest probability, will be chosen and mapped with its label (i.e., common name) and scientific name.

    """

    def __init__(
        self, path_to_images, path_to_chosen_model, path_to_class_indices_json
    ):
        """

        This initialization part is to load required items, including a chosen model, class indices, and the test image.

        """

        self.path_to_chosen_model = path_to_chosen_model
        self.path_to_images = path_to_images

        with open(path_to_class_indices_json, "r") as file:
            self.class_indices = json.load(file)

        # Convert keys to integers
        self.class_indices = {int(k): v for k, v in self.class_indices.items()}

        self.detect_model = YOLO("yolov8n.pt")

        self.classify_model = YOLO(self.path_to_chosen_model)

        # remove previous detection/s
        path_to_runs = os.path.join(script_path, "runs")
        if os.path.exists(path_to_runs):
            shutil.rmtree(path_to_runs)

    # ________________ MAKE PREDICTION IMAGES ________________ #
    def make_prediction(self):
        """

        This function handles the prediciton process.

        """

        try:
            # Process each image in the bird class folder
            for image in os.listdir(self.path_to_images):
                if image.lower().endswith((".jpg", ".jpeg", ".png")):
                    path_to_the_image = os.path.join(self.path_to_images, image)
                    image_org = cv2.imread(path_to_the_image)

                    detected_results = self.detect_model(path_to_the_image, save=True)

                    bboxes = detected_results[0].boxes.xyxy.cpu().numpy().reshape(-1, 4)

                    if bboxes.size > 0:
                        for bbox in bboxes:
                            x1, y1, x2, y2 = map(int, bbox)
                            # Crop the image
                            image_to_use = image_org[y1:y2, x1:x2]
                    else:
                        image_to_use = image_org

                    prediction = self.classify_model.predict(image_to_use, device="cpu")
                    predicted_index = prediction[0].probs.top1
                    predicted_probability = prediction[0].probs.top1conf

                    prediction_dict = self.class_indices[predicted_index]
                    predicted_label = prediction_dict["label"]
                    predicted_scientific_name = prediction_dict["scientific_name"]

            return (predicted_probability, predicted_label, predicted_scientific_name)

        except Exception as e:
            # CustomException should be defined elsewhere in your code
            raise CustomException(e, sys)


if __name__ == "__main__":
    # ________________ HANDLE THE PATH THING ________________ #
    # get the absolute path of the script's directory
    script_path = os.path.dirname(os.path.abspath(__file__))
    # get the parent directory of the script's directory
    parent_path = os.path.dirname(script_path)
    sys.path.append(parent_path)

    path_to_images = os.path.join(script_path, "input", "data", "real_imgs")
    path_to_chosen_model = os.path.join(
        script_path, "models", "YOLOv8", "test0", "train", "weights", "last.pt"
    )
    path_to_class_indices_json = os.path.join(
        script_path, "input", "data", "class_indices.json"
    )
    yo = YOLOv8_classifier(
        path_to_images, path_to_chosen_model, path_to_class_indices_json
    )

    (
        predicted_probability,
        predicted_label,
        predicted_scientific_name,
    ) = yo.make_prediction()

    print(predicted_label)
