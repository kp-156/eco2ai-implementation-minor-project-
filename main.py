import eco2ai
import time
from models.sign_language_mnist import train_model as train_model_basic
from models.sign_language_mnist_relu6 import train_model as train_model_relu6


if __name__ == "__main__":

    # Basic training
    tracker_basic = eco2ai.Tracker(
        project_name="ECO2AI Usage Example",
        experiment_description="training_sign_language_mnist_model",
        file_name="output/emissions_sign_mnist_basic.csv",
    )
    tracker_basic.start()
    print("Tracker started for basic Model training")
    train_model_basic()
    tracker_basic.stop()
    print("Tracker stopped")

    # Relu6 training
    tracker_basic = eco2ai.Tracker(
        project_name="ECO2AI Usage Example",
        experiment_description="training_sign_language_mnist_model",
        file_name="output/emissions_sign_mnist_relu6.csv",
    )
    tracker_basic.start()
    print("Tracker started for relu6 based model training")
    train_model_relu6()
    tracker_basic.stop()
    print("Tracker stopped")
