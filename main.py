import eco2ai
import time
from models.sign_language_mnist import train_model

if __name__ == "__main__":
    tracker = eco2ai.Tracker(
        project_name="ECO2AI Usage Example",
        experiment_description="training_sign_language_mnist_model",
        file_name="output/emissions_sign_mnist_basic.csv",
    )
    tracker.start()
    print("Tracker started")
    print("Model training started")
    train_model()
    tracker.stop()
    print("Tracker stopped")
