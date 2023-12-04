import eco2ai
import time
from models.sign_language_mnist import train_model as train_model_basic
from models.sign_language_mnist_relu6 import train_model as train_model_relu6
from models.sign_language_mnist_loss_function import train_model as train_model_hinge
from models.sign_language_mnist_optimizer import train_model as train_model_sgd


if __name__ == "__main__":

    # Basic training
    # tracker_basic = eco2ai.Tracker(
    #     project_name="ECO2AI Usage Example",
    #     experiment_description="training_sign_language_mnist_model",
    #     file_name="output/emissions_sign_mnist_basic.csv",
    # )
    # tracker_basic.start()
    # print("Tracker started for basic Model training")
    # train_model_basic()
    # tracker_basic.stop()
    # print("Tracker stopped")

    # Relu6 training
    # tracker_relu6 = eco2ai.Tracker(
    #     project_name="ECO2AI Usage Example",
    #     experiment_description="training_sign_language_mnist_model",
    #     file_name="output/emissions_sign_mnist_relu6.csv",
    # )
    # tracker_relu6.start()
    # print("Tracker started for relu6 based model training")
    # train_model_relu6()
    # tracker_relu6.stop()
    # print("Tracker stopped")

    # Hinge loss based training
    # tracker_hinge = eco2ai.Tracker(
    #     project_name="ECO2AI Usage Example",
    #     experiment_description="training_sign_language_mnist_model",
    #     file_name="output/emissions_sign_mnist_hinge_loss.csv",
    # )
    # tracker_hinge.start()
    # print("Tracker started for relu6 based model training")
    # train_model_hinge()
    # tracker_hinge.stop()
    # print("Tracker stopped")

    # SGD optimizer training
    tracker_sgd = eco2ai.Tracker(
        project_name="ECO2AI Usage Example",
        experiment_description="training_sign_language_mnist_model",
        file_name="output/emissions_sign_mnist_sgd_loss.csv",
    )
    tracker_sgd.start()
    print("Tracker started for SGD based model training")
    train_model_sgd()
    tracker_sgd.stop()
    print("Tracker stopped")
