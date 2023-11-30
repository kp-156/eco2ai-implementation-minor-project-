import eco2ai
import time


def run_experiment():
    # Your code here
    for i in range(1000):
        print(i)
        time.sleep(1)


if __name__ == "__main__":
    tracker = eco2ai.Tracker(
        project_name="ECO2AI Usage Example",
        experiment_description="training_sign_language_mnist_model")
    tracker.start()
    run_experiment()
    tracker.stop()
