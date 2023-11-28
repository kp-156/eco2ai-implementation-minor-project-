from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import eco2ai
from eco2ai import track

app = Flask(__name__)

# Load Mnist Dataset (for demonstration purposes)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train_and_visualize', methods=['POST'])
def train_and_visualize():
    selected_model = request.form['selected_model']

    # Load the selected model
    model_path = f'models/{selected_model.lower()}_model.h5'
    loaded_model = load_model(model_path)

    # Train the loaded model
    history = train_model(loaded_model, x_train, y_train, x_test, y_test)

    # Plot the training history
    chart_data = plot_training_history(history, selected_model)

    return render_template('index.html', chart_data=chart_data)

def train_model(model, x_train, y_train, x_test, y_test):
    # Creating tracker object
    tracker = eco2ai.Tracker(project_name="mnist", experiment_description="Convolutional model")
    
    # Start tracking
    tracker.start()

    batch_size = 256 

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=2, validation_data=(x_test, y_test), verbose=1)

    # End command
    tracker.stop()

    return history

def plot_training_history(history, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Training History for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Save the plot to a static image file
    chart_filename = f'static/{model_name.lower()}_chart.png'
    plt.savefig(chart_filename)
    plt.close()

    # Return the filename for rendering in the HTML
    return chart_filename

if __name__ == '__main__':
    app.run(debug=True)
