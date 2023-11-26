from flask import Flask, render_template
from eco2ai import calculate_carbon_footprint  # Import the eco2AI library

app = Flask(__name__)

@app.route('/')
def index():
    # Static data for demonstration; replace this with actual data
    model_names = ['Model A', 'Model B', 'Model C']
    power_consumption = [150, 200, 180]  # Power consumption in watts for each model
    training_time = [8, 10, 9]  # Training time in hours for each model

    # Calculate carbon footprint using eco2AI
    carbon_footprints = [calculate_carbon_footprint(power, time) for power, time in zip(power_consumption, training_time)]

    return render_template('index.html', model_names=model_names, carbon_footprints=carbon_footprints)

if __name__ == '__main__':
    app.run(debug=True)

