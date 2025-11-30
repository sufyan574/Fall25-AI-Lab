from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained ML model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get all 9 numeric values from form
        unnamed_0 = float(request.form['f0'])
        source_city = float(request.form['f1'])
        departure_time = float(request.form['f2'])
        stops = float(request.form['f3'])
        arrival_time = float(request.form['f4'])
        destination_city = float(request.form['f5'])
        class_value = float(request.form['f6'])
        duration = float(request.form['f7'])
        days_left = float(request.form['f8'])

        # Arrange inputs in the same order as training
        input_data = np.array([[unnamed_0, source_city, departure_time, stops,
                                arrival_time, destination_city, class_value,
                                duration, days_left]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template("index.html", result=prediction)

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
