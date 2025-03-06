from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


# Load the trained model and vectorizer
model = pickle.load(open("model.pkl", "rb" ))
vectorizer = pickle.load(open("model2.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        message = request.form["message"]
        transformed_message = vectorizer.transform([message])  # Convert text to vector
        prediction = model.predict(transformed_message)[0]

        result = "Spam" if prediction == 1 else "Not Spam"
        return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
