from flask import Flask, request
import pandas as pd
import pickle
from flasgger import Swagger


app = Flask(__name__)
Swagger(app)

# Load the pickle file
pickle_in = open('bank_note_classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    """
    landing page
    Returns:
        string: welcome text
    """
    return "Welcome to Bank Note Authenticator"


@app.route('/predict', methods=['GET'])
def pred_note_authentication():
    """Let's Authenticate a Bank Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    if prediction[0] == 0:
        return "The bank note was not successfully authenticated"
    else:
        return "The bank note was successfully authenticated"


@app.route('/predict_file', methods=['POST'])
def pred_note_file_authentication():
    """Let's Batch Authenticate Bank Notes
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
    """
    data_df = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(data_df)
    return f"The predicted values are {prediction}"


if __name__ == '__main__':
    app.run()
