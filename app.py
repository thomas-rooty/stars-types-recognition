from flask import Flask, request, jsonify
import pandas as pd
from keras.saving.save import load_model
import numpy as np

app = Flask(__name__)


@app.route('/')
def base():
    return 'Star type prediction API using AI'


@app.route('/post', methods=["POST"])
def predict_startype():
    # load the stars dataset (excluding rows with null values)
    stars = pd.read_csv('assets/stars.csv', na_values=['?']).dropna()
    stars_classes = ['Brown Dwarf', 'Red Dwarf', 'White Dwarf', 'Main Sequence', 'Super Giants', 'Hyper Giants']

    # Load the saved model
    model = load_model('assets/stars-classifier.h5')

    # CReate a new array of features
    input_json = request.get_json(force=True)
    res = {'star_values': input_json['star_values']}

    x_new = np.array([res['star_values']])
    print('New sample: {}'.format(x_new))

    # Use the model to predict the class
    class_probabilities = model.predict(x_new)
    predictions = np.argmax(class_probabilities, axis=1)

    print(stars_classes[predictions[0]])

    # return the prediction
    return jsonify({'prediction': stars_classes[predictions[0]]})
