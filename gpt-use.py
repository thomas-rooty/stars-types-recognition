import pandas as pd
from keras.saving.save import load_model
import numpy as np

# load the stars dataset (excluding rows with null values)
stars = pd.read_csv('assets/stars.csv', na_values=['?']).dropna()
stars_classes = ['Brown Dwarf', 'Red Dwarf', 'White Dwarf', 'Main Sequence', 'Super Giants', 'Hyper Giants']

# Load the saved model
model = load_model('assets/gpt-model.h5')

# Create a new array of features (Temperature, Luminosity, Radius, Absolute Magnitude)
x_new = np.array([[12098, 689, 7.01, 0.02]])
print('New sample: {}'.format(x_new))

# Use the model to predict the class
class_probabilities = model.predict(x_new)
predictions = np.argmax(class_probabilities, axis=1)

print(stars_classes[predictions[0]])
