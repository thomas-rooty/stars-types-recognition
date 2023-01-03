import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data into a Pandas dataframe
df = pd.read_csv('assets/stars.csv')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
  df[['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']], df['Star type'], test_size=0.2)

# Convert the data into a format that can be used by TensorFlow
X_train = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test.values, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test.values, dtype=tf.float32)

# Define the input and output placeholders
ph_input = tf.keras.Input(shape=(4,))
ph_output = tf.keras.layers.Dense(6, activation='softmax')(ph_input)

# Define the model
model = tf.keras.Model(inputs=ph_input, outputs=ph_output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5000)

# Evaluate the model
model.evaluate(X_test, y_test)

# Plot the confusion matrix
cm = confusion_matrix(y_test, model.predict(X_test).argmax(axis=1))
fig2 = plt.gcf()
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(6)
plt.xticks(tick_marks, ['Red Dwarf', 'Brown Dwarf', 'White Dwarf', 'Main Sequence', 'Supergiant', 'Hypergiant'], rotation=45)
plt.yticks(tick_marks, ['Red Dwarf', 'Brown Dwarf', 'White Dwarf', 'Main Sequence', 'Supergiant', 'Hypergiant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Ask user if they want to save the plot
save_plot = input("Do you want to save the plot? (y/n): ")
if save_plot == 'y':
  fig2.savefig('assets/confusion_matrix.png')

# Save the model yes or no
save = input('Save the model? (y/n): ')
if save == 'y':
  model.save('assets/gpt-model.h5')
  # Save it as a TensorFlow.js model
  tfjs.converters.save_keras_model(model, 'assets/gpt-model')
else:
  print('Model not saved')

# Path: gpt-predict.py
