import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# load the stars dataset (excluding rows with null values)
stars = pd.read_csv('assets/stars.csv', na_values=['?']).dropna()
stars_classes = ['Brown Dwarf', 'Red Dwarf', 'White Dwarf', 'Main Sequence', 'Super Giants', 'Hyper Giants']

# Scale the data
# Scale the columns "Temperature (K)", "Luminosity(L/Lo)", "Radius(R/Ro)", "Absolute magnitude(Mv)"
stars['Temperature (K)'] = stars['Temperature (K)'].apply(lambda x: (x - stars['Temperature (K)'].min()) / (stars['Temperature (K)'].max() - stars['Temperature (K)'].min()))
stars['Luminosity(L/Lo)'] = stars['Luminosity(L/Lo)'].apply(lambda x: (x - stars['Luminosity(L/Lo)'].min()) / (stars['Luminosity(L/Lo)'].max() - stars['Luminosity(L/Lo)'].min()))
stars['Radius(R/Ro)'] = stars['Radius(R/Ro)'].apply(lambda x: (x - stars['Radius(R/Ro)'].min()) / (stars['Radius(R/Ro)'].max() - stars['Radius(R/Ro)'].min()))
stars['Absolute magnitude(Mv)'] = stars['Absolute magnitude(Mv)'].apply(lambda x: (x - stars['Absolute magnitude(Mv)'].min()) / (stars['Absolute magnitude(Mv)'].max() - stars['Absolute magnitude(Mv)'].min()))

features = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Star color', 'Spectral Class']
label = 'Star type'

# Split data 70%-30% into training and test sets
x_train, x_test, y_train, y_test = train_test_split(stars[features].values,
                                                    stars[label].values,
                                                    test_size=0.30,
                                                    random_state=0)

# Set random seed for reproducability
tensorflow.random.set_seed(0)

# Use all the threads
tensorflow.config.threading.set_intra_op_parallelism_threads(1)
tensorflow.config.threading.set_inter_op_parallelism_threads(40)

# Set data types for categorical labels
# Set data types for float features
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = tensorflow.keras.utils.to_categorical(y_train, 6)
y_test = tensorflow.keras.utils.to_categorical(y_test, 6)
print("Ready to build model.")

# Define a neural network
# Define a classifier network
hl = 10  # Number of hidden layer nodes
model = Sequential()
model.add(Dense(hl, input_dim=len(features), activation='relu'))
model.add(Dense(hl, input_dim=hl, activation='relu'))
model.add(Dense(len(stars_classes), input_dim=hl, activation='softmax'))

print(model.summary())

# Train the model
# hyper-parameters for optimizer
learning_rate = 0.001
opt = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Train the model over 50 epochs using 10-observation batches and using the test holdout dataset for validation
# Ask user the number of epochs
num_epochs = int(input("Enter the number of epochs: "))
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=10, validation_data=(x_test, y_test))

# Plot the training and validation loss
epoch_nums = range(1, num_epochs + 1)
training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
fig1 = plt.gcf()
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

# Ask user if they want to save the plot
save_plot = input("Do you want to save the plot? (y/n): ")
if save_plot == 'y':
    fig1.savefig('assets/loss.png')

# View the learned weights and biases
for layer in model.layers:
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    print('------------\nWeights:\n', weights, '\nBiases:\n', biases)

# Evaluate the model
class_probabilities = model.predict(x_test)
predictions = np.argmax(class_probabilities, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Plot the confusion matrix
cm = confusion_matrix(true_labels, predictions)
fig2 = plt.gcf()
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(stars_classes))
plt.xticks(tick_marks, stars_classes, rotation=85)
plt.yticks(tick_marks, stars_classes)
plt.xlabel("Predicted Types")
plt.ylabel("Actual Types")
plt.subplots_adjust(bottom=0.3)
plt.show()

# Ask user if they want to save the plot
save_plot = input("Do you want to save the plot? (y/n): ")
if save_plot == 'y':
    fig2.savefig('assets/confusion_matrix.png')

# Show the model accuracy
print('----------------------------------------')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('----------------------------------------')

# Ask if user wants to save the model
save = input("Save model? (y/n): ")
if save == 'y':
    # Saved the trained model
    modelFileName = 'assets/stars-classifier.h5'
    model.save(modelFileName)
    del model
    print("Saved model to disk as", modelFileName)
else:
    print("Model not saved.")
