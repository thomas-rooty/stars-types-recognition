import pandas as pd
from sklearn.model_selection import train_test_split

# load the stars dataset (excluding rows with null values)
stars = pd.read_csv('assets/stars.csv', na_values=['?']).dropna()
stars_classes = ['Brown Dwarf', 'Red Dwarf', 'White Dwarf', 'Main Sequence', 'Super Giants', 'Hyper Giants']

features = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Star type', 'Star color',
            'Spectral Class']
label = 'Star type'

# Split data 70%-30% into training and test sets
x_train, x_test, y_train, y_test = train_test_split(stars[features].values,
                                                    stars[label].values,
                                                    test_size=0.30,
                                                    random_state=0)

print('Training Set: %d, Test Set: %d \n' % (len(x_train), len(x_test)))
print("Sample of features and labels:")

# Take a look at the first 25 training features and corresponding labels
for n in range(0, 24):
    print(x_train[n], y_train[n], '(' + stars_classes[y_train[n]] + ')')
