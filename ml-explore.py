import pandas as pd

# load the stars dataset (excluding rows with null values)
stars = pd.read_csv('assets/stars.csv', na_values=['?']).dropna()

# Make a sample
sample = stars.sample(10)

# Classes
stars_classes = ['Brown Dwarf', 'Red Dwarf', 'White Dwarf', 'Main Sequence', 'Super Giants', 'Hyper Giants']
print(sample.columns[0:6].values, 'StarType')
for index, row in sample.iterrows():
    print('[', row[0], row[1], row[2], row[3], int(row[4]), row[5], row[6], ']', stars_classes[int(row[4])])
