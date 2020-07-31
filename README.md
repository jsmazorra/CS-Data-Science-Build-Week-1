k Nearest Neighbor Classifier Algorithm

Required Packages:
- numpy
- scikit-learn (for testing)
Usage:
Imports
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from knn import Knn

First test data:
wine = load_wine()
data = wine.data
target = wine.target

Second test data:
digits = load_digits()
data = digits.data
target = digits.target

Model:
# Instantiate model
classifier = Knn(n_neighbors=10)

# Fit
classifier.fit(data, target)

# Prediction
classifier.predictor([[1,2,3,4,5,6,7,8,9,10]])

# Nearest neighbors and euclidean distance (specified in n_neighbors)
classifier.dis_neighbors[[1,2,3,4,5,6,7,8,9,10]])