from knn import Knn
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Let's load the data, I will be using the dataset of house prices in Boston.
boston = load_boston()
data = boston.data
target = iris.target

# In this test, I'll be comparing the performance of my knn algorithm with
# sklearn's own K-Neighbors Classifier.