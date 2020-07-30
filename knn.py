# Library imports
import numpy as np


class Knn:
    """
    IMPLEMENTATION
    Method:
    - euclidean_distance(row1, row2):
        Returns euclidian distance of values
    - fit(X_train, y_train):
        Fits model to training data
    - predictor(X):
        Returns predictions for X based on fitted model"""
    
    # Initialization of the algorithm.
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors

    # Going to first start by implementing an euclidean distance function.

    def euclidean_distance(self, row1, row2):
        """
        Returns euclidian distance of values between row1 and row2.
        Inputs: row1 : int or float.
                row2 : int or float.
        Output: euclidian_distance : float
        """
        distance = 0.0
        for i in range(len(row1) - 1):
            """Calculation: Subtract b from a, square difference,
            add to euclidean_distance.
            """
            distance += (row1[i] - row2[i]) ** 2
            euclidean_distance = distance ** (1 / 2)
        return euclidean_distance

    # Let's add some dummy data to test this function out.
    # sample = [
    # [2.7810836, 2.550537003, 0],
    # [1.465489372, 2.362125076, 0],
    # [3.396561688, 4.400293529, 0],
    # [1.38807019, 1.850220317, 0],
    # [3.06407232, 3.005305973, 0],
    # [7.627531214, 2.759262235, 1],
    # [5.332441248, 2.088626775, 1],
    # [6.922596716, 1.77106367, 1],
    # [8.675418651, -0.242068655, 1],
    # [7.673756466, 3.508563011, 1], ]
    # row0 = sample[0]
    # for row in sample:
    #    distance = euclidean_distance(self=row0, row1=row0, row2=row)
    #    print(distance)

    def fit(self, X_train, y_train):
        """Fit the model using X as training data and y as target values
        Parameters
        ----------
        X_train : array-like shape of int or float.
        y : array-like shape or list of target.
        Output will pass along to the predictor function below.
        """

        # Function to fit models for training data.

        self.X_train = X_train
        self.y_train = y_train

       # Predict x for knn.
    def predictor(self, X):
        """Predict the class labels for the provided data.
        Parameters
        ----------
        X : array-like shape or list.
        Returns
        -------
        y : list of floats for each vector in X.
        """
        predictor = []

        # Main loop doing an iteration through the length of X.

        for index in range(len(X)):
            euclidian_distances = []

            for row in self.X_train:
                eucl_distance = self.euclidean_distance(row, X[index])
                euclidian_distances.append(eucl_distance)

            neighbors = \
                np.array(euclidian_distances).argsort()[:self.n_neighbors]
            count_neighbors = {}

            for val in neighbors:
                if self.y_train[val] in count_neighbors:
                    count_neighbors[self.y_train[val]] += 1
                else:
                    count_neighbors[self.y_train[val]] = 1

            predictor.append(max(count_neighbors,
                             key=count_neighbors.get))

        return predictor
