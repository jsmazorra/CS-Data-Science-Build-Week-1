class knn:
    
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
    
    # Going to first start by implementing an euclidean distance function.
    
    def euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
            euclidean_distance = distance ** (1 / 2)
        return euclidean_distance


    # Test distance function

    sample = [
    [2.7810836, 2.550537003, 0],
    [1.465489372, 2.362125076, 0],
    [3.396561688, 4.400293529, 0],
    [1.38807019, 1.850220317, 0],
    [3.06407232, 3.005305973, 0],
    [7.627531214, 2.759262235, 1],
    [5.332441248, 2.088626775, 1],
    [6.922596716, 1.77106367, 1],
    [8.675418651, -0.242068655, 1],
    [7.673756466, 3.508563011, 1],
    ]
    row0 = sample[0]
    for row in sample:
        distance = euclidean_distance(self=row0, row1=row0, row2=row)
        print(distance)

'''
Output should be: 
0.0
1.3290173915275787
1.9494646655653247
1.5591439385540549
0.5356280721938492
4.850940186986411
2.592833759950511
4.214227042632867
6.522409988228337
4.985585382449795
'''
