from __future__ import annotations
from typing import *

def distance(a: List[int], b: List[int]) -> float:
    """determine the distance between two vectors
    Args:
        a (List[int]): A list representing a vector, must be the same length as b
        b (List[int]): A list representing a vector, must be the same length as a
    Raises:
        ValueError: if the length of a is not equal to b
    Returns:
        float: distance between vectors (or similarity between points)
    """

    if len(a) != len(b):
        raise ValueError("length of vectors must be equal")

    # distance formula
    return sum([(a[i] - b[i]) ** 2 for i in range(len(a))]) ** 0.5

class KNNClassification():
    def __init__(self, n_neighbors: int = 5):
        """Create a new KNN Classifier

        Args:
            n_neighbors (int, optional): Amount of neighbors to take into consideration. Defaults to 5.
        """
        
        self.k = n_neighbors
        self.distances = []
        self.X = []  
        self.y = []

    def fit(self, X: List[Any], y: List[Any]):
        """Set the training data

        Args:
            X (List[Any]): Input data
            y (List[Any]): Expected output data
        """

        self.X = X
        self.y = y

    def predict(self, input: List[Any]) -> Any:
        """Make a prediction given some data

        Args:
            input (List[Any]): Input row

        Returns:
            Any: Predicted output label
        """

        distances = [distance(row, input) for row in self.X]
        neighbors = []

        targets = self.y
        k = self.k

        while k > 0:
            idx = distances.index(min(distances))
            neighbors.append(targets[idx])

            distances.pop(idx)
            targets.pop(idx)

            k -= 1

        return max(set(neighbors), key=neighbors.count)

# dummy data
X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
y = ["A", "A", "A", "A", "B", "B"]

# create a new KNNClassifier
classifier = KNNClassification()
classifier.fit(X, y)

print(classifier.predict([3, 4]))