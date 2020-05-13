""" knn_classification.py
   This is a class to make an instance of a Knn classifier.
    Author: Alan Rocha González
    Email: alan.rocha@udem.edu
    Institution: Universidad de Monterrey
    Created: Thursday 14th, 2020
"""
# Import Knn class 
from knn import Knn

# Create a Knn object
knn_classifier = Knn("diabetes.csv")

# Set k as 5
knn_classifier.set_k(5)
knn_classifier.get_confusion_matrix()

# Set k as 10
knn_classifier.set_k(10)
knn_classifier.get_confusion_matrix()

# Set k as 20
knn_classifier.set_k(20)
knn_classifier.get_confusion_matrix()