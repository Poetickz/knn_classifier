import sys
sys.path.append(".")

from Knn import Knn


knn_classifier = Knn("diabetes.csv", 3)
knn_classifier.evaluate_testing()