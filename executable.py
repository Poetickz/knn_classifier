import sys
sys.path.append(".")

from Knn import Knn


knn_classifier = Knn("diabetes.csv")
knn_classifier.set_k(5)
knn_classifier.get_confusion_matrix()
knn_classifier.set_k(10)
knn_classifier.get_confusion_matrix()
knn_classifier.set_k(20)
knn_classifier.get_confusion_matrix()