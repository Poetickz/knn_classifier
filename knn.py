""" Knn.py
   This is a class to make an instance of a Knn classifier.
    Author: Alan Rocha González
    Email: alan.rocha@udem.edu
    Institution: Universidad de Monterrey
    Created: Thursday 14th, 2020
"""
# Importing necessary libraries
import numpy as np
import pandas as pd

# Knn class
class Knn(object):
    def __init__(self, file, k=3):
        """
        INPUT: data set path & K
        OUTPUT: Knn instance
        """
        """
        Construction Class method. This method set all attributes of the class
        and call the load_data method.
        
        """
        self.x_data = None
        self.y_data = None
        self.x_testing_data = None
        self.y_testing_data = None
        self.x_testing_data_unscaled = None
        self.k = k
        self.mean = []
        self.std = []
        self.load_data(file)
    


    def load_data(self, file):
        """ 
        load data from comma-separated-value (CSV) file and set
        x_data, y_data, y_testing_data, x_testing_data.
        """
        """
        INPUT: path_and_filename: the csv file name
        OUTPUT: None
        """
        # Opens file
        try:
            training_data = self.__data_frame(file)

        except IOError:
            print("Error: El archivo no existe o verifique la dirección path.")
            exit(0)

        # Gets rows and columns
        n_rows, n_columns = training_data.shape

        # Gets the testing set
        self.x_testing_data = self.x_testing_data_unscaled = pd.DataFrame.to_numpy(training_data.iloc[:int(n_rows*.05),0:n_columns-1])
        self.y_testing_data = pd.DataFrame.to_numpy(training_data.iloc[:int(n_rows*0.05),-1]).reshape(int(n_rows*0.05),1)

        # Set the training set
        self.x_data = pd.DataFrame.to_numpy(training_data.iloc[int(n_rows*0.05):,0:n_columns-1])

        self.y_data = pd.DataFrame.to_numpy(training_data.iloc[int(n_rows*0.05):,-1]).reshape((n_rows-int(n_rows*0.05)),1)

        self.x_data = self.__feature_scaling(self.x_data, "training")

        self.x_testing_data = self.__feature_scaling(self.x_testing_data, "testing")


    def predict(self, x0):
        """ 
        Method to predict x0 on the training set
        """
        """
        INPUT: x0 Numpy array with the N characteristics
        OUTPUT: Prediction, Prob. Diabetes, Prob. No Diabetes
        """

        # Get N samples
        N = self.x_data.shape[0]

        # Set dictonary
        distances = {}

        # Calculate all euclidean distances
        for x in range(0, N):
            distances[x] = self.__compute_euclidean_distance(self.x_data[x]-x0)

        # Sorting distances
        distances = sorted(distances.items(), key = lambda kv:(kv[1], kv[0]))

        return self.__compute_conditional_probabilities(distances)


    def get_confusion_matrix(self):
        """ 
        Method to get Testing point features and confusion matrix of the 
        """
        """
        INPUT: None
        OUTPUT: None
        """
        # Initiate variables to the confusion matrix
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        print("Testing point (features)")
        print("Pregnancies\tGlucose\t\tBloodPressure\tSkinThickness\tInsulin\t\tBMI\tDiabetes.Ped.Fun.\tAge\tPb. Diabetes\tPb.NO Diabetes")
        
        # Evaluate x_testing
        for x,y,x_testing_data_unscaled in zip(self.x_testing_data, self.y_testing_data, self.x_testing_data_unscaled):
            prediction, zero, one = self.predict(x)
            self.__print_unscaled_result(x_testing_data_unscaled, zero, one)
            if(prediction == 1 and y == 1):
                tp += 1
            if(prediction == 0 and y == 0):
                tn += 1
            if(prediction == 0 and y == 1):
                fn += 1
            if(prediction == 1 and y == 0):
                fp += 1
        
        self.__print_perfomance_metrics(tp, tn, fp, fn)


    def set_k(self, k):
        """ 
        Method to set K attribute
        """
        """
        INPUT: k a integer method
        OUTPUT: None
        """
        self.k = k
        print("Testing K = "+str(k))



    # Private methods
    # ----------------------------------------------------------------------------
    # Set up data methods
    def __data_frame(self, file): 
        """ shuffle data of the csv file """
        """
        INPUT: filename: the csv file name
        OUTPUT: Return the shuffled dataframe
        """
        df = pd.read_csv(file, header=0)
        # return the pandas dataframe
        return df.reindex(np.random.permutation(df.index))

    def __feature_scaling_operation(self, data, mean_value, std_value):
        """ standarize the x data and saves mean value & std value"""
        """
        INPUT: data: data from de data set that will be standarized (numpy array)
            mean_value: mean_value (float)
            std_value: standard variation value (float)
        OUTPUT: Returns de data set standarized, the mean value and std value
        """
        if mean_value == 0 and std_value == 0:
            std_value = data.std()
            mean_value = data.mean()
        scaling_scores = (data - mean_value) / std_value
        return scaling_scores, mean_value, std_value

    def __feature_scaling(self, x, data_type):
        """ Apply feature scaling for the data set """
        """
        INPUT: x: numpy array dataset
               data_type: string 
        OUTPUT: numpy array 
        """
        scaled_array = []

        if data_type == "training":

            for feature in x.T:
                dataScaled, meanX, stdX = self.__feature_scaling_operation(feature, 0, 0)
                scaled_array.append(np.array(dataScaled))
                self.mean.append(meanX)
                self.std.append(stdX)
        else:
            for feature,mean,std in zip (x.T, self.mean, self.std):
                dataScaled = self.__feature_scaling_operation(feature, mean, std)
                scaled_array.append(np.array(dataScaled[0]))

        return np.array(scaled_array).T



    # ----------------------------------------------------------------------------
    # Prediction Methods
    
    def __compute_euclidean_distance(self, eval_x):
        """ Apply euclidean distance for an array """
        """
        INPUT: eval_x: numpy array dataset
        OUTPUT: float euclidean distance
        """
        return np.sqrt(np.sum((eval_x)**2))

    def __compute_conditional_probabilities(self, distances):
        """ Define classification probabilities """
        """
        INPUT: distances: dictonary [key: position, value: distance]
        OUTPUT: Prediction, Prob. Diabetes, Prob. No Diabetes
        """
        zeros = 0
        ones = 0
        for predict in range(0, self.k):
            element = distances[predict][0]
            data = self.y_data[element][0]
            if data == 0:
                zeros += 1
            else:
                ones += 1

        if zeros > ones:
            return 0, zeros/self.k, ones/self.k
        else:
            return 1, zeros/self.k, ones/self.k



    # ----------------------------------------------------------------------------
    # Prints methods
    def __print_data_set(self, x_data, y_data, leyend):
        """ prints x and y data """
        """
        INPUT: x_data & y_data: numpy arrays, leyent: title (string)
        OUTPUT: None
        """
        print("\n")
        print("--"*23)
        print(leyend)
        print("--"*23)
        for x,y in zip(x_data, y_data):
            print(x, y)
        print("\n\n\n")
    
    def __print_unscaled_result(self, x_testing_data_unscaled, zero, one):
        """ prints x_testing_data_unscaled, Prob. Diabetes and Prob. No Diabetes  """
        """
        INPUT: x_testing_data_unscaled: numpy array, zero: float prob, one float prob
        OUTPUT: None
        """
        for characteristic in x_testing_data_unscaled:
            print(round(characteristic, 3), end="\t\t")
        print(str(zero)+"\t"+str(one))

    def __print_perfomance_metrics(self, tp, tn, fp, fn):
        """ Display confusion matrix and performance metrics"""
        """
        INPUT:  tp: True positive (count)
                tn = True negative (count)
                fp = False positive (count)
                fn = False negative (count)
        OUTPUT: NONE
        """
        #Prints confusion matrix
        print("\n")
        print("--"*23)
        print("Confusion Matrix")
        print("--"*23)
        print("\t\t\t\t\t\tActual Class")
        print("\t\t\t\t\tGranted(1)\tRefused(0)")
        print("Predicted Class\t\tGranted(1)\tTP: "+str(tp)+"\t\tFP: "+str(fp)+"")
        print("\t\t\tRefused(0)\tFN: "+str(fn)+"\t\tTN: "+str(tn)+"")
        print("\n")

        # Calculate accuracy
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        # Calculate precision
        precision = (tp)/(tp+fp)
        # Calculate recall
        recall = (tp/(tp+fn))
        # Calculate specifity
        specifity = (tn/(tn+fp))
        # Calculate f1 score
        f1 = (2.0*((precision*recall)/(precision+recall)))

        # Print performance metrics
        print("Accuracy:"+str(accuracy))
        print("Precision:"+str(precision))
        print("Recall:"+str(recall))
        print("Specifity:"+str(specifity))
        print("F1 Score: " + str(f1))
        print("\n\n")
