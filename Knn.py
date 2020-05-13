""" Knn.py
   This is a class to Knn classifier.

    Author: Alan Rocha González
    Email: alan.rocha@udem.edu
    Institution: Universidad de Monterrey
    Created: Thursday 14th, 2020
"""

import numpy as np
import pandas as pd


class Knn(object):
    def __init__(self, file):
        self.x_data = None
        self.y_data = None
        self.x_testing_data = None
        self.y_testing_data = None
        self.mean = []
        self.std = []
        self.create_data_set(file)
    


    def create_data_set(self, file):
        # Opens file
        try:
            training_data = self.__data_frame(file)

        except IOError:
            print("Error: El archivo no existe")
            exit(0)

        # Gets rows and columns
        n_rows, n_columns = training_data.shape

        # Gets the testing set
        self.x_testing_data = pd.DataFrame.to_numpy(training_data.iloc[:int(n_rows*.05),0:n_columns-1])
        self.y_testing_data = pd.DataFrame.to_numpy(training_data.iloc[:int(n_rows*0.05),-1]).reshape(int(n_rows*0.05),1)

        # Gets the training set
        self.x_data = pd.DataFrame.to_numpy(training_data.iloc[int(n_rows*0.05):,0:n_columns-1])

        self.y_data = pd.DataFrame.to_numpy(training_data.iloc[int(n_rows*0.05):,-1]).reshape((n_rows-int(n_rows*0.05)),1)

        self.__print_data_set(self.x_data, self.y_data, "Training Dataset Set")

        self.x_data = self.__feature_scaling(self.x_data, "training")

        self.x_testing_data = self.__feature_scaling(self.x_testing_data, "testing")

        self.__print_data_set(self.x_data, self.y_data, "Training Scaled Dataset Set")









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
        # Applying feature scaling for training set
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
