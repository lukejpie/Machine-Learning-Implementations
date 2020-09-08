import numpy as np
import pandas as pd
from decimal import Decimal

class knn:


    # Initialization of KNN Classifier Object
    def __init__(self,x_training,y_training,k=3):
        self.x_training = pd.DataFrame(x_training)
        self.y_training = y_training
        self.k = k

    # distance - Private function calculates the distance between given vectors v1 and v2
    # Params:
    #   v1 - Given vector
    #   v2 - Given vector
    #   distance_function - Selector for distance function to utilize (1 = Euclidean, 2 = Manhattan, 3 = Minkowski)
    #   p_value - P value to be untilized in Minkowski distance calculation (Defaults to 3)
    # Output:
    #   distance between v1 and v2 based on selected distance function

    def __distance(self,v1,v2,distance_function,p_value=3):
          if len(v1) != len(v2): return -1

          distance = 0

          if(distance_function == 1):
              distance = np.linalg.norm(v1-v2)
          elif(distance_function == 2):
              # Not as efficient, could be vectorized but simple implementation for now
              for v1_i,v2_i in zip(v1,v2):
                  distance += abs(Decimal(v1_i) - Decimal(v2_i))
          elif(distance_function == 3):
              # Not as efficient, could be vectorized but simple implementation for now
              for v1_i,v2_i in zip(v1,v2):
                  distance += pow((abs(Decimal(v1_i)-Decimal(v2_i))),p_value)
              print(distance)
              distance = distance**(1/Decimal(p_value))

          return distance


    # nearest neighbors - Private function to return the indices of the nearest k neighbors to given test vector
    # Params:
    #   test - given vector to measure nearest neighbors to
    #   k - number of nearest neighbors to evaluate for classifcation
    #   distance_function - Selector for distance function to utilize (1 = Euclidean, 2 = Manhattan, 3 = Minkowski)
    #   p_value - P value to be untilized in Minkowski distance calculation
    # Output:
    #   indices of nearest k neighbors to given test vector
    def __nearest_neighbors(self,test,k,distance_function,p_value):
        #SHOULD USE PANDAS to perform "lapply" type of function for using distance function over all of the rows
        distances = self.x_training.apply(self.__distance,axis=1,args=(test,distance_function,p_value))

        print(self.x_training)
        print()
        print(distances)
        distances_sorted = distances.sort_values(0)
        print(distances_sorted)
        print(distances_sorted[:k].index)
        return

    def __predict(self,nearest_neighbors):
        return

    # classify - Classifies given test point based on k-nearest neighbors from training set
    # Params:
    #   test - given vector to classify
    #   k - number of nearest neighbors to evaluate for classifcation (Defaults to initialized self.k value)
    #   distance_function - Selector for distance function to utilize (1 = Euclidean, 2 = Manhattan, 3 = Minkowski)
    # Output:
    #   classifcation of test vector
    def classify(self,test,distance_function,k = None,p_value=3):
        if k is None:
            k = self.k

        nearest_neighbors = self.__nearest_neighbors(test,k,distance_function,p_value)

        # prediction = __predict(nearest_neighbors)
