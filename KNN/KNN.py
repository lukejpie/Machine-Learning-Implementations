import numpy as np

class knn:


    # Initialization of KNN Classifier Object
    def __init__(self,x_training,y_training,k):
        self.x_training = x_training
        self.y_training = y_training
        self.k = k

    # Distance - Calculates the distance between given vectors v1 and v2
    # Params:
    #   v1 - Given vector
    #   v2 - Given vector
    #   distance_function - Selector for distance function to utilize (1 = Euclidean, 2 = Manhattan, 3 = Minkowski)
    #   p_value - P value to be untilized in Minkowski distance calculation (Defaults to 3)
    # Output:
    #   distance between v1 and v2 based on selected distance function

    def __distance(v1,v2,distance_function,p_value=3):
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


    def classifier(,k=self.k)
