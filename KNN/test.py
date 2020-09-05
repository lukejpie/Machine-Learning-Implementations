import numpy as np
from decimal import Decimal

def distance(v1,v2,distance_function,p_value=3):
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


vector1 = np.array([0, 2, 3, 4])
vector2 = np.array([2, 4, 3, 7])
p = 3

print(distance(vector1,vector2,1,3))
print(distance(vector1,vector2,2,3))
print(distance(vector1,vector2,3,p))
