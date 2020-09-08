from sklearn import datasets
from sklearn.model_selection import train_test_split
import KNN


# Import iris dataset

iris = datasets.load_iris()


X = iris.data
Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# print(type(X_train))

# print (X_train.shape, y_train.shape)
# print (X_test.shape, y_test.shape)

k_nearest = KNN.knn(X_train,y_train,4)

print(X_test[:2])
print(y_test[:2])

classify = k_nearest.classify(X_test[:2],1,4)
