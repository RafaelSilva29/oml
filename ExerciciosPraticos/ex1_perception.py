from mlxtend.plotting import plot_decision_regions
from sklearn import datasets
from sklearn.svm import SVC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
def load_data():
    file = 'ex4_D.csv'
    data = pd.read_csv(file, header = None)

    # make the dataset linearly separable
    data = data[:100]
    data[2] = np.where(data.iloc[:, -1]==-1, 0, 1)
    data = np.asmatrix(data, dtype = 'float64')
    return data

data = load_data()

# Loading some example data
X = np.array(data[:, [0, 1]])
y = data[:, 2]
y = np.int64(np.squeeze(np.asarray(y)))

# Training a classifier
svm = SVC(C=0.5, kernel='linear')
svm.fit(X, y)

# Plotting decision regions
plot_decision_regions(X, y, clf=svm, legend=2)

# Adding axes annotations
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.title('Plotting Decision Regions')
plt.show()


# # Scatter Plot
# plt.scatter(np.array(data[:100,0]), np.array(data[:100,1]), marker='o', label='points')
# # plt.scatter(np.array(data[50:,0]), np.array(data[50:,2]), marker='x', label='versicolor')
# plt.xlabel('petal length')
# plt.ylabel('sepal length')
# plt.legend()
# plt.show()


# Algorithm Perceptron
def perceptron(data, num_iter):
    features = data[:, :-1]
    labels = data[:, -1]

    # set weights to zero
    w = np.zeros(shape=(1, features.shape[1]+1))
    
    misclassified_ = [] 
  
    for epoch in range(num_iter):
        misclassified = 0
        for x, label in zip(features, labels):
            x = np.insert(x,0,1)
            y = np.dot(w, x.transpose())
            target = 1.0 if (y > 0) else 0.0
            print(label.item(0,0))
            delta = (label.item(0,0) - target)
            
            if(delta): # misclassified
                misclassified += 1
                w += (delta * x)
        
        misclassified_.append(misclassified)
    return (w, misclassified_)
             
num_iter = 10
w, misclassified_ = perceptron(data, num_iter)

epochs = np.arange(1, num_iter+1)
plt.plot(epochs, misclassified_)
plt.xlabel('iterations')
plt.ylabel('misclassified')
plt.title('Algorithm Perceptron - Ex4')
plt.show()
