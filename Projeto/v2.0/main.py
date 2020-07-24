#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:04:30 2020

@author: GrupoA - OML - 19/20
"""

import tensorflow as tf
import numpy as np
from MCP_Primal import MCP_Primal
from MCP_Dual import MCP_Dual 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_digits
import time
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings('ignore')


###################################################################################################
################################# Functions #######################################################


def perceptron_primal(x_train, y_train, x_test, y_test, classifier='ova', max_iters=200):
    print('\n')
    if classifier == 'ova':
        title = 'OneVsAll'
    elif classifier == 'ovo':
        title = 'OneVsOne'
    elif classifier == 'ecoc':
        title = 'ECOC'
    print('#'*10 + ' Results - '+ title + ' - Primal ' + '#'*10)
    
    # Number of unique class labels  which is also the number of classifiers we will train 
    unique_labels = np.unique([y_train])
    num_classifiers = unique_labels.size
    
    # Train the model
    model = MCP_Primal(num_classifiers, unique_labels, max_iters, learningrate=0.1, classifier=classifier)
    model.fit(x_train, y_train)
    
    # Run predictions on test data 
    y_predicted = model.predict(x_test)
    
    # Score model
    score = accuracy_score(y_test, y_predicted)
    print('\nAccuracy of '+ title + ' - Primal: ' + str(score))

    return score, y_predicted, model, title


def perceptron_dual(x_train, y_train, x_test, y_test, classifier='ova', max_iters=200, kernel='none'):
    print('\n')
    if classifier == 'ova':
        title = 'OneVsAll'
    elif classifier == 'ovo':
        title = 'OneVsOne'
    elif classifier == 'ecoc':
        title = 'ECOC'
    print('#'*10 + ' Results - '+ title + ' - Dual - Kernel(' + kernel + ') ' + '#'*10)
    
    # Number of unique class labels  which is also the number of classifiers we will train 
    unique_labels = np.unique([y_train])
    num_classifiers = unique_labels.size
    
    # Train the model
    model = MCP_Dual(num_classifiers, unique_labels, max_iters=10, kernel=kernel, classifier=classifier)
    model.fit(x_train, y_train)
    
    print('> Make predictions, please wait...')
    
    # Run predictions on test data 
    y_predicted = model.predict(x_test)
    
    # Score model
    score = accuracy_score(y_test, y_predicted)

    print('\nAccuracy of '+ title + ' - Dual - Kernel(' + kernel + '): ' + str(score))
    
    return score, y_predicted, model, title


def confusion_matrix(y_test, predictions, score, title):
    unique_labels = np.unique([y_test])
    cm = metrics.confusion_matrix(y_test, predictions)
    plt.figure(figsize=(12,12))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r', xticklabels=unique_labels, yticklabels=unique_labels);
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = title + ' - Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15);
    plt.show()


def prepare_data(X, y, lista):    
    threshold = np.where(y[:, None] == np.array(lista).ravel())[0]
    y_new, x_new = y[threshold], X[threshold]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_new_scaled = scaler.fit_transform(x_new)

    x_train, x_test, y_train, y_test = train_test_split(x_new_scaled, y_new, test_size=0.20, random_state=0)
    
    #print(x_train.shape)
    #print(y_train.shape)
    #print(x_test.shape)
    #print(y_test.shape)
    
    return x_train, y_train, x_test, y_test


def plot_erros(model, title):
    for classifier in model.classifiers:
        plt.plot(range(1, len(classifier.errors_) + 1), classifier.errors_, marker='o')
        plt.title('Misclassification' + title)
        plt.xlabel('Iterations')
        plt.ylabel('Number of misclassification')
        plt.show()
        

def plot_predictions(teste_x, predicted, teste_y, title):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title, size=16)
    for i in range(32):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(teste_x.reshape(-1, 8, 8)[i], cmap=plt.cm.binary, interpolation='nearest')
        if predicted[i] == teste_y[i]:
            ax.text(0, 7, str(predicted[i]), color='green')
        else:
            ax.text(0, 7, str(predicted[i]), color='red') 
    plt.show()


def explore(x, y, num_class):
    plt.figure(figsize=(10, 9))
    plt.title('Decision regions')
    pca = PCA(n_components=num_class)
    proj = pca.fit_transform(x)
    plt.scatter(proj[:, 0], proj[:, 1], c=y, cmap="Paired")
    plt.colorbar()
    plt.show()


###################################################################################################
################################# Load Data #######################################################


# Load digits
digits_total = load_digits()

# Load mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

X_mnist = np.concatenate((x_train, x_test))
y_mnist = np.concatenate((y_train, y_test))

X_mnist_threshold = X_mnist[:1500]
y_mnist__threshold = y_mnist[:1500]
X_mnist_threshold = X_mnist_threshold.reshape(X_mnist_threshold.shape[0], -1)
y_mnist__threshold = y_mnist__threshold.reshape(y_mnist__threshold.shape[0], -1)
y_mnist__threshold = y_mnist__threshold.ravel()


###################################################################################################
################################# Main ############################################################


begin = False
while begin == False:
    print("### Perceptron type ###")
    print("1- Primal;")
    print("2- Dual;")   
    print("3- Kernel (RBF);")
    print("4- Polynomial;")
    print("0- exit;")
    kernel = input("Pick an option: ")
    
    if kernel == '1' or kernel == '2' or kernel == '3' or kernel == '4':
        finish = False
        while finish == False:
            inside_loop = False
            print("### Dataset ###")
            print("1- Digits;")
            print("2- Mnist;")   
            print("0- exit;") 
            op = input("Pick an option: ")
            
            if op == '1':
                # Digits
                digits = digits_total
                while inside_loop == False:
                    print("### Classiflier ###")
                    print("1- OneVsAll;")
                    print("2- OneVsOne;") 
                    print("3- ECOC;")
                    print("0- exit;") 
                    op_temp = input("Pick an option: ")
                    
                    if op_temp == '1':
                        # One Vs All
                        lista = list(map(int, input("Enter the values (max 4): ").split()))
                        if len(lista) <= 4 and max(lista) <= 9 and len(lista) >= 2 and min(lista) >= 0:
                            treino_x, treino_y, teste_x, teste_y = prepare_data(digits.data, digits.target, lista)
                            unique_labels = np.unique([treino_y])
                            num_classifiers = unique_labels.size
                            explore(treino_x, treino_y, num_classifiers)
                            
                            if kernel == '1':
                                
                                start_time = time.time()
                                
                                score, predicted, model, title = perceptron_primal(treino_x, treino_y , teste_x, teste_y, classifier='ova', max_iters=treino_x.shape[0])
                            
                                finish_time = time.time() - start_time
                                
                                print('Time OneVsAll - Primal: %.5fs' % finish_time)
                                print('> Dataset digits')
                                print('> Build the results, please wait...')
                            
                                plot_erros(model, title + ' - Primal')
                                
                                confusion_matrix(teste_y, predicted, score, title + ' - Primal')
                                
                                plot_predictions(teste_x, predicted, teste_y, title + ' - Primal - ' + str(score))
        
                                print('\n')
                                
                            else:
                                
                                if kernel == '2':
                                    kernel_type = 'none'
                                elif kernel == '3':
                                    kernel_type = 'rbf'
                                elif kernel == '4':
                                    kernel_type = 'polynomial'
                                    
                                start_time = time.time()
                                
                                score, predicted, model, title = perceptron_dual(treino_x, treino_y , teste_x, teste_y, classifier='ova', max_iters=10, kernel=kernel_type)
                            
                                finish_time = time.time() - start_time
                                
                                print('Time OneVsAll - Dual - Kernel(%s): %.5fs' % (kernel_type, finish_time))
                                print('> Dataset digits')
                                print('> Build the results, please wait...')
                            
                                confusion_matrix(teste_y, predicted, score, title + ' - Dual - Kernel(' + kernel_type + ')')
                                
                                plot_erros(model, title + ' - Dual - Kernel(' + kernel_type + ')')
                                
                                plot_predictions(teste_x, predicted, teste_y, title + ' - Dual - Kernel(' + kernel_type + ') - ' + str(score))
        
                                print('\n')
                            
                        else:
                            print("Error: Numbers size incorrect!")
                        
                    elif op_temp == '2':
                        # One Vs One
                        lista = list(map(int, input("Enter the values (max 4): ").split()))
                        if len(lista) <= 4 and max(lista) <= 9 and len(lista) >= 2 and min(lista) >= 0:
                            treino_x, treino_y , teste_x, teste_y = prepare_data(digits.data, digits.target, lista)
                            unique_labels = np.unique([treino_y])
                            num_classifiers = unique_labels.size
                            explore(treino_x, treino_y, num_classifiers)
                            
                            if kernel == '1':
                                start_time = time.time()
                                
                                score, predicted, model, title = perceptron_primal(treino_x, treino_y , teste_x, teste_y, classifier='ovo', max_iters=treino_x.shape[0])
                            
                                finish_time = time.time() - start_time
                                
                                print('Time OneVsOne - Primal: %.5fs' % finish_time)
                                print('> Dataset digits')
                                print('> Build the results, please wait...')
                                
                                plot_erros(model, title + ' - Primal')
                                
                                confusion_matrix(teste_y, predicted, score, title + ' - Primal')
                                
                                plot_predictions(teste_x, predicted, teste_y, title + ' - Primal - ' + str(score))

                                print('\n')
                            
                            else:
                                
                                if kernel == '2':
                                    kernel_type = 'none'
                                elif kernel == '3':
                                    kernel_type = 'rbf'
                                elif kernel == '4':
                                    kernel_type = 'polynomial'
                                    
                                
                                start_time = time.time()
                                
                                score, predicted, model, title = perceptron_dual(treino_x, treino_y, teste_x, teste_y, classifier='ovo', max_iters=10, kernel=kernel_type)
                            
                                finish_time = time.time() - start_time
                                
                                print('Time OneVsOne - Dual - %s: %.5fs' % (kernel_type, finish_time))
                                print('> Dataset digits')
                                print('> Build the results, please wait...')
                                
                                plot_erros(model, title + ' - Dual - Kernel(' + kernel_type + ')')
                            
                                confusion_matrix(teste_y, predicted, score, title + ' - Dual - ' + kernel_type)
                                
                                plot_predictions(teste_x, predicted, teste_y, title + ' - Dual - Kernel(' + kernel_type + ') - ' + str(score))
        
                                print('\n')
                            
                        else:
                            print("Error: Numbers size incorrect!")
                        
                    elif op_temp == '3':
                        # ECOC
                        lista = list(map(int, input("Enter the values (max 4): ").split()))
                        if len(lista) <= 4 and max(lista) <= 9 and len(lista) >= 2 and min(lista) >= 0:
                            treino_x, treino_y, teste_x, teste_y = prepare_data(digits.data, digits.target, lista)
                            unique_labels = np.unique([treino_y])
                            num_classifiers = unique_labels.size
                            explore(treino_x, treino_y, num_classifiers)
                            
                            if kernel == '1':
                                
                                start_time = time.time()
                                
                                score, predicted, model, title = perceptron_primal(treino_x, treino_y , teste_x, teste_y, classifier='ecoc', max_iters=treino_x.shape[0])
                            
                                finish_time = time.time() - start_time
                                
                                print('Time ECOC - Primal: %.5fs' % finish_time)
                                print('> Dataset digits')
                                print('> Build the results, please wait...')
                            
                                plot_erros(model, title + ' - Primal')
                                
                                confusion_matrix(teste_y, predicted, score, title + ' - Primal')
                                
                                plot_predictions(teste_x, predicted, teste_y, title + ' - Primal - ' + str(score))
        
                                print('\n')
                                
                            else:
                                
                                if kernel == '2':
                                    kernel_type = 'none'
                                elif kernel == '3':
                                    kernel_type = 'rbf'
                                elif kernel == '4':
                                    kernel_type = 'polynomial'
                                    
                                start_time = time.time()
                                
                                score, predicted, model, title = perceptron_dual(treino_x, treino_y , teste_x, teste_y, classifier='ecoc', max_iters=10, kernel=kernel_type)
                            
                                finish_time = time.time() - start_time
                                
                                print('Time ECOC - Dual - Kernel(%s): %.5fs' % (kernel_type, finish_time))
                                print('> Dataset digits')
                                print('> Build the results, please wait...')
                            
                                confusion_matrix(teste_y, predicted, score, title + ' - Dual - Kernel(' + kernel_type + ')')
                                
                                plot_erros(model, title + ' - Dual - Kernel(' + kernel_type + ')')
                                
                                plot_predictions(teste_x, predicted, teste_y, title + ' - Dual - Kernel(' + kernel_type + ') - ' + str(score))
        
                                print('\n')
                            
                        else:
                            print("Error: Numbers size incorrect!")
                    
                    elif op_temp == '0':
                        inside_loop = True
                    
                    else:
                       print("Error: Enter a valid number!")
     
            elif op == '2':
                # MNIST
                x_mnist = X_mnist_threshold
                y_mnist = y_mnist__threshold
                while inside_loop == False:
                    print("### Classiflier ###")
                    print("1- OneVsAll;")
                    print("2- OneVsOne;") 
                    print("3- ECOC;")
                    print("0- exit;") 
                    op_temp = input("Pick an option: ")
                    
                    if op_temp == '1':
                        # One Vs All
                        lista = list(map(int, input("Enter the values (max 4): ").split()))
                        if len(lista) <= 4 and max(lista) <= 9 and len(lista) >= 2 and min(lista) >= 0:
                            treino_x, treino_y, teste_x, teste_y = prepare_data(x_mnist, y_mnist, lista)
                            unique_labels = np.unique([treino_y])
                            num_classifiers = unique_labels.size
                            explore(treino_x, treino_y, num_classifiers)
                            
                            if kernel == '1':
                                
                                start_time = time.time()
                                
                                score, predicted, model, title = perceptron_primal(treino_x, treino_y , teste_x, teste_y, classifier='ova', max_iters=treino_x.shape[0])
                            
                                finish_time = time.time() - start_time
                                
                                print('Time OneVsAll - Primal: %.5fs' % finish_time)
                                print('> Dataset mnist')
                                print('> Build the results, please wait...')
                                
                                plot_erros(model, title + ' - Primal')
                            
                                confusion_matrix(teste_y, predicted, score, title + ' - Primal')

                                print('\n')
                                
                            else:
                                
                                if kernel == '2':
                                    kernel_type = 'none'
                                elif kernel == '3':
                                    kernel_type = 'rbf'
                                elif kernel == '4':
                                    kernel_type = 'polynomial'
                                    
                                start_time = time.time()
                                
                                score, predicted, model, title = perceptron_dual(treino_x, treino_y , teste_x, teste_y, classifier='ova', max_iters=10, kernel=kernel_type)
                            
                                finish_time = time.time() - start_time
                                
                                print('Time OneVsAll - Dual - Kernel(%s): %.5fs' % (kernel_type, finish_time))
                                print('> Dataset mnist')
                                print('> Build the results, please wait...')
                            
                                confusion_matrix(teste_y, predicted, score, title + ' - Dual - Kernel(' + kernel_type + ')')
                                
                                plot_erros(model, kernel_type)
        
                                print('\n')
                            
                        else:
                            print("Error: Numbers size incorrect!")
                        
                    elif op_temp == '2':
                        # One Vs One
                        lista = list(map(int, input("Enter the values (max 4): ").split()))
                        if len(lista) <= 4 and max(lista) <= 9 and len(lista) >= 2 and min(lista) >= 0:
                            treino_x, treino_y, teste_x, teste_y = prepare_data(x_mnist, y_mnist, lista)
                            unique_labels = np.unique([treino_y])
                            num_classifiers = unique_labels.size
                            explore(treino_x, treino_y, num_classifiers)
                            
                            if kernel == '1':
                                
                                start_time = time.time()
                                
                                score, predicted, model, title = perceptron_primal(treino_x, treino_y , teste_x, teste_y, classifier='ovo', max_iters=treino_x.shape[0])
                            
                                finish_time = time.time() - start_time
                                
                                print('Time OneVsOne - Primal: %.5fs' % finish_time)
                                print('> Dataset mnist')
                                print('> Build the results, please wait...')
                                
                                plot_erros(model, title + ' - Primal')
                                
                                confusion_matrix(teste_y, predicted, score, title + ' - Primal')
        
                                print('\n')
                            
                            else:
                                start_time = time.time()
                                
                                if kernel == '2':
                                    kernel_type = 'none'
                                elif kernel == '3':
                                    kernel_type = 'rbf'
                                elif kernel == '4':
                                    kernel_type = 'polynomial'
                                    
                                
                                score, predicted, model, title = perceptron_dual(treino_x, treino_y, teste_x, teste_y, classifier='ovo', max_iters=10, kernel=kernel_type)
                            
                                finish_time = time.time() - start_time
                                
                                print('Time OneVsOne - Dual - %s: %.5fs' % (kernel_type, finish_time))
                                print('> Dataset mnist')
                                print('> Build the results, please wait...')
                                
                                plot_erros(model, kernel_type)
                            
                                confusion_matrix(teste_y, predicted, score, title + ' - Dual - ' + kernel_type)
        
                                print('\n')
                            
                        else:
                            print("Error: Numbers size incorrect!")
                        
                    elif op_temp == '3':
                        # ECOC
                        lista = list(map(int, input("Enter the values (max 4): ").split()))
                        if len(lista) <= 4 and max(lista) <= 9 and len(lista) >= 2 and min(lista) >= 0:
                            treino_x, treino_y, teste_x, teste_y = prepare_data(x_mnist, y_mnist, lista)
                            unique_labels = np.unique([treino_y])
                            num_classifiers = unique_labels.size
                            explore(treino_x, treino_y, num_classifiers)
                            
                            if kernel == '1':
                                
                                start_time = time.time()
                                
                                score, predicted, model, title = perceptron_primal(treino_x, treino_y , teste_x, teste_y, classifier='ecoc', max_iters=treino_x.shape[0])
                            
                                finish_time = time.time() - start_time
                                
                                print('Time ECOC - Primal: %.5fs' % finish_time)
                                print('> Dataset mnist')
                                print('> Build the results, please wait...')
                            
                                plot_erros(model, title + ' - Primal')
                                
                                confusion_matrix(teste_y, predicted, score, title + ' - Primal')
        
                                print('\n')
                                
                            else:
                                
                                if kernel == '2':
                                    kernel_type = 'none'
                                elif kernel == '3':
                                    kernel_type = 'rbf'
                                elif kernel == '4':
                                    kernel_type = 'polynomial'
                                    
                                start_time = time.time()
                                
                                score, predicted, model, title = perceptron_dual(treino_x, treino_y , teste_x, teste_y, classifier='ecoc', max_iters=10, kernel=kernel_type)
                            
                                finish_time = time.time() - start_time
                                
                                print('Time ECOC - Dual - Kernel(%s): %.5fs' % (kernel_type, finish_time))
                                print('> Dataset mnist')
                                print('> Build the results, please wait...')
                            
                                confusion_matrix(teste_y, predicted, score, title + ' - Dual - Kernel(' + kernel_type + ')')
                                
                                plot_erros(model, title + ' - Dual - Kernel(' + kernel_type + ')')
        
                                print('\n')
                            
                        else:
                            print("Error: Numbers size incorrect!")
                    
                    elif op_temp == '0':
                        inside_loop = True
                    
                    else:
                       print("Error: Enter a valid number!")
                
            elif op == '0':
                finish = True
                
            else:
                print("Error: Enter a valid number!")
    
    elif kernel == '0':
        begin = True

    else:
        print("Error: Enter a valid number!")
 
     
print('\n> Bye bye, see you soon...')
