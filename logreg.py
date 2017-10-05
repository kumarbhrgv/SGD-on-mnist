import random
import argparse
import gzip
import pickle

import numpy as np
import matplotlib.pyplot as plt
from math import exp, log
from collections import defaultdict


SEED = 1735

random.seed(SEED)


class Numbers:
    """
    Class to store MNIST data for images of 0 and 1 only
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if you'd like

        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)

        self.train_x, self.train_y = train_set
        train_indices = np.where(self.train_y > 7)
        self.train_x, self.train_y = self.train_x[train_indices], self.train_y[train_indices]
        self.train_y = self.train_y - 8

        self.valid_x, self.valid_y = valid_set
        valid_indices = np.where(self.valid_y > 7)
        self.valid_x, self.valid_y = self.valid_x[valid_indices], self.valid_y[valid_indices]
        self.valid_y = self.valid_y - 8

        self.test_x, self.test_y = test_set
        test_indices = np.where(self.test_y > 7)
        self.test_x, self.test_y = self.test_x[test_indices], self.test_y[test_indices]
        self.test_y = self.test_y - 8

    @staticmethod
    def shuffle(X, y):
        """ Shuffle training data """
        shuffled_indices = np.random.permutation(len(y))
        return X[shuffled_indices], y[shuffled_indices]


class LogReg:
    def __init__(self, num_features, eta):
        """
        Create a logistic regression classifier
        :param num_features: The number of features (including bias)
        :param eta: A function that takes the iteration as an argument (the default is a constant value)
        """

        self.w = np.zeros(num_features)
        self.bias = 0
        self.eta = eta

    def progress(self, examples_x, examples_y):
        """
        Given a set of examples, compute the probability and accuracy
        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for x_i, y in zip(examples_x, examples_y):
            p = sigmoid(self.w.dot(x_i))
            if y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if abs(y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples_y))

    def sgd_update(self, x_i, y,iteration=None):
        """
        Compute a stochastic gradient update to improve the log likelihood.
        :param x_i: The features of the example to take the gradient with respect to
        :param y: The target output of the example to take the gradient with respect to
        :return: Return the new value of the regression coefficients
        """
        #1. Initialize a vector β to be all zeros
        #2. For t = 1, . . . , T
        '''◦ For each example xi, yi and feature j:
            • Compute πi ≡ Pr(yi = 1 | x_i)
            • Set βj = βj + η(yi − πi)xi
        3. Output the parameters β1, . . . , βd.
        '''


        #using dynamic eta if nothing is provided else default to eta set in 
        pi = exp(self.w.dot(x_i))/ (1+ exp(self.w.dot(x_i)))
        x_ibias = 1
        n = self.eta
        
        if iteration is not None:
            n = self.eta(iteration)
        
        learning = n*(y -pi)
        self.bias = self.bias + learning* x_ibias
        for i in range(len(self.w)):
            self.w[i] = self.w[i] + learning*x_i[i]
        return self.w

def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.
    :param score: A real valued number to convert into a number between 0 and 1
    """
    if abs(score) > threshold:
        score = threshold * np.sign(score)
    return 1 / (1 + exp(-score))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--eta", help="Initial SGD learning rate",
                           type=float, default=-1, required=False)
    argparser.add_argument("--passes", help="Number of passes through training data",
                           type=int, default=50, required=False)

    args = argparser.parse_args()

    data = Numbers('../data/mnist.pkl.gz')
    constant =1
    if args.eta > 0:
        print("Using learning rate as :", args.eta)
        eta = lambda  iter: args.eta
    else:
        eta  = lambda arg1: constant / (constant+ arg1)
    # Initialize model
    lr = LogReg(data.train_x.shape[1], eta)
    # Iterations
    
    #dynamic eta = constant/(constant+number of iterations)
    accuracy_array = []
    epoch_array = range(args.passes)
    for epoch in range(args.passes):
        iteration = 0
        data.train_x, data.train_y = Numbers.shuffle(data.train_x, data.train_y)
        for i in range(len(data.train_x)):#(len(data.train_x)):
            lr.sgd_update(data.train_x[i],data.train_y[i],iteration)
            iteration+=1
        testprobability, testaccuracy = lr.progress(data.test_x,data.test_y)
        trainprobability, trainaccuracy = lr.progress(data.train_x,data.train_y)
        print("test accuracy for pass : ", epoch ," is ", testaccuracy)
        print("train accuracy for pass : ", epoch ," = ", trainaccuracy)
        accuracy_array.append(1-testaccuracy)
    plt.plot(epoch_array,accuracy_array,'bo')
    plt.ylabel('Error rate')
    plt.xlabel('number of epoch')
    plt.show()


