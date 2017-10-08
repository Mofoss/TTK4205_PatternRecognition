import numpy as np
import matplotlib.pyplot as plt
import itertools
from ast import literal_eval
import copy
from mpl_toolkits.mplot3d import Axes3D

N_CLASSES = 2


# Reads data from files
def load_data(file_name):
    # Første element i hver linje angir klassetilhørigheten (verdien 1 eller 2) og de øvrige elementene i linjen
    # angir egenskapsverdiene knyttet til det samme objektet.
    file = open(file_name, 'r').readlines()
    global WIDTH, LENGTH
    LENGTH = len(file) // N_CLASSES
    WIDTH = len(file[0].split())

    # Treningssettet består av odde nummererte objekter, testsettet av de med partall
    train = np.zeros((LENGTH, WIDTH), dtype=float)
    test = np.zeros((LENGTH, WIDTH + 1), dtype=float)

    for row in range(0, LENGTH):
        for col in range(WIDTH):
            train[row][col] = float(file[row * 2].split()[col])
            test[row][col] = float(file[row * 2 + 1].split()[col])

    return train, test


# Estimates the error rate on test data
def find_error_rate(test_data):
    fail = 0
    for row in range(LENGTH):
        fail += test_data[row][-1] != test_data[row][0]
    return fail


def results_to_file(task, header, results):
    res = open(''.join(['results_', str(task), '_', file_name.split('.')[0], '.txt']), 'w')
    res.writelines(header)
    res.writelines('\n'.join('\t'.join(str(column) for column in line) for line in results))


# NN = Nearest Neighbour method
def NN(train, test):
    combination_matrix = list(itertools.product([0, 1], repeat=(WIDTH - 1)))[1:]
    error_rates = []

    for combination in combination_matrix:
        for test_row in test:
            min_d = 9999999
            for train_row in train:
                d = abs(np.linalg.norm(test_row[1:-1] * combination - train_row[1:] * combination))
                if d < min_d:
                    min_d = d
                    test_row[-1] = train_row[0]  # assign estimated class

        # Estimate error rate
        error = find_error_rate(test)
        error_rates.append((sum(combination), combination, "{0:.3f}".format(round(error / LENGTH, 3))))

    error_rates.sort(key=lambda x: (x[0], x[2]))
    header = '#features\tSelected features\tFail rate\n'
    results_to_file(1, header, error_rates)


def get_combinations(file_name, data_set):
    combinations_file = open(file_name, 'r').read()
    combinations = []
    for row in combinations_file.split('\n')[1:]:  # separate each line and exclude header line
        row = row.split('\t')
        if row[0] != str(data_set):
            continue
        combinations.append(literal_eval(row[2]))
    return combinations


# Discriminant function g(x)
def discriminant_function(x, W, w, w_0, i):
    return np.matmul(np.matmul(x.T, W[i]), x) + np.matmul(w[i].T, x) + w_0[i]


def min_error_rate(training_data, test_data, data_set):

    # We shall use the best feature combinations from exersise 1
    combinations = get_combinations('best_feature_combinations.txt', data_set)
    error_rates = []

    for combo in combinations:
        temp_training_data = training_data
        temp_test_data = test_data
        delete_columns = []
        for feature, include in enumerate(combo):
            if include == 0:
                delete_columns.append(1+feature)
        temp_training_data = np.delete(temp_training_data, delete_columns, axis=1)
        temp_test_data = np.delete(temp_test_data, delete_columns, axis=1)

        # Find properties for each class
        train_class, apriori, mean, inv_cov_matrix, W, w, w_0 = [], [], [], [], [], [], []
        for c in range(N_CLASSES):
            train_class.append(temp_training_data[training_data[:, 0] == c + 1])
            apriori.append(len(temp_training_data[training_data[:, 0] == c + 1]) / LENGTH)
            mean.append(train_class[c][:, 1:].mean(axis=0))  # finds the mean of the training set with the given class
            cov_matrix = np.cov(train_class[c][:, 1:], rowvar=False)  # finds the covariance matrix of the training set
            inv_cov_matrix.append(np.linalg.inv(np.atleast_2d(cov_matrix)))
            W.append(-0.5 * inv_cov_matrix[c])
            w.append(np.matmul(inv_cov_matrix[c], mean[c]))
            w_0.append(-0.5 * np.matmul(np.matmul(mean[c].T, inv_cov_matrix[c]), mean[c]) -
                       0.5 * np.log(np.linalg.det(np.atleast_2d(cov_matrix))) + np.log(apriori[c]))

        # Run discriminant function for every feature vector in the test data
        for x in temp_test_data:
            g1 = discriminant_function(x[1:-1], W, w, w_0, 0)
            g2 = discriminant_function(x[1:-1], W, w, w_0, 1)
            g = g1 - g2
            if g > 0:
                x[-1] = 1  # assign estimated class
            else:
                x[-1] = 2

        error = find_error_rate(temp_test_data)
        error_rates.append((sum(combo), combo, "{0:.3f}".format(round(error / LENGTH, 3))))

    error_rates.sort(key=lambda x: (x[0], x[2]))
    header = '#features\tSelected features\tFail rate\n'
    results_to_file('2a', header, error_rates)


# minimum squared error
def MSE(training_data, test_data, data_set):

    # We shall use the best feature combinations from exersise 1
    combinations = get_combinations('best_feature_combinations.txt', data_set)
    error_rates = []

    for combo in combinations:
        temp_training_data = training_data
        temp_test_data = test_data
        delete_columns = []
        for feature, include in enumerate(combo):
            if include == 0:
                delete_columns.append(1+feature)
        temp_training_data = np.delete(temp_training_data, delete_columns, axis=1)
        temp_test_data = np.delete(temp_test_data, delete_columns, axis=1)

        # Expand training matrix with one column
        train_with_bias = np.ones((len(temp_training_data), len(temp_training_data[0]) + 1))
        train_with_bias[:, 0] = temp_training_data[:, 0]
        train_with_bias[:, 2:] = temp_training_data[:, 1:]

        # Create vector b where b[i] is 1 for class 1 and b[i] is -1 for all other classes, for all i
        b = train_with_bias[:, [0, 1]]
        b[1][b[1] == 1] = 1
        b[1][b[1] != 1] = -1

        # Create weigth vector

        # Run discriminant function for every feature vector in the test data
        for x in temp_test_data:
            g1 = discriminant_function(x[1:-1], W, w, w_0, 0)
            g2 = discriminant_function(x[1:-1], W, w, w_0, 1)
            g = g1 - g2
            if g > 0:
                x[-1] = 1  # assign estimated class
            else:
                x[-1] = 2

        error = find_error_rate(temp_test_data)
        error_rates.append((sum(combo), combo, "{0:.3f}".format(round(error / LENGTH, 3))))

    error_rates.sort(key=lambda x: (x[0], x[2]))
    header = '#features\tSelected features\tFail rate\n'
    results_to_file('2a', header, error_rates)

# ****************************

if __name__ == '__main__':
    file_name = 'ds-1.txt'  # syntetisk, 300 objekter med 4 egenskaper
    # file_name = 'ds-2.txt'    # syntetisk, 300 objekter med 3 egenskaper
    # file_name = 'ds-3.txt'  # generert ved uttrekking, 400 objekter med 4 egenskaper
    data_set = int(file_name.split('.')[0][-1])     # chosen data_set

    train, test = load_data(file_name)

    # Plot training set with correct classes
    # plt.scatter(train[:, 1], train[:, 2], c=train[:, 0])

    # NN(train, test)
    # min_error_rate(train, test, data_set)
    MSE(train, test, data_set)

    # Plot test set with estimated classes
    plt.scatter(test[:, 1], test[:, 2], c=test[:, -1])

    # Plot test set with estimated classes in 3D
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(train[:, 1], train[:, 2], test[:, 3], c=test[:, -1])

    # plt.show()
