import numpy as np
import matplotlib.pyplot as plt
import itertools

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


# Beregner feilrate
def fail_rate(training_data, test_data):
    fail = 0
    for row in range(len(training_data)):
        fail += test_data[row][-1] != test_data[row][0]
    return fail


def results_to_file(task, header, results):
    res = open(''.join(['results_', str(task), '_', file_name.split('.')[0], '.txt']), 'w')
    res.writelines(header)
    res.writelines('\n'.join('\t'.join(str(column) for column in line) for line in results))


# NN = Nearest Neighbour method
def NN(train, test):
    combination_matrix = list(itertools.product([0, 1], repeat=(WIDTH - 1)))[1:]
    fail_rates = []

    for combination in combination_matrix:
        for test_row in test:
            min_d = 9999999
            for train_row in train:
                d = abs(np.linalg.norm(test_row[1:-1] * combination - train_row[1:] * combination))
                if d < min_d:
                    min_d = d
                    test_row[-1] = train_row[0]  # assign estimated class

        # Beregner feilrate
        fail = fail_rate(train, test)
        fail_rates.append((sum(combination), combination, "{0:.3f}".format(round(fail / LENGTH, 3))))

    fail_rates.sort(key=lambda x: (x[0], x[2]))
    header = '#features\tSelected features\tFail rate\n'
    results_to_file(1, header, fail_rates)


# Discriminant function g(x)
def discriminant_function(x, W, w, w_0):
    return np.matmul(np.matmul(x.T, W[0]), x) + np.matmul(w[0].T, x) + w_0[0] - \
            np.matmul(np.matmul(x.T, W[1]), x) + np.matmul(w[1].T, x) + w_0[1]


def min_ER(train, test):
    """ Minimum error rate. """

    # Find properties for each class
    train_class, apriori, mean, inv_cov_matrix, W, w, w_0 = [], [], [], [], [], [], []
    for c in range(N_CLASSES):
        train_class.append(train[train[:, 0] == c + 1])
        apriori.append(len(train[train[:, 0] == c + 1]) / LENGTH)
        mean.append(train_class[c][:, 1:].mean(axis=0))  # finds the mean of the training set with the given class
        inv_cov_matrix.append(np.linalg.inv(
            np.cov(train_class[c][:, 1:], rowvar=False)))  # finds the inverted covariance matrix of the training set

        W.append(-0.5 * inv_cov_matrix[c])
        w.append(np.matmul(inv_cov_matrix[c], mean[c]))
        w_0.append(-0.5 * np.matmul(np.matmul(mean[c].T, inv_cov_matrix[c]), mean[c]) -
                   0.5 * np.log(np.linalg.det(inv_cov_matrix[c])) + np.log(apriori[c]))

    for x in test:
        g = discriminant_function(x[1:-1], W, w, w_0)
        if g > 0:
            x[-1] = 1  # assign estimated class
        else:
            x[-1] = 2

    fail_rates = fail_rate(train, test)
    print(fail_rates)
    # header = '#features\tSelected features\tFail rate\n'


if __name__ == '__main__':
    file_name = 'ds-1.txt'  # syntetisk, 300 objekter med 4 egenskaper
    # file_name = 'ds-2.txt'    # syntetisk, 300 objekter med 3 egenskaper
    # file_name = 'ds-3.txt'    # generert ved uttrekking av formegenskaper fra segmenter av to ulike bilmodeller,
    # 400 objekter med 4 egenskaper

    train, test = load_data(file_name)

    # Plot training set
    plt.scatter(train[:, 1], train[:, 2], c=train[:, 0])

    # NN(train, test)
    min_ER(train, test)

    # plt.show()