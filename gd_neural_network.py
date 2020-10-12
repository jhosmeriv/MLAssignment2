import data_process
import mlrose_hiive
from mlrose_hiive.algorithms import decay
import numpy as np
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

X_train, X_val, X_test, y_train, y_val, y_test = data_process.get_titanic()


# Gradient Descent
gd_vals = []
for learning_rate in [.001, .0001, .00005]:
    nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes = [20, 20, 20, 20], activation = 'tanh',
                                    algorithm = 'gradient_descent',
                                    max_iters = 10000, bias = True, is_classifier = True,
                                    learning_rate = learning_rate, early_stopping = True,
                                    max_attempts = 100, random_state = 0, curve=True)

    nn_model.fit(X_train, y_train)
    y_test_pred = nn_model.predict(X_test)
    acc = accuracy_score(y_test, y_test_pred)
    print(acc)

    curve = np.log(-1* nn_model.fitness_curve)

    gd_vals.append([learning_rate, curve, acc])


plt.title("Gradient Descent (Backprop) Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Log Loss")
for x in gd_vals:
    plt.plot(x[1], label="Learning Rate: " + str(x[0]))
plt.legend()
plt.savefig("nn_gd_plt.png")

# Random Hill Climbing
rhc_vals = []
for restarts in [0, 1, 2]:
    nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes = [20, 20, 20, 20], activation = 'tanh',
                                          algorithm = 'random_hill_climb', restarts=restarts,
                                          max_iters = 20000, bias = True, is_classifier = True,
                                          learning_rate = 0.002, early_stopping = True,
                                          max_attempts = 100, random_state = 0, curve=True)

    nn_model.fit(X_train, y_train)
    y_test_pred = nn_model.predict(X_test)
    acc = accuracy_score(y_test, y_test_pred)
    print(acc)

    curve = nn_model.fitness_curve

    rhc_vals.append([restarts, curve, acc])


plt.title("Random Hill Climbing Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
for x in rhc_vals:
    plt.plot(x[1], label="Restarts: " + str(x[0]))
plt.legend()
plt.savefig("nn_rhc_plt.png")


# Simulated Annealing
sa_vals = []
for schedule in [decay.ExpDecay(exp_const=.01), decay.ExpDecay(), decay.GeomDecay(decay=.95),  decay.GeomDecay()]:
    nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes = [20, 20, 20, 20], activation = 'tanh',
                                          algorithm = 'simulated_annealing', schedule=schedule,
                                          max_iters = 20000, bias = True, is_classifier = True,
                                          learning_rate = 0.002, early_stopping = True,
                                          max_attempts = 100, random_state = 0, curve=True)

    nn_model.fit(X_train, y_train)
    y_test_pred = nn_model.predict(X_test)
    acc = accuracy_score(y_test, y_test_pred)
    print(acc)

    curve = nn_model.fitness_curve

    sa_vals.append([schedule, curve, acc])

scheds = ['Exp: .01', 'Exp: .005', 'Geom: .95', 'Geom: .99']

plt.title("Simulated Annealing Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
for x in range(4):
    plt.plot(sa_vals[x][1], label=scheds[x])
plt.legend()
plt.savefig("nn_sa_plt.png")


# Genetic Algorithm
ga_vals = []
for mutation_prob in [.02, .1, .2]:
    nn_model = mlrose_hiive.NeuralNetwork(hidden_nodes = [20, 20, 20, 20], activation = 'tanh',
                                          algorithm = 'genetic_alg', mutation_prob=mutation_prob, pop_size=20,
                                          max_iters = 20000, bias = True, is_classifier = True,
                                          learning_rate = 0.002, early_stopping = True,
                                          max_attempts = 1000, random_state = 0, curve=True)

    nn_model.fit(X_train, y_train)
    y_test_pred = nn_model.predict(X_test)
    acc = accuracy_score(y_test, y_test_pred)
    print(acc)

    curve = nn_model.fitness_curve

    ga_vals.append([mutation_prob , curve, acc])

plt.title("Genetic Algorithm Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
for x in ga_vals:
    plt.plot(x[1], label="Mutation Probability: " + str(x[0]))
plt.legend()
plt.savefig("nn_ga_plt.png")

