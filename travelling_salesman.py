import mlrose_hiive
from mlrose_hiive.algorithms import decay
import numpy as np
import time
import matplotlib.pyplot as plt

n_cities = 25
height = 250
width = 250
generator = mlrose_hiive.TSPGenerator()
problems = []
for i in range(30): # GA DID BEST
    problem = generator.generate(i, n_cities, height, width)
    problem.maximize = True
    problem.set_mimic_fast_mode(True)
    problems.append(problem)


# Randomized Hill Climbing
np.random.seed(0)
rhc_best = []
rhc_steps = []
rhc_plts = []
rhc_times = []
for problem in problems:
    start = time.time()
    rhc_best_state, rhc_best_fit, rhc_fits = mlrose_hiive.random_hill_climb(problem, max_attempts=100, max_iters=500, random_state=0, curve=True)
    end = time.time()
    rhc_best.append(rhc_best_fit)
    rhc_steps.append(len(rhc_fits))
    rhc_plts.append(rhc_fits)
    rhc_times.append(end-start)
rhc_time = np.mean(rhc_times)
rhc_score = np.mean(rhc_best)
print("RHC Score:")
print(rhc_score)
print("RHC Time:")
print(rhc_time)

rhc_pads = [np.pad(x, (0, 500 - len(x)), 'constant', constant_values=x[-1]) for x in rhc_plts]
rhc_plot = np.mean(np.vstack(rhc_pads), axis=0)

# Simulated Annealing
np.random.seed(0)
sa_scheds = []
for schedule in [decay.ExpDecay(exp_const=.001), decay.ExpDecay(exp_const=.01), decay.ExpDecay(), decay.GeomDecay(decay=.999), decay.GeomDecay(decay=.95),  decay.GeomDecay()]:
    best = []
    steps = []
    plts = []
    times = []
    for problem in problems:
        start = time.time()
        sa_best_state, sa_best_fit, sa_fits = mlrose_hiive.simulated_annealing(problem, max_attempts=100, max_iters=500, random_state=0, schedule=schedule, curve=True)
        end = time.time()
        best.append(sa_best_fit)
        steps.append(len(sa_fits))
        plts.append(sa_fits)
        times.append(end-start)
    sa_scheds.append([schedule, np.mean(best), np.mean(steps), plts, np.mean(times)])

best_sa_row = np.argmax([s[1] for s in sa_scheds])
best_sa = sa_scheds[best_sa_row]
sa_decay, sa_score, sa_length, sa_vals, sa_time = best_sa
print("SA Score:")
print(sa_score)
print("SA Time:")
print(sa_time)

sa_pads = [np.pad(x, (0, 500 - len(x)), 'constant', constant_values=x[-1]) for x in sa_vals]
sa_plot = np.mean(np.vstack(sa_pads), axis=0)



# Genetic Algorithm
np.random.seed(0)
ga_scores = []
for pop_breed_percent in [.25, .5, .75]:
    for mutation_prob in [.05, .2, .5]:
        best = []
        steps = []
        plts = []
        times = []
        for problem in problems:
            start = time.time()
            ga_best_state, ga_best_fit, ga_fits = mlrose_hiive.genetic_alg(problem, pop_size=25, max_attempts=10, max_iters=500, random_state=0, pop_breed_percent=pop_breed_percent, mutation_prob=mutation_prob, curve=True)
            end = time.time()
            best.append(ga_best_fit)
            steps.append(len(ga_fits))
            plts.append(ga_fits)
            times.append(end-start)
        ga_scores.append([pop_breed_percent, mutation_prob, np.mean(best), np.mean(steps), plts, np.mean(times)])
best_ga_row = np.argmax([s[2] for s in ga_scores])
best_ga = ga_scores[best_ga_row]
pop_breed_percent, mutation_prob, ga_score, ga_length, ga_vals, ga_time = best_ga
print("GA Score:")
print(ga_score)
print("GA Time:")
print(ga_time)

ga_pads = [np.pad(x, (0, 500 - len(x)), 'constant', constant_values=x[-1]) for x in ga_vals]
ga_plot = np.mean(np.vstack(ga_pads), axis=0)


# MIMIC
np.random.seed(0)
MIMIC_scores = []
for noise in [0, .05]:
    for keep_pct in [.2, .5]:
        best = []
        steps = []
        plts = []
        times = []
        for problem in problems:
            start = time.time()
            MIMIC_best_state, MIMIC_best_fit, MIMIC_fits = mlrose_hiive.mimic(problem, pop_size=20, keep_pct=keep_pct, noise=noise, max_attempts=10, max_iters=500,  curve=True, random_state=0)
            end = time.time()
            best.append(MIMIC_best_fit)
            steps.append(len(MIMIC_fits))
            plts.append(MIMIC_fits)
            times.append(end - start)
        MIMIC_scores.append([noise, keep_pct, np.mean(best), np.mean(steps), plts, np.mean(times)])

best_MIMIC_row = np.argmax([s[2] for s in MIMIC_scores])
best_MIMIC = MIMIC_scores[best_MIMIC_row]
noise, keep_pct, MIMIC_score, MIMIC_length, MIMIC_vals, MIMIC_time = best_MIMIC
print("MIMIC Score:")
print(MIMIC_score)
print("MIMIC Time:")
print(MIMIC_time)

MIMIC_pads = [np.pad(x, (0, 500 - len(x)), 'constant', constant_values=x[-1]) for x in MIMIC_vals]
MIMIC_plot = np.mean(np.vstack(MIMIC_pads), axis=0)

plt.plot(rhc_plot, label="Random Hill Climbing")
plt.plot(sa_plot, label="Simulated Annealing")
plt.plot(ga_plot, label="Genetic Algorithm")
plt.plot(MIMIC_plot, label="MIMIC")
plt.title("Travelling Salesman Fitness vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.legend()
plt.savefig("tsp_plt.png")

scheds = ['Exp: .01', 'Exp: .001', 'Exp: .005', 'Geom: .999', 'Geom: .95', 'Gemo: .99']

for i in range(6):
    sched = scheds[i]

    sa_decay, sa_score, sa_length, sa_vals, sa_time = sa_scheds[i]
    sa_pads = [np.pad(x, (0, 500 - len(x)), 'constant', constant_values=x[-1]) for x in sa_vals]
    sa_plot = np.mean(np.vstack(sa_pads), axis=0)

    plt.plot(sa_plot, label=sched)
    plt.title("Simulated Annealing Decay Schedule vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig('sa_hpt.png')

