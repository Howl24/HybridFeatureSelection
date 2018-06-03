import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from imblearn.metrics import geometric_mean_score
import matplotlib.pyplot as plt

gmean_score = make_scorer(geometric_mean_score)


class Individual(object):
    def __init__(self, dna):
        self.dna = dna
        self.fitness = self.evaluate_fitness(dna)

    @classmethod
    def Random(cls):
        dna = np.random.choice([True, False], size=cls.dna_size)
        return cls(dna)

    def print(self):
        print("G-mean: ", self.fitness)

    @classmethod
    def Crossover(cls, ind1, ind2):
        mask = np.random.choice([True, False], size=cls.dna_size)
        new_dna1 = np.logical_or(np.logical_and(ind1.dna, mask),
                                 np.logical_and(ind2.dna, ~mask))

        new_dna2 = np.logical_or(np.logical_and(ind2.dna, mask),
                                 np.logical_and(ind1.dna, ~mask))

        return Individual(new_dna1), Individual(new_dna2)

    def evaluate_fitness(self, dna):
        selected_features = self.features[dna]
        X_selected = self.X[:, selected_features]
        acc = np.mean(cross_val_score(self.model, X_selected, self.y,
                                      scoring=gmean_score))
        return acc

    def mutate(self):
        pos = np.random.randint(self.dna_size)
        self.dna[pos] = not self.dna[pos]
        self.fitness = self.evaluate_fitness(self.dna)

    @classmethod
    def Configure(cls, features, X, y, model):
        cls.features = features
        cls.dna_size = len(features)
        cls.X = X
        cls.y = y
        cls.model = model


class GeneticAlgorithm(object):

    def __init__(self, population_size=30, mutation_rate=0.5,
                 tournament_size=5, generations=20):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.generations = generations
        self.current_generation = 0

    def random_population(self):
        self.population = []
        for i in range(self.population_size):
            self.population.append(Individual.Random())

    def selection(self):
        """Tournament parent selection."""
        participants = np.random.choice(self.population,
                                        size=self.tournament_size,
                                        replace=False)  # no repeat

        return sorted(participants, key=lambda i: i.fitness, reverse=True)[:2]

    def mutation(self, individual):
        if np.random.choice([True, False], size=1,
                            p=[self.mutation_rate, 1-self.mutation_rate]):
            individual.mutate()
        return individual

    def new_gen_selection(self, new_population):
        all_pop = self.population + new_population
        all_pop = sorted(all_pop, key=lambda i: i.fitness, reverse=True)
        return all_pop[: self.population_size]

    def stop_condition(self):
        return self.current_generation >= self.generations

    def run(self, plot=True):
        results_by_gen = []

        self.random_population()
        while not self.stop_condition():
            self.current_generation += 1

            new_population = []
            for i in range(self.population_size//2):
                ind1, ind2 = self.selection()
                offspring1, offspring2 = Individual.Crossover(ind1, ind2)

                self.mutation(offspring1)
                self.mutation(offspring2)

                new_population.append(offspring1)
                new_population.append(offspring2)

            self.population = self.new_gen_selection(new_population)
            results_by_gen.append(self.population[0])

        plt.plot(range(1, self.generations+1),
                 [r.fitness for r in results_by_gen])
