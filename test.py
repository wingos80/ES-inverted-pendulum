import numpy as np
import matplotlib.pyplot as plt

class Individual_1:
    def __init__(self, num_dimensions=1):
        self.num_dimensions      = num_dimensions
        self.genotype            = None
        self.strategy_parameters = None
        self.fitness             = None
        self.unstable            = False

    def procreate(self, num_offsprings=1):
        epsilon = 10**(-5)
        offsprings = []
        z = np.random.normal(0,1/self.num_dimensions)
        for _ in range(num_offsprings):
            offspring = Individual_1(self.num_dimensions)
            offspring.genotype = np.array([self.genotype[i]+np.random.normal(0,self.strategy_parameters[0]) for i in range(self.num_dimensions)])
            zi = np.random.normal(0,1/2/np.sqrt(self.num_dimensions))
            offspring.strategy_parameters = np.array([max(sigma*np.exp(z+zi), epsilon) for sigma in self.strategy_parameters])
            offsprings.append(offspring)
        return offsprings
    
class ES_1:
    def __init__(self, fitness_function=None, num_dimensions=1, num_generations=100, num_individuals=50, num_offspring_per_individual=5, verbose=False):
        self.fitness_function = fitness_function
        self.num_dimensions   = num_dimensions
        self.num_generations  = num_generations
        self.num_individuals  = num_individuals
        self.num_offspring_per_individual = num_offspring_per_individual
        self.verbose          = verbose
        self.noconfidence_stretch = 0

        assert fitness_function is not None, "Fitness function needs to be defined"


    def run(self):
        population = [self.generate_random_individual() for _ in range(self.num_individuals)]
        best_individual = sorted(population, key=lambda individual: self.fitness_function(individual))[0]

        for generation in range(self.num_generations):
            offsprings = []
            for parent in population:
                offsprings += parent.procreate(self.num_offspring_per_individual)

            # print(f'size of pop: {len(population)}')
            population += offsprings
            # print(f'size of pop after: {len(population)}\n')
            population = sorted(population, key=lambda individual: self.fitness_function(individual))[:self.num_individuals]
            best = population[0]

            # if best individual is unstable, increment noconfidence_stretch
            if best.unstable:
                if self.verbose: print(f'    best individual is unstable')
                self.noconfidence_stretch += 1
            # if best individual is stable, reset noconfidence_stretch
            else:
                if self.verbose: print(f'    best individual is stable????')
                self.noconfidence_stretch = 0
            
            # if noconfidence_stretch is too large, reset every individual's strategy parameters
            if self.noconfidence_stretch > 10:
                if self.verbose: print(f'    noconfidence_stretch is too large, resetting all strategy parameters\n\n')
                for individual in population:
                    individual.strategy_parameters = np.array([np.maximum(np.random.normal(2,5), 0.01)])
                self.noconfidence_stretch = 0

            if self.verbose:
                if self.verbose: print(f"[gen {generation:3}] Best fitness: {self.fitness_function(best)}")


        return self.fitness_function(best)
    
    def generate_random_individual(self):
        # --- Initialize the population here ---
        # - For the genotype, sample a standard random normal distribution for each variable separately
        # - For the strategy parameter, sample a standard random normal distribution and then take the maximum of that sample and 0.1 
        #   (to ensure it is not negative and not too small for exploration)
        ind = Individual_1(self.num_dimensions)
        ind.genotype = np.random.uniform(-15,15, size=self.num_dimensions)
        ind.strategy_parameters = np.array([np.maximum(np.random.normal(2,5), 0.01)])
        return ind


class Individual:
    def __init__(self, genotype, strategy_parameters):
        self.genotype = genotype
        self.strategy_parameters = strategy_parameters

class ES:
    def __init__(self, fitness_function=lambda x: 0, num_dimensions=1, 
                 num_generations=100, num_individuals=50, 
                 num_offspring_per_individual=5, verbose=False):
        self.fitness_function = fitness_function
        self.num_dimensions = num_dimensions
        self.num_generations = num_generations
        self.num_individuals = num_individuals
        self.num_offspring_per_individual = num_offspring_per_individual
        self.verbose = verbose
        
        assert num_individuals % 2 == 0, "Population size needs to be divisible by 2 for cross-over"
    
    def run(self):
        population = [self.generate_random_individual() for _ in range(self.num_individuals)]
        best = sorted(population, key=lambda individual: self.fitness_function(individual))[0]
        print(population[1])
        for generation in range(self.num_generations):
            # --- Perform mutation and selection here ---
            # - Each parent individual should produce `num_offspring_per_individual` children by mutation
            #   (recombination is ignored for this exercise)
            # - Implement P+O (parent+offspring) with truncation selection (picking the best n individuals)
            # - Update the `best` variable to hold the best individual of this generation (to then be printed below)

            offsprings = []
            for parent in population:
                for _ in range(self.num_offspring_per_individual):
                    parent_genotype = parent.genotype
                    parent_strategy_parameter = parent.strategy_parameters[0]
                    new_genotype = np.array([parent_genotype[i]+np.random.normal(0,parent_strategy_parameter) for i in range(self.num_dimensions)])
                    new_strategy_parameters = np.array([max(parent_strategy_parameter*np.exp(np.random.normal(0,1/self.num_dimensions)),10**(-6))])
                    offsprings.append(Individual(new_genotype, new_strategy_parameters))
            population += offsprings
            population = sorted(population, key=lambda individual: self.fitness_function(individual))[:self.num_individuals]
            best = population[0]

            if self.verbose: print(f"[gen {generation:3}] Best fitness: {self.fitness_function(best)}")

        return self.fitness_function(best)
    
    def generate_random_individual(self):
        # --- Initialize the population here ---
        # - For the genotype, sample a standard random normal distribution for each variable separately
        # - For the strategy parameter, sample a standard random normal distribution and then take the maximum of that sample and 0.1 
        #   (to ensure it is not negative and not too small for exploration)
        return Individual(np.array([np.random.uniform(-15,15) for _ in range(self.num_dimensions)]), np.array([max(np.random.normal(2,5), 0.1)]))


def f(individual):
    x = individual.genotype
    return np.sum(x**2)

dims = 10
D = np.diag(np.random.uniform(1,2, size=dims)**(-2))
Q = np.random.uniform(-1,1, size=(dims,dims))

def re(individual):
    A = Q@D@Q.T
    x = individual.genotype
    return x.T@A@x

es = ES(fitness_function=re, num_dimensions=dims, num_generations=100, num_individuals=50, num_offspring_per_individual=5, verbose=False)
print(es.run(), "ehehehe")
es_1 = ES_1(fitness_function=re, num_dimensions=dims, num_generations=100, num_individuals=50, num_offspring_per_individual=5, verbose=False)
print(es_1.run())
