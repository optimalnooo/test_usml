# -*- coding: utf-8 -*-
import random
import time
import numpy as np
import pandas as pd
from bitarray import bitarray

class Solver_8_queens:

    DIM_SIZE = 8
    GENE_SIZE = 3
    ABS_MAX_FITNESS_VALUE = DIM_SIZE * (DIM_SIZE - 1) / 2

    def __init__(self, pop_size=2000, cross_prob=1, mut_prob=0.7):
        if pop_size % 2:
            self.pop_size = pop_size + 1
        else:
            self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.best_fitness_value = 0
        self.best_individ = None
        self.population = self.get_start_population()

    def solve(
        self,
        min_fitness=1,
        max_epochs=20
    ):
        self.min_fitness = min_fitness
        for epoch in range(1, max_epochs+1):
            weights, weights_sum = self.fitness_population()
            if self.best_fitness_value >= min_fitness:
                return (self.best_fitness_value,
                    epoch,
                    self.get_individ_visualization(self.best_individ))
            parents1 = [
                self.get_selected_individual(weights, weights_sum)
                for _ in range(self.pop_size//2)
            ]
            parents2 = [
                self.get_selected_individual(weights, weights_sum)
                for _ in range(self.pop_size//2)
            ]
            self.population = []
            for parent1, parent2 in zip(parents1, parents2):
                new_ind1, new_ind2 = self.crossover(parent1, parent2)
                self.population.append(new_ind1)
                self.population.append(new_ind2)
            self.mutation()
        return (self.best_fitness_value,
            epoch,
            self.get_individ_visualization(self.best_individ))

    def get_selected_individual(self, weights, weights_sum):
        '''roulette wheel'''
        rand_value = random.random() * weights_sum
        for i, w in enumerate(weights):
            rand_value -= w
            if rand_value <=0:
                return self.population[i]
    
    def fitness_population(self):
        '''fitness function for population
        return:
          float array - weights array for each individual
          float weights_sum - sum of weights
        '''
        weights = []
        for bit_individ in self.population:
            weight = self.fitness_individ(bit_individ)
            weights.append(weight)
            if weight >= self.min_fitness:
                return weights, None
        weights_sum = sum(weights) 
        return weights, weights_sum

    def fitness_individ(self, bit_individ):
        '''fitness function for individual
        return:
          weight - in range from 0 to 1
          the more weight the better'''
        individ = self.decode_individual(bit_individ)
        count = 0
        for i in range(Solver_8_queens.DIM_SIZE):
            for j in range(i+1, Solver_8_queens.DIM_SIZE):
                if self.check_pair_queens([i, individ[i]], [j, individ[j]]):
                    count += 1
        #update best weight
        weight = count / Solver_8_queens.ABS_MAX_FITNESS_VALUE;
        if weight >= self.best_fitness_value:
            self.best_fitness_value = weight
            self.best_individ = bit_individ
        return weight
    
    def crossover(self, indiv1, indiv2):
        if np.random.rand() < self.cross_prob:
            return indiv1, indiv2
        else:
            crossover_point = np.random.randint(
                0,
                Solver_8_queens.DIM_SIZE * Solver_8_queens.GENE_SIZE
            )
            new_indiv1 = bitarray(
                indiv1[:crossover_point]
                + indiv2[crossover_point:]
            )
            new_indiv2 = bitarray(
                indiv2[:crossover_point]
                + indiv1[crossover_point:]
            )
            return new_indiv1, new_indiv2

    def mutation(self):
        for individ in self.population:
            if np.random.rand() < self.mut_prob:
                locus = np.random.randint(0,
                    Solver_8_queens.DIM_SIZE*Solver_8_queens.GENE_SIZE)
                individ[locus] = not individ[locus]
    
    def check_pair_queens(self, q1, q2):
        '''check only diagonal intersection
        return:
          True  - if queens doesn't intersect
          False - if queens intersect'''
        if abs(q1[0] - q2[0])==abs(q1[1] - q2[1]):
            return False
        else:
            return True
    
    def get_start_population(self):
        return [self.get_random_individual() for _ in range(self.pop_size)]
    
    def get_random_individual(self):
        individ = bitarray(endian='little')
        individ.frombytes(np.random.bytes(Solver_8_queens.GENE_SIZE))
        return individ
    
    def decode_individual(self, individ):
        decoded_individ = []
        occupied_cols = [False for _ in range(Solver_8_queens.DIM_SIZE)]
        current_cols = 0
        biases = [int.from_bytes(
            individ[Solver_8_queens.GENE_SIZE*i:
                Solver_8_queens.GENE_SIZE*(i+1)].tobytes(),
            'little') for i in range(Solver_8_queens.DIM_SIZE)
        ]
        for i in range(Solver_8_queens.DIM_SIZE):
            current_cols += biases[i]
            current_cols %= Solver_8_queens.DIM_SIZE
            for _ in range(Solver_8_queens.DIM_SIZE):
                if not occupied_cols[current_cols]:
                    decoded_individ.append(current_cols)
                    occupied_cols[current_cols] = True
                    break
                else:
                    current_cols += 1
                    current_cols %= Solver_8_queens.DIM_SIZE
        return decoded_individ
    
    def get_individ_visualization(self, individ):
        '''individual string interpretation for vizualization'''
        display = [['+' for _ in range(Solver_8_queens.DIM_SIZE)]
            for _ in range(Solver_8_queens.DIM_SIZE)]
        decoded_individ = self.decode_individual(individ)
        for index, row in zip(decoded_individ, display):
            row[index] = 'Q'
        rows = [''.join(row) for row in display]
        viz = '\n'.join(rows)
        return viz

def tuning():
    df = []
    iters = 1000
    max_epochs = 100
    for pop_size in range(200, 3000, 100):
        print('Pop_size: {0}'.format(pop_size))
        for mut_prob in np.arange(0, 0.9, 0.1):
            print('Mut_prob: {0}'.format(mut_prob))
            errors = 0
            all_time = 0
            all_epochs = 0
            for iterat in range(iters):
                start = time.clock()
                solver = Solver_8_queens(pop_size, 1, mut_prob)
                best_fit, epoch, _ = solver.solve(Solver_8_queens.MAX_FITNESS_VALUE, max_epochs)
                finish = time.clock()
                all_time += finish - start
                all_epochs += epoch
                if best_fit < Solver_8_queens.MAX_FITNESS_VALUE:
                    errors += 1
            average_epoch = all_epochs / iters
            average_time = all_time / iters
            df.append([pop_size, mut_prob, max_epochs, errors, average_time, average_epoch])
    df = pd.DataFrame(df, columns=['pop_size', 'mut_prob', 'max_epochs', 'errors', 'average_time', 'average_epoch'])
    df.to_csv('tuning.csv')