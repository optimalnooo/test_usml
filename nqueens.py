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

    def __init__(self, pop_size=2000, cross_prob=1, mut_prob=0.7, tournament_size=3, crossover_points_size=None):
        if pop_size % 2:
            self.pop_size = pop_size + 1
        else:
            self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.tournament_size = tournament_size
        self.crossover_points_size = crossover_points_size
        self.best_fitness_value = 0
        self.best_individ = None
        self.population = self.get_start_population()

    def solve(self, min_fitness=1, max_epochs=20):
        if min_fitness is None:
            self.min_fitness = 2  #will not reach
        else:
            self.min_fitness = min_fitness

        epoch = 0
        while(True):
            if max_epochs is not None:
                if epoch >= max_epochs:
                    break
            epoch += 1
            weights, weights_sum = self.fitness_population()

            # check fitness condition
            if self.best_fitness_value >= self.min_fitness:
                return (self.best_fitness_value,
                        epoch,
                        self.get_individ_visualization(self.best_individ))

            #selection (get parents)
            parents1 = [
                self.get_selected_individual()
                for _ in range(self.pop_size//2)]
            parents2 = [
                self.get_selected_individual()
                for _ in range(self.pop_size//2)]

            #create new population
            self.population = []
            for parent1, parent2 in zip(parents1, parents2):
                new_ind1, new_ind2 = self.crossover(parent1, parent2)
                self.population.append(new_ind1)
                self.population.append(new_ind2)
            self.mutation()

        return (self.best_fitness_value,
                epoch,
                self.get_individ_visualization(self.best_individ))

    def get_selected_individual(self):
        '''Tournament selection.'''
        tour_list = [random.choice(self.population) for _ in range(self.tournament_size)]
        return max(tour_list, key=self.fitness_individ)
    
    def fitness_population(self):
        '''Fitness function for population.

        Return:
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
        '''Fitness function for individual.

        Return:
            weight - in range from 0 to 1
            the more weight the better
        '''
        individ = self.decode_individual(bit_individ)
        count = 0
        for i in range(Solver_8_queens.DIM_SIZE):
            for j in range(i+1, Solver_8_queens.DIM_SIZE):
                if self.check_pair_queens([i, individ[i]], [j, individ[j]]):
                    count += 1
        #update best weight
        weight = count / Solver_8_queens.ABS_MAX_FITNESS_VALUE;
        if weight > self.best_fitness_value:
            self.best_fitness_value = weight
            self.best_individ = bitarray(bit_individ)
        return weight
    
    def crossover(self, individ1, individ2):
        '''Crossover.

        If crossover_points_size
            - is None, then use uniform crossover
            - int number, then use multy-point crossover
        '''
        if self.crossover_points_size is None:
            mask = bitarray(endian='little')
            mask.frombytes(np.random.bytes(Solver_8_queens.GENE_SIZE))
        else:
            mask = bitarray(Solver_8_queens.DIM_SIZE
                                * Solver_8_queens.GENE_SIZE)
            mask.setall(False)
            for i in range(self.crossover_points_size):
                locus = np.random.randint(0,
                    Solver_8_queens.DIM_SIZE * Solver_8_queens.GENE_SIZE)
                mask[locus:] = ~mask[locus:]
        print(mask)
        new_individ1 = bitarray()
        new_individ2 = bitarray()
        for i in range(len(mask)):
            if mask[i]:
                new_individ1.append(individ1[i])
                new_individ2.append(individ2[i])
            else:
                new_individ1.append(individ2[i])
                new_individ2.append(individ1[i])
        return new_individ1, new_individ2

    def mutation(self):
        for individ in self.population:
            if np.random.rand() < self.mut_prob:
                locus = np.random.randint(0,
                    Solver_8_queens.DIM_SIZE * Solver_8_queens.GENE_SIZE)
                individ[locus] = not individ[locus]
    
    def check_pair_queens(self, q1, q2):
        '''Check intersection of two queens.

        Check only diagonal intersection.

        Return:
            True  - if queens doesn't intersect
            False - if queens intersect
        '''
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
        '''Decode individual.

        Get bitarray individual.

        Return list of positions in rows.
        '''
        occupied_cols = bitarray(8)
        occupied_cols.setall(False)
        current_cols = 0
        biases = [individ[Solver_8_queens.GENE_SIZE*i:
            Solver_8_queens.GENE_SIZE*(i+1)].tobytes()[0]
            for i in range(Solver_8_queens.DIM_SIZE)]
        decoded_individ = []
        for b in biases:
            current_cols += b
            current_cols %= Solver_8_queens.DIM_SIZE
            while True:
                if not occupied_cols[current_cols]:
                    decoded_individ.append(current_cols)
                    occupied_cols[current_cols] = True
                    break
                else:
                    current_cols += 1
                    current_cols %= Solver_8_queens.DIM_SIZE
        return decoded_individ
    
    def get_individ_visualization(self, individ):
        '''Individual string interpretation for vizualization.'''
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