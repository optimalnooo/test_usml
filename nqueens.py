# -*- coding: utf-8 -*-
import random
import numpy as np
from bitarray import bitarray

class Solver_8_queens:

    DIM_SIZE = 8
    GENE_SIZE = 3

    FOUND_DECISION = False
    RESULT = 0

    def __init__(self, pop_size=100, cross_prob=0.5, mut_prob=0.25):
        pass

    def solve(self, min_fitness=0.9, max_epochs=100):
        pass

    def get_selected_individual(self, population, weights, weights_sum):
        '''roulette wheel'''
        rand_value = random.random() * weights_sum
        for i, w in enumerate(weights):
            rand_value -= w
            if rand_value <=0:
                return population[i]
    
    def fitness_population(self, population):
        '''fitness function
        return:
          int array - weights array for each individual
          int weights_sum - sum of weights

          weight in range from 0 to DIM_SIZE * (DIM_SIZE - 1) / 2
          the more weight the better
        '''
        weights = []
        for individual in population:
            count = 0
            for i in range(Solver_8_queens.DIM_SIZE):
                for j in range(i+1, Solver_8_queens.DIM_SIZE):
                    if self.check_pair_queens([i, individual[i]], [j, individual[j]]):
                        count += 1
            if count == (Solver_8_queens.DIM_SIZE * (Solver_8_queens.DIM_SIZE - 1) / 2):
                Solver_8_queens.RESULT += 1
                #Solver_8_queens.FOUND_DECISION = True
                #return None, None
            weights.append(count)
        weights_sum = sum(weights) 
        return weights, weights_sum
    
    def crossover(self, indiv1, indiv2):
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
        individual = bitarray(endian='little')
        individual.frombytes(np.random.bytes(Solver_8_queens.GENE_SIZE))
        return individual
    
    def decode_individual(self, individual):
        decoded_individual = []
        occupied_cols = [False for _ in range(Solver_8_queens.DIM_SIZE)]
        current_cols = 0
        biases = [int.from_bytes(
            individual[Solver_8_queens.GENE_SIZE*i:
                Solver_8_queens.GENE_SIZE*(i+1)].tobytes(),
            'little') for i in range(Solver_8_queens.DIM_SIZE)
        ]
        for i in range(Solver_8_queens.DIM_SIZE):
            current_cols += biases[i]
            current_cols %= Solver_8_queens.DIM_SIZE
            for _ in range(Solver_8_queens.DIM_SIZE):
                if not occupied_cols[current_cols]:
                    decoded_individual.append(current_cols)
                    occupied_cols[current_cols] = True
                    break
                else:
                    current_cols += 1
                    current_cols %= Solver_8_queens.DIM_SIZE
        return decoded_individual
    
    def show_individual(self, individual):
        '''individual vizualization'''
        display = np.empty(
            (Solver_8_queens.DIM_SIZE, Solver_8_queens.DIM_SIZE),
            dtype='str'
        )
        display.fill('+')
        decoded_individual = self.decode_individual(individual)
        for index, row in zip(decoded_individual, display):
            row[index] = 'Q'
            for el in row:
                print(el, end='')
            print()

def main():
    
    q = Solver_8_queens()

    ar = []
    for i in range(10000):
        indiv = q.get_random_individual()
        ar.append(indiv)
    print('Start population was created')
    new_ar = []
    weights, weights_sum = q.fitness_population(ar)

    for i in range(10000):
        ind1 = q.get_selected_individual(ar, weights, weights_sum)
        ind2 = q.get_selected_individual(ar, weights, weights_sum)
        a,b = q.crossover(ind1, ind2)
        new_ar.append(a)
        new_ar.append(b)
        if i % 1000 == 0:
            print(i)
    print("Was success: ", end='')
    print(Solver_8_queens.RESULT)
    Solver_8_queens.RESULT = 0

    print("Now success: ", end='')
    q.fitness_population(new_ar)
    print(Solver_8_queens.RESULT)

    #q->solver
    #best_fit, epoch_num, visualization = solver.solve()
