# -*- coding: utf-8 -*-
import random
import numpy as np

class Solver_8_queens:

    DIM_SIZE = 8
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
                    if self.check_pair_chromosomes([i, individual[i]], [j, individual[j]]):
                        count += 1
            if count == (Solver_8_queens.DIM_SIZE * (Solver_8_queens.DIM_SIZE - 1) / 2):
                Solver_8_queens.RESULT += 1
                #Solver_8_queens.FOUND_DECISION = True
                #return None, None
            weights.append(count)
        weights_sum = sum(weights) 
        return weights, weights_sum

    def crossover(self, indiv1, indiv2):
        crossover_point = np.random.randint(0, Solver_8_queens.DIM_SIZE)
        new_indiv1 = np.concatenate((
            indiv1[:crossover_point],
            indiv2[crossover_point:]
        ))
        new_indiv2 = np.concatenate((
            indiv2[:crossover_point],
            indiv1[crossover_point:]
        ))
        return new_indiv1, new_indiv2

    def check_pair_chromosomes(self, chrm1, chrm2):
        '''check chromosomes
        return:
          True  - if chromosomes not intersect
          False - if chromosomes intersect
        '''
        bias1 = chrm1[0] - chrm2[0]
        bias2 = chrm1[1] - chrm2[1]

        if bias1==0 or bias2==0 or abs(bias1)==abs(bias2):
            return False
        else:
            return True

    def get_start_population(self, size):
        return np.array([self.get_random_individual() for _ in range(size)])

    def get_random_individual(self):
        return np.random.permutation(Solver_8_queens.DIM_SIZE)

    def show_individual(self, individual):
        '''individual vizualization'''
        display = np.empty((Solver_8_queens.DIM_SIZE, Solver_8_queens.DIM_SIZE), dtype='str')
        display.fill('+')
        for index, row in zip(individual, display):
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
