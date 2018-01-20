# -*- coding: utf-8 -*-
import random
import numpy as np

class Solver_8_queens:

    DIM_SIZE = 8

    def __init__(self, pop_size=100, cross_prob=0.5, mut_prob=0.25):
        pass

    def solve(self, min_fitness=0.9, max_epochs=100):
        pass

    def get_selected_individual(self, population):
        '''roulette wheel'''
        weights = [self.fitness(individ) for individ in population]
        weights_sum = sum(weights)
        rand_value = random.random() * weights_sum

        for i, w in enumerate(weights):
            rand_value -= w
            if rand_value <=0:
                return population[i]

    def fitness(self, individual):
        '''fitness function
        return:
          int value - from 0 to DIM_SIZE * (DIM_SIZE - 1) / 2
          the more the better
        '''
        count = 0
        for i in range(Solver_8_queens.DIM_SIZE):
            for j in range(i+1, Solver_8_queens.DIM_SIZE):
                if self.check_pair_chromosomes(individual[i], individual[j]):
                    count += 1
        if count == (Solver_8_queens.DIM_SIZE * (Solver_8_queens.DIM_SIZE - 1) / 2):
            self.show_individual(individual)
        return count

    def crossover(self, indiv1, indiv2):
        crossover_point = np.random.randint(0, 8)
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

    indiv1 = q.get_random_individual()
    indiv2 = q.get_random_individual()
    new1, new2 = q.crossover(indiv1, indiv2)

    print(indiv1)
    print(indiv2)
    print()
    print(new1)
    print(new2)

    #q->solver
    #best_fit, epoch_num, visualization = solver.solve()
