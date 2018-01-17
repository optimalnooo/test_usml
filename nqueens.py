# -*- coding: utf-8 -*-
import random
import numpy as np

class Solver_8_queens:

    DIM_SIZE = 8
    DIMENSION = 2

    def __init__(self, pop_size=100, cross_prob=0.5, mut_prob=0.25):
        pass

    def solve(self, min_fitness=0.9, max_epochs=100):
        pass

    def create_start_population(self):
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
        return count


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

    def get_init_population(self, size):
        population = [self.get_random_individual() for _ in range(size)]
        return population

    def get_random_individual(self):
        individual = [self.get_random_chromosome()
            for _ in range(Solver_8_queens.DIM_SIZE)]
        return individual

    def get_random_chromosome(self):
        chromosome = random.choices(
            range(Solver_8_queens.DIM_SIZE),
            k=Solver_8_queens.DIMENSION
        )
        return chromosome

    def show_individual(self, individual):
        '''individual vizualization'''
        display = [['+' for _ in range(Solver_8_queens.DIM_SIZE)]
            for _ in range(Solver_8_queens.DIM_SIZE)
        ]
        for chrm in individual:
            display[chrm[0]][chrm[1]] = 'Q'
        #show
        for i in range(Solver_8_queens.DIM_SIZE):
            for j in range(Solver_8_queens.DIM_SIZE):
                print(display[i][j], end='')
            print('')


def main():
    
    q = Solver_8_queens()

    indiv = q.get_random_individual()
    print(indiv)

    print(q.fitness(indiv))

    #q->solver
    #best_fit, epoch_num, visualization = solver.solve()
