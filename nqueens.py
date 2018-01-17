import random
import numpy as np

class Solver_8_queens:

	DIM_SIZE = 8
	DIMENSION = 2

	def __init__(self):
		pass


	def create_start_population(self):
		pass


	def fitness(sel, individual):
		#dummy method
		pass


	def get_init_population(self, size):
		population = [self.get_random_individual() for _ in range(size)]
		return population


	def get_random_individual(self):
		individual = [self.get_random_chromosome() for _ in range(Solver_8_queens.DIM_SIZE)]
		return individual


	def get_random_chromosome(self):
		chromosome = random.choices(range(Solver_8_queens.DIM_SIZE), k=Solver_8_queens.DIMENSION)
		return chromosome


def main():
	
	q = Solver_8_queens()