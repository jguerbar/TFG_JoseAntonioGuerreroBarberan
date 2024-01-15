import graphviz  
import operator
import random
import numpy as np
import math
import copy
from random import randrange
from tqdm import tqdm
import os
import warnings
import torch.nn as nn
import torch
warnings.filterwarnings('ignore') #TODO? 
FITNESS_THRESHOLD = 1e-10
MAX_DATA = 50000000

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value  # Can be an operation or an operand
        self.childs = 0
        self.left = left
        self.right = right

def tree_depth(node):
    if node is None:
        return 0
    else:
        left_depth = tree_depth(node.left)
        right_depth = tree_depth(node.right)
        return max(left_depth, right_depth) + 1
    
def mutate(tree, mutation_rate=0.1):
    if (tree == None): return
    if random.random() < mutation_rate:
        if random.random() < 0.5:
            # Mutate the current node's value
            if isinstance(tree.value, (int, float)):  # If it's a constant
                tree.value = random_constant()
            else:
                tree.value = random.choice(list(operations.keys()) + variables)
        else:
            # Replace a subtree
            if random.random() < 0.5 and tree.left:
                tree.left = create_random_tree(depth=random.randint(1, 3))
            elif tree.right:
                tree.right = create_random_tree(depth=random.randint(1, 3))
    else:
        if tree.left != None:
            mutate(tree.left, mutation_rate)
        if tree.right != None:
            mutate(tree.right, mutation_rate)
def crossover(parent1, parent2, crossover_rate=0.1):
    if random.random() < crossover_rate:
        # Randomly choose a node from each parent and swap
        node1 = random.choice(get_all_nodes(parent1))
        node2 = random.choice(get_all_nodes(parent2))
        if (node1 == None or node2 == None): return
        node1.value, node2.value = node2.value, node1.value
        node1.left, node2.left = node2.left, node1.left
        node1.right, node2.right = node2.right, node1.right

def get_all_nodes(tree):
    nodes = [tree]
    if tree == None: return nodes
    if tree.left:
        nodes += get_all_nodes(tree.left)
    elif tree.right:
        nodes += get_all_nodes(tree.right)
    return nodes

def print_tree(root):
    if not root:
        return "Empty Tree"
    def height(node):
        if not node:
            return 0
        return max(height(node.left), height(node.right)) + 1

    def print_level(node, level):
        if not node:
            print("   ", end="")
            return
        if level == 1:
            print(f'{node.value:3}', end="")
        elif level > 1:
            print_level(node.left, level - 1)
            print_level(node.right, level - 1)

    h = height(root)
    for i in range(1, h + 1):
        print_level(root, i)
        print()
def tree_to_equation(node):
    if node is None:
        return ""

    # If the node is a leaf (an operand)
    if node.left is None and node.right is None:
        return str(node.value)
    
    # If the node is an operation
    left_subtree = tree_to_equation(node.left)
    right_subtree = tree_to_equation(node.right)

    # Format the equation with parentheses to denote operations
    if node.value in operations:  # operations should be a dictionary or set of your operations
        if node.value in two_operads_op:
            return f"({left_subtree} {node.value} {right_subtree})"
        else:
            return f"({node.value} ({left_subtree}))"
    else:
        # Handle the case if the node value is not an operation
        return str(node.value)
def are_trees_equivalent(tree1, tree2):
    # Both trees are empty
    if tree1 is None and tree2 is None:
        return True

    # One tree is empty, the other is not
    if tree1 is None or tree2 is None:
        return False

    # Check if the current nodes have the same value and
    # recursively check their left and right children
    return (tree1.value == tree2.value and
            are_trees_equivalent(tree1.left, tree2.left) and
            are_trees_equivalent(tree1.right, tree2.right))

#Variables
variables = ['x','y',
             'U00','U10','U01','U11',
             'V00','V10','V01','V11',
             'P00','P10','P01','P11',
             'x0','y0','x1','y1']

# Basic arithmetic operations
def power2(a):
    return operator.pow(a,2)
def power3(a):
    return operator.pow(a,3)

operations = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv, 
    '^2': power2,
    '^3': power3,
    'sin': np.sin, 
    'exp': np.exp,  
    'log': np.log,  
}
two_operads_op = ['+','-','*','/','^2','^3']


# Function to generate a random constant between -10 and 10
def random_constant():
    return random.uniform(-10, 10)

def create_random_tree(depth=3):
    if depth == 0 or (depth != -1 and random.random() > 0.5):
        if random.random() < 0.5:
            return Node(random_constant())  # Return a constant
        else:
            var = random.choice(variables)
            return Node(var)  # Return a variable
    else:
        operation = random.choice(list(operations.keys()))
        if operation in two_operads_op:
            return Node(operation, create_random_tree(depth-1), create_random_tree(depth-1))
        else:
            return Node(operation, create_random_tree(depth-1), None)

def evaluate_tree(tree, input_value):
    if tree.value in operations:
        left_val = evaluate_tree(tree.left, input_value)
        right_val = evaluate_tree(tree.right, input_value) if tree.right != None else None
        if right_val != None:
            return operations[tree.value](left_val, right_val)
        else: 
            return operations[tree.value](left_val)
    elif tree.value in variables:
        i = variables.index(tree.value)
        return input_value[i]
    else: #Is constant
        return tree.value
    
def fitness(tree, data, printPred=False):
    error = 0
    for datapoint in data:
        y = datapoint[0]
        x = datapoint[1:]
        try:
            prediction = evaluate_tree(tree, x)
            if printPred: print("Pred value", prediction)
            if np.iscomplex(prediction):
               return float('inf')  # Penalize invalid equations
            else:
                error += (prediction - y) ** 2 #Squared error
        except: #Exception as e:
            return float('inf')  # Penalize invalid equations
    return   error/data.shape[0] if not np.isnan(error) else float('inf') # Penalize invalid equations

#Selection
def tournament_selection(population, fitness_values, tournament_size=10):
    tournament = random.sample(population, tournament_size)
    sample_index = np.random.choice(len(population),tournament_size)

    tournament = np.take(population, sample_index)
    tournament_fitness = np.take(fitness_values, sample_index)

    fittest_index = np.argmin(tournament_fitness)
    return tournament[fittest_index]

def select_population(current_population, fitness_values, size=5):
    new_population = []
    for _ in range(size):
        individual = tournament_selection(current_population,fitness_values)
        new_population.append(individual)
    return new_population

def remove_duplicate_trees_by_fitness(population, fitness_values):
    unique_trees = []
    unique_fitness_values = []

    for idx, tree in enumerate(population):
        if not any(abs(fitness_values[idx] - other_fitness) < FITNESS_THRESHOLD
                   for other_fitness in unique_fitness_values):
            unique_trees.append(tree)
            unique_fitness_values.append(fitness_values[idx])

    return unique_trees, unique_fitness_values

def print_population(population):
    sample_index = np.random.choice(data.shape[0],round(DATA_SAMPLE/10))
    for individual in population:
        print(fitness(individual, np.take(data,sample_index,0)), " | z =",tree_to_equation(individual))



DOWNSAMPLE_FACTOR = 4
downsampler = nn.AvgPool2d(DOWNSAMPLE_FACTOR)
#Import data
data_raw = np.load('./data/trainData.npy')
data = np.empty((MAX_DATA, 19))  #Allocate spacebefore hand
dataIndex = 0
for i in range(data_raw.shape[0]):
    if dataIndex > MAX_DATA: break
    data_hr = data_raw[i]
    data_lr = downsampler(torch.tensor(data_hr))
    data_hr = np.transpose(data_hr,(0,2,3,1))
    data_lr = np.transpose(data_lr,(0,2,3,1))
    for t in range(data_hr.shape[1]):
        data_iteration_hr = data_hr[t]
        data_iteration_lr = data_lr[t]
        for i in range(0,data_iteration_hr.shape[0]-DOWNSAMPLE_FACTOR):
            for j in range(0,data_iteration_hr.shape[1]-DOWNSAMPLE_FACTOR):
                    i_lr_equivalent = math.floor(i/(DOWNSAMPLE_FACTOR))
                    j_lr_equivalent = math.floor(j/(DOWNSAMPLE_FACTOR))
                        
                    x0 = i_lr_equivalent/data_iteration_lr.shape[0]
                    y0 = j_lr_equivalent/data_iteration_lr.shape[0]
                    x1 = (i_lr_equivalent+1)/data_iteration_lr.shape[0]
                    y1 = (j_lr_equivalent+1)/data_iteration_lr.shape[0]

                    x = i/data_iteration_hr.shape[0]
                    y = j/data_iteration_hr.shape[0]
                        
                    u00 = data_iteration_lr[i_lr_equivalent][j_lr_equivalent][0]
                    u10 = data_iteration_lr[i_lr_equivalent+1][j_lr_equivalent][0]
                    u01 = data_iteration_lr[i_lr_equivalent][j_lr_equivalent+1][0]
                    u11 = data_iteration_lr[i_lr_equivalent+1][j_lr_equivalent+1][0]
                    v00 = data_iteration_lr[i_lr_equivalent][j_lr_equivalent][1]
                    v10 = data_iteration_lr[i_lr_equivalent+1][j_lr_equivalent][1]
                    v01 = data_iteration_lr[i_lr_equivalent][j_lr_equivalent+1][1]
                    v11 = data_iteration_lr[i_lr_equivalent+1][j_lr_equivalent+1][1]
                    p00 = data_iteration_lr[i_lr_equivalent][j_lr_equivalent][2]
                    p10 = data_iteration_lr[i_lr_equivalent+1][j_lr_equivalent][2]
                    p01 = data_iteration_lr[i_lr_equivalent][j_lr_equivalent+1][2]
                    p11 = data_iteration_lr[i_lr_equivalent+1][j_lr_equivalent+1][2]
                    z = data_iteration_hr[i][j][2]
                    new_row = np.array((z, x,y,
                                u00,u10,u01,u11,
                                v00,v10,v01,v11,
                                p00,p10,p01,p11,
                                x0,y0,x1,y1)).reshape(1, -1)
                    if dataIndex >= MAX_DATA: break
                    data[dataIndex] = new_row
                    dataIndex += 1
    del data_hr, data_lr
data[:dataIndex]

# Parameters
tree_initial_depth = 3
population_size = 300
N_SELECTION = round(population_size/3)
DATA_SAMPLE = 50000 #Amount of data used to train each generation, which will be randomly sampled. We have to do this to deal with the limitations of the algorithm.
max_generations = 100
desired_fitness = -1  # Set a threshold for desired fitness
crossover_rate = 0.3
mutation_rate = 0.3
# Create initial population
population = [create_random_tree(tree_initial_depth) for _ in range(population_size)]
best_individual = None
best_fitness = float('inf')
best_copies = 3


for generation in tqdm(range(max_generations), desc="Optimizing...", ascii=False, ncols=64):
    # Evaluate fitness
    sample_index = np.random.choice(data.shape[0],DATA_SAMPLE)
    fitness_values = [fitness(individual, np.take(data,sample_index,0)) for individual in population]

    # Remove duplicate trees based on fitness
    population, fitness_values = remove_duplicate_trees_by_fitness(population, fitness_values)

    # Find the best individual
    current_best_idx = fitness_values.index(min(fitness_values))
    current_best_individual = population[current_best_idx]
    current_best_fitness = fitness_values[current_best_idx]
    if current_best_fitness < best_fitness:
        best_fitness = current_best_fitness
        best_individual = current_best_individual
        print(f"\nGeneration {generation}: New best fitness = {best_fitness:.2e} Eq: z = {tree_to_equation(best_individual)}")
    # Check for termination condition
    if best_fitness <= desired_fitness:
        print(f"Desired fitness level reached at generation {generation}")
        break
    
    # Selection of N best individuals
    new_population = select_population(population, fitness_values, size = N_SELECTION)
    #Explicitly include best individual
    new_population.append(best_individual)
    #Add a few mutations of best individual
    for i in range(best_copies):
        child_best = copy.deepcopy(best_individual)
        mutate(child_best,mutation_rate)
        new_population.append(child_best)

    if generation > 0 and generation % 5 == 0: print_population(new_population[-10:])

    selectedPopulation = len(new_population)
    # Crossover and mutation until filling population
    while (len(new_population) < population_size):
        #Sample two parents and create child
        parent1 = new_population[randrange(selectedPopulation)]
        parent2 = new_population[randrange(selectedPopulation)]
        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)  

        #Crossover and mutate
        if random.random() < 0.5:
            crossover(child1, child2, crossover_rate)
        else:
            mutate(child1,mutation_rate)
            mutate(child2,mutation_rate)
        #Introduce new random tree
        child3 = create_random_tree(tree_initial_depth)
        new_population.extend([child1, child2, child3])

    population = new_population

print("Fitness: ",fitness(best_individual,data,printPred=False))
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)
print(tree_to_equation(best_individual))

