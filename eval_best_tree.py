import pickle
import random
import operator
import csv
import itertools
import os
import numpy
from deap import algorithms, creator, base, tools, gp
from embedder_algorithms import evaluate_classifier

########################################################################################################################
# LOAD PARAMETERS
########################################################################################################################

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path+'\\pkl_dump\\params.pkl', 'rb') as output:
    params_dict = pickle.load(output)

word_len = params_dict['word_len']

gen_n = params_dict['gen_n_class']  # number of generations to evolve
population = params_dict['population_class']  # size of the initial population
path_training = params_dict['dataset_training_class']
fgen_minh = params_dict['fgen_minh']  # first generation min height
fgen_maxh = params_dict['fgen_maxh']  # first generation max height
tournsize = params_dict['tournsize']  # selection tournament size
mut_app_minh = params_dict['mut_app_minh']  # min height of an appendable tree in mutation
mut_app_maxh = params_dict['mut_app_maxh']  # max height of an appendable tree in mutation
mate_maxh = params_dict['mate_maxh']  # max reachable height during mate
mut_maxh = params_dict['mut_maxh']  # max reachable height during mutation
mate_maxs = params_dict['mate_maxs']  # max reachable size during mate
mut_maxs = params_dict['mut_maxs']  # max reachable size during mutation

use_erc = params_dict['use_erc']
use_const1 = params_dict['use_const0']
use_const0 = params_dict['use_const1']

ks_const = params_dict['ks_const']

target_digit = params_dict['target_digit'] # digit to recognize

########################################################################################################################
# DEFINE PRIMITIVES SET
########################################################################################################################

# circular shift operators (1, 2 and 4 bits)

def rshift1(val):
    return ((val & (2 ** word_len - 1)) >> 1 % word_len) | \
           (val << (word_len - (1 % word_len)) & (2 ** word_len - 1))


def lshift1(val):
    return (val << 1 % word_len) & (2 ** word_len - 1) | \
           ((val & (2 ** word_len - 1)) >> (word_len - (1 % word_len)))


def rshift2(val):
    return ((val & (2 ** word_len - 1)) >> 2 % word_len) | \
           (val << (word_len - (2 % word_len)) & (2 ** word_len - 1))


def lshift2(val):
    return (val << 2 % word_len) & (2 ** word_len - 1) | \
           ((val & (2 ** word_len - 1)) >> (word_len - (2 % word_len)))


def rshift4(val):
    return ((val & (2 ** word_len - 1)) >> 4 % word_len) | \
           (val << (word_len - (4 % word_len)) & (2 ** word_len - 1))


def lshift4(val):
    return (val << 4 % word_len) & (2 ** word_len - 1) | \
           ((val & (2 ** word_len - 1)) >> (word_len - (4 % word_len)))


# nand nor operators and 0 1 constants

def nand(val1, val2):
    return operator.__invert__((operator.__and__(val1, val2)))


def nor(val1, val2):
    return operator.__invert__((operator.__or__(val1, val2)))


def const1():
    return 1


def const0():
    return 0


# defined a new primitive set for strongly typed GP
# TODO controllare che sia corretta la sintassi
pset = gp.PrimitiveSetTyped("MAIN", [int], int)

# bitwise operators
pset.addPrimitive(operator.__and__, [int, int], int)
pset.addPrimitive(operator.__or__, [int, int], int)
pset.addPrimitive(operator.__invert__, [int], int)
pset.addPrimitive(operator.__xor__, [int, int], int)
pset.addPrimitive(nand, [int, int], int)
pset.addPrimitive(nor, [int, int], int)
pset.addPrimitive(rshift1, [int], int)
pset.addPrimitive(lshift1, [int], int)
pset.addPrimitive(rshift2, [int], int)
pset.addPrimitive(lshift2, [int], int)
pset.addPrimitive(rshift4, [int], int)
pset.addPrimitive(lshift4, [int], int)
if use_const0:
    pset.addPrimitive(const0, [], int)

if use_const1:
    pset.addPrimitive(const1, [], int)

# constants random int in range 0 - 2^word_len -1
if use_erc:
    pset.addEphemeralConstant("ERC", lambda: random.randint(0, 2 ** word_len - 1), int)

########################################################################################################################
# DEFINE FITNESS FUNCTION
########################################################################################################################

# Fitness to be minimized: weights < 0
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

########################################################################################################################
# LOAD BEST TREE FROM TRAINING
########################################################################################################################

with open(dir_path+'\\pkl_dump\\hof_10.pkl', "rb") as hof_best_file:
    hof_best = pickle.load(hof_best_file)

best_classifier = gp.compile(hof_best, pset)
best_classifier_size = hof_best.__len__()

########################################################################################################################
# TEST BEST CLASSIFIER ON TRAINING SET
########################################################################################################################

data_set = dir_path+path_training
data_set_type = 'TRAINING'
dump_file = dir_path+"\\result\\results_dump.csv"

evaluate_classifier(data_set, data_set_type, dump_file, word_len, best_classifier, best_classifier_size,
                    ks_const, target_digit)

########################################################################################################################
# TEST BEST CLASSIFIER ON TEST SET
########################################################################################################################

data_set = dir_path+'\\dataset\\embedded_bigalltest_ul104.csv'
data_set_type = 'TEST    '
dump_file = dir_path+"\\result\\results_dump.csv"

evaluate_classifier(data_set, data_set_type, dump_file, word_len, best_classifier, best_classifier_size,
                    ks_const, target_digit)