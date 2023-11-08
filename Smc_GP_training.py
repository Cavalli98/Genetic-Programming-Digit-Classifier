import random
import operator
import itertools
from datetime import datetime
import os
import numpy
import csv
import multiprocessing
import os
import pickle
import sys
import time
from deap import algorithms, base,  creator, tools, gp
from embedder_algorithms import eaSimple_cp

########################################################################################################################
# LOAD PARAMETERS
########################################################################################################################

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path+'\\pkl_dump\\params.pkl', 'rb') as output:
    params_dict = pickle.load(output)

word_len = params_dict['word_len']
path_training = params_dict['dataset_training_class']
gen_n = params_dict['gen_n_class']  # number of generations to evolve
population = params_dict['population_class']  # size of the initial population
fgen_minh = params_dict['fgen_minh']  # first generation min height
fgen_maxh = params_dict['fgen_maxh']  # first generation max height
tournsize = params_dict['tournsize']  # selection tournament size
mut_app_minh = params_dict['mut_app_minh']  # min height of an appendable tree in mutation
mut_app_maxh = params_dict['mut_app_maxh']  # max height of an appendable tree in mutation
mate_maxh = params_dict['mate_maxh']  # max reachable height during mate
mut_maxh = params_dict['mut_maxh']  # max reachable height during mutation
mate_maxs = params_dict['mate_maxs']  # max reachable size during mate
mut_maxs = params_dict['mut_maxs']  # max reachable size during mutation
cxpb = params_dict['cxpb']  # crossover probability
mutpb = params_dict['mutpb']  # mutation probability

use_erc = params_dict['use_erc']
use_const1 = params_dict['use_const0']
use_const0 = params_dict['use_const1']

ks_const = params_dict['ks_const']

target_digit = params_dict['target_digit']  # digit to recognize


########################################################################################################################
# IMPORT DATA SET
########################################################################################################################


# measure elapsed time
start_time = time.time()

bigall3_matrix = []

with open(dir_path+path_training, newline='\n') as bigall3_file:
    bigall3_data = csv.reader(bigall3_file, delimiter=';')
    for row in bigall3_data:
        bigall3_matrix.append([int(row[0]), int(row[1])])

row_num = bigall3_data.line_num


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


# define a new primitive set for strongly typed GP
# todo qui dobiamo specificare che anch un ERC può essere in input come foglia?
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

# TODO capire come usare le costanti correttamente,
#   sembrano peggiorare le performance

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


# fitness function for classifier0
def eval_fitness0(individual):
    # Transform the tree expression into a callable function
    # func takes [ARG0, ARG1] as inputs
    func = toolbox.compile(expr=individual)
    bit_mask = 0x1

    tp = [0] * word_len  # true positive
    tn = [0] * word_len  # true negative
    fp = [0] * word_len  # false positive
    fn = [0] * word_len  # false negative
    pdata = 0  # positives in the data set
    ndata = 0  # negatives in the data set

    # Evaluate func over the entire data set

    for j in range(0, row_num):
        ARG0 = bigall3_matrix[j][0]
        label = bigall3_matrix[j][1]

        tree_result = func(ARG0)

        if label == 0:
            pdata = pdata + 1
        else:
            ndata = ndata + 1

        # mask the tree result
        for i in range(0, word_len):
            tree_result_bit = (bit_mask << i) & tree_result

            if label == target_digit:
                if tree_result_bit:
                    tp[i] = tp[i] + 1
                else:
                    fn[i] = fn[i] + 1
            else:
                if tree_result_bit:
                    fp[i] = fp[i] + 1
                else:
                    tn[i] = tn[i] + 1

    # evaluate the fitness on each stat vector

    # use of map() for speed
    fit_vec = list(map(lambda a, b: numpy.sqrt(50 * (a ** 2 + b ** 2) / ((pdata + ndata) ** 2)) +
                                    len(individual) * ks_const, fp, fn))

    # return actual fitness (tuple of 1 element)
    return min(fit_vec),


########################################################################################################################
# INITIALIZE TOOLBOX: register some parameters specific to the evolution process
########################################################################################################################

toolbox = base.Toolbox()
# initialize first generation and compile function
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=fgen_minh, max_=fgen_maxh)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", eval_fitness0)
toolbox.register("select", tools.selTournament, tournsize=tournsize)

# mate method (one point crossover with uniform probability over all the nodes)
toolbox.register("mate", gp.cxOnePoint)
# mutation method (an uniform probability mutation which may append a new full sub-tree to a node)
toolbox.register("expr_mut", gp.genFull, min_=mut_app_minh, max_=mut_app_maxh)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Set tree depth limit
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=mate_maxh))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=mut_maxh))

# Set tree size limit
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=mate_maxs))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=mut_maxs))

########################################################################################################################
# EVOLUTION
########################################################################################################################

# TODO dovrebbe essere 1000 generazioni con popolazione iniziale 1000

random.seed(318)


pop = toolbox.population(n=population)

hof_size = 10
hof = tools.HallOfFame(hof_size)

def min_inv(fitness_values):
    return 1 / (1 + numpy.min(fitness_values))


stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
mstats = tools.MultiStatistics(fitness=stats_fit)
mstats.register("min", numpy.min)
mstats.register("min_inv", min_inv)


if os.name == 'nt':
    # multiprocessing on windows
    if __name__ == '__main__':
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

        pop, log = eaSimple_cp(pop, toolbox, cxpb, mutpb, gen_n, stats=mstats, halloffame=hof, verbose=True, FREQ=10)

        # TODO salvare in un file statistiche degli individui migliori

        ########################################################################################################################
        # DUMP HALL OF FAME (HOF)
        ########################################################################################################################

        output = open(dir_path+"\\pkl_dump\\hof_10.pkl", 'wb')
        pickle.dump(hof[0], output, -1)
        output.close()

        elapsed_time = round(time.time() - start_time)
        elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
        # DUMP PARAMETERS_______________________________________________________________________________________________________

        # datetime object containing current date and time
        now = datetime.now()

        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        results_str = "\nTRAINING ENDED " + dt_string + " with PARAMS\n" + \
                    str(115 * "-") + "\n" + \
                    str(word_len) + "\t" + \
                    str(gen_n) + "\t" + \
                    str(population) + "\t" + \
                    str(fgen_minh) + "\t" + \
                    str(fgen_maxh) + "\t" + \
                    str(tournsize) + "\t" + \
                    str(mut_app_minh) + "\t" + \
                    str(mut_app_maxh) + "\t" + \
                    str(mate_maxh) + "\t" + \
                    str(mut_maxh) + "\t" + \
                    str(mate_maxs) + "\t" + \
                    str(mut_maxs) + "\t" + \
                    str(cxpb) + "\t" + \
                    str(mutpb) + "\t" + \
                    str(use_erc) + "\t" + \
                    str(use_const1) + "\t" + \
                    str(use_const0) + "\t" + \
                    str(ks_const) + "\t" + \
                    str(target_digit) + "\t" + \
                    elapsed_time + "(hh:mm:ss)\n"

        print("\n\nRESULTs\n" + results_str)

        # open the dump file or create it
        f = open(dir_path+"\\result\\results_dump.csv", "a")
        f.write(results_str)
        f.close()