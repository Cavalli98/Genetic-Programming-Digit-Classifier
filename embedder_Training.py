import os
import sys
import csv
import multiprocessing
import pickle
import time
import random
import operator
import itertools
from datetime import datetime
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from embedder_algorithms import eaMuPlusLambda_mod
from embedder_algorithms import evaluate_differentiator
from math import sqrt
from math import exp
import numpy
import matplotlib.pyplot as plt


########################################################################################################################
# LOAD PARAMETERS
########################################################################################################################

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path+'\\pkl_dump\\params.pkl', "rb") as params_file:
    params_dict = pickle.load(params_file)

word_len = params_dict['word_len']
gen_n = params_dict['gen_n_diff']  # number of generations to evolve
population = params_dict['population_diff']  # size of the initial population
path_training = params_dict['dataset_training_diff']
path_to_transform = params_dict['dataset_training_class']
evolution = params_dict['evolution']

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
hof_size = params_dict['hof_size']     # per Elitismo
use_erc = params_dict['use_erc']
use_const1 = params_dict['use_const0']
use_const0 = params_dict['use_const1']
ks_const = params_dict['k_const']

########################################################################################################################
# IMPORT DATA SET
########################################################################################################################

path_testing = "\\dataset\\bigalltest_ul104.csv"
path_save_ResultStr = "\\result\\embedder_result.csv"
path_save_encoderHOF = "\\pkl_dump\\emdedderHOF.pkl"
path_save_DatasetTraining = "\\dataset\\"+path_to_transform[18:]
path_save_encodedDatasetTest = "\\dataset\\embedded_"+path_testing[9:]

bigall3_matrix = []
with open(dir_path+path_training, newline='\n') as bigall3_file:
    bigall3_data = csv.reader(bigall3_file, delimiter=';')
    for row in bigall3_data:
        bigall3_matrix.append([int(row[0]),int(row[1]),int(row[2])])
row_num = bigall3_data.line_num

########################################################################################################################
# DEFINE PRIMITIVES SET
########################################################################################################################

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
def nand(val1, val2):
    return operator.__invert__((operator.__and__(val1, val2)))
def nor(val1, val2):
    return operator.__invert__((operator.__or__(val1, val2)))

# Inizializza il PrimitiveSet per gli operatori e i terminali
pset = gp.PrimitiveSetTyped("MAIN", [int, int], int)

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
    pset.addTerminal(0, int)
if use_const1:
    pset.addTerminal(1, int)
if use_erc:
    pset.addEphemeralConstant("ERC", lambda: random.randint(0, 2 ** word_len - 1), int)

########################################################################################################################
# DEFINE FITNESS FUNCTION
########################################################################################################################

creator.create("Fitnesses", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitnesses)

# fitness function for classifier
def evaluate(individual):
    # Trasforma l'individuo (albero sintattico) in un funzione chiamabile
    # func prende [ARG0, ARG1] come input
    func = toolbox.compile(expr=individual)
    list_Transf = []
    sum_same = 0
    count_same = 0
    sum_diff = 0
    count_diff = 0
    sum_dev_s = 0
    sum_dev_d = 0
    similar_penalty = 0

    # Valuto la funzione sul dataset
    for j in range(0, row_num):
        ARG0 = bigall3_matrix[j][0]
        ARG1 = bigall3_matrix[j][1]
        label = bigall3_matrix[j][2]

        tree_result = func(ARG0, ARG1)
        list_Transf.append([tree_result, label])
    
    # Calcolo la distanza di Hamming dei valori trasformati
    for i in range(0, row_num):
        for j in range(i+1, row_num):
            binxor = list_Transf[i][0] ^ list_Transf[j][0]
            h_d = bin(binxor).count("1")
            if (list_Transf[i][1] == list_Transf[j][1]):
                sum_same += h_d
                sum_dev_s += (h_d - 1)**2
                count_same += 1
            else:
                sum_diff += h_d
                sum_dev_d += (h_d - 36)**2
                count_diff += 1
                # Penalità per individui appartenenti a classi diverse ma uguali (evito albero predefinito da const)
                #if (list_Transf[i][0] == list_Transf[j][0]):
                #    similar_penalty += ks_const

    w1 = 1
    w2 = 180
    fatt_Norm = 40
    HD_same = sum_same / count_same
    HD_diff = sum_diff / count_diff
    Dev_same = sqrt(sum_dev_s / count_same)
    Dev_diff = sqrt(sum_dev_d / count_diff)

    #Norm_same = HD_same
    #Norm_diff = fatt_Norm - HD_diff
    Distanza = HD_diff - HD_same
    #Sigm_diff = w2/(1 + exp(HD_diff-4))
    #try:
    #   Distanza_quadr = sqrt(HD_diff**2 - HD_same**2)
    #except:
    #    Distanza_quadr = 0

    if (HD_diff > 2):                  # riaggiungere similar_penalty, sembra migliorare le prestazioni rispetto a questo
        fitness_x = Dev_same + Dev_diff
        fitness_y = Distanza
    else:
        fitness_x = 1000
        fitness_y = 0

    # Ritorna la fitness (tupla)
    return fitness_x, fitness_y



########################################################################################################################
# INITIALIZE TOOLBOX: register some parameters specific to the evolution process
########################################################################################################################

toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=fgen_minh, max_=fgen_maxh)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=mut_app_minh, max_=mut_app_maxh)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Fissa l'altezza massima degli individui
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=mate_maxh))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=mut_maxh))

# Fissa la dimensione massima degli individui
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=mate_maxs))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=mut_maxs))

########################################################################################################################
# EVOLUTION
########################################################################################################################

random.seed(318)

def min_inv(fitness_values):
    return 1 / (1 + numpy.min(fitness_values))

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
mstats = tools.MultiStatistics(fitness=stats_fit)
mstats.register("min", numpy.min, axis=0)
mstats.register("avg", numpy.average, axis=0)
mstats.register("max", numpy.max, axis=0)

# Creo la popolazione iniziale
pop = toolbox.population(n=population)
hof = tools.ParetoFront()


if __name__ == '__main__':

    if evolution:
        # Inizio cronometro
        start_time = time.time()
        pool = multiprocessing.Pool(os.cpu_count())
        toolbox.register("map", pool.map)

        # Inizio Algoritmo
        pop, log = eaMuPlusLambda_mod(pop, toolbox, cxpb=cxpb, mutpb=mutpb, mu=population, lambda_=population, ngen=gen_n, stats=mstats, halloffame=hof, verbose=True)
        
        # Stop cronometro
        elapsed_time = round(time.time() - start_time)
        elapsed_time = time.strftime('Durata Training : %H:%M:%S', time.gmtime(elapsed_time))
        print(elapsed_time+"(hh:mm:ss)")

        pool.close()   
        pool.join() 



        ########################################################################################################################
        # VALUTAZIONE ENCODER E SCRITTURA VARI FILES
        ########################################################################################################################
        best_encoder = hof[0]
        best_encoder2 = hof[0]
        val = [1, 1]
        x = []
        y = []
        print("Lunghezza hof: {}".format(len(hof)))
        for ind in hof:
            
            func = toolbox.compile(ind)
            sum_same = 0
            sum_diff = 0
            count_same = 0
            count_diff = 0
            temp_list = []
            for j in range(0, row_num):
                ARG0 = bigall3_matrix[j][0]
                ARG1 = bigall3_matrix[j][1]
                label = bigall3_matrix[j][2]

                tree_result = func(ARG0, ARG1)
                temp_list.append([tree_result, label])
            
            for i in range(0, row_num):
                for j in range(i+1, row_num):
                    binxor = temp_list[i][0] ^ temp_list[j][0]
                    h_d = bin(binxor).count("1")
                    if (temp_list[i][1] == temp_list[j][1]):
                        sum_same += h_d
                        count_same += 1
                    else:
                        sum_diff += h_d
                        count_diff += 1
        
            HDs = sum_same / count_same
            HDd = sum_diff / count_diff
            x.append(HDs)
            y.append(HDd)
            if ((val[1] - val[0]) < (HDd - HDs)):
                best_encoder = ind
                val[1] = HDd
                val[0] = HDs
        
        print("Valori finali: {}, {}".format(val[0], val[1]))    

        #seaborn.set(style='whitegrid')
        #seaborn.set_context('notebook')
        #plt.subplot(111)
        #plt.plot(x, y, 'ro', label='HOF', alpha=0.7)
        #plt.xlabel('HDs')
        #plt.ylabel('HDd')
        #plt.title("Valori Pareto")
        #mxy = max(y)+2
        #mxx = max(x)+2
        #plt.axis([0, mxx, 0, mxy])
        #plt.legend()
        #plt.show()

        
        best_encoder_func = toolbox.compile(best_encoder)

        bigall3test_matrix = []
        with open(dir_path+path_testing, newline='\n') as bigall3_file:
            bigall3_data = csv.reader(bigall3_file, delimiter=';')
            for row in bigall3_data:
                bigall3test_matrix.append([int(row[0]),int(row[1]),int(row[2])])

        results_string = "-"*80
        time_now = time.time()
        time_now_str = time.strftime('\nProcedura Encoder finita alle: %H:%M:%S', time.gmtime(time_now))
        results_string += time_now_str + "\n"
        results_string += "-"*80
        results_string += "\n" + elapsed_time + "\n"
        results_string += "Risultati su : " + path_training + "\n"
        results_string += evaluate_differentiator(toolbox, best_encoder, bigall3_matrix, onlyMed=True)
        with open(dir_path+path_save_ResultStr, "a") as f:
            f.write(results_string)
        results_string = "Risultati su : " + path_testing + "\n"
        results_string += evaluate_differentiator(toolbox, best_encoder, bigall3test_matrix, onlyMed=True)
        with open(dir_path+path_save_ResultStr, "a") as f:
            f.write(results_string)

    else:
        with open(dir_path+'\\pkl_dump\\embedderHOF.pkl', "rb") as embedder_file:
            best_encoder = pickle.load(embedder_file)
        
        best_encoder_func = toolbox.compile(best_encoder)
        
        bigall3test_matrix = []
        with open(dir_path+path_testing, newline='\n') as bigall3_file:
            bigall3_data = csv.reader(bigall3_file, delimiter=';')
            for row in bigall3_data:
                bigall3test_matrix.append([int(row[0]),int(row[1]),int(row[2])])

    
    # trasforma il dataset per il test
    with open(dir_path+path_save_encodedDatasetTest, 'w', newline='\n') as f:
        for row in bigall3test_matrix:
            res = best_encoder_func(row[0], row[1])
            encoded_row = str(res)+";"+str(row[2])+"\n"
            f.write(encoded_row)

    bigall3train_matrix = []
    with open(dir_path+path_save_DatasetTraining, newline='\n') as bigall3_file:
        bigall3_data = csv.reader(bigall3_file, delimiter=';')
        for row in bigall3_data:
            bigall3train_matrix.append([int(row[0]),int(row[1]),int(row[2])])

    #trasforma il dataset per il training del classificatore
    with open(dir_path+path_to_transform, 'w', newline='\n') as f:
        for row in bigall3train_matrix:
            res = best_encoder_func(row[0], row[1])
            encoded_row = str(res)+";"+str(row[2])+"\n"
            f.write(encoded_row)

    with open(dir_path+path_save_encoderHOF, 'wb') as output:
        pickle.dump(best_encoder, output, -1)




# TODO Aggiungere PLOT ???
# TODO sistemare VALUTAZIONE ENCODER E SALVATAGGI