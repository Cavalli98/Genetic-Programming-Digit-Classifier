import os
import random
import pickle
import csv
import numpy
from deap import tools
from deap.algorithms import varOr
import matplotlib.pyplot as plt
import seaborn

dir_path = os.path.dirname(os.path.realpath(__file__))


def varAnd(population, toolbox, cxpb, mutpb):

    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def eaSimple_cp(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, FREQ=1):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # save checkpoint (last gen always saved)
        if (gen % FREQ == 0) or (gen == ngen + 1):
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            dir_path = path.dirname(path.realpath(__file__))
            cp = dict(population=population, generation=gen, halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate())
            print("\n checkpoint")
            with open(dir_path+"\\pkl_dump\\checkpoint.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)

    return population, logbook


def eaMuPlusLambda_mod(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
        stampaValori(toolbox, halloffame)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)


        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if (gen % 10 == 1) or (gen == ngen + 1):
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate())
            stampaValori(toolbox, halloffame, True)
            with open(dir_path+"\\pkl_dump\\checkpoint.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)

        
        #if (gen == 1):
            #plot(toolbox, population)

    return population, logbook


def fitness_best(data_matrix, row_num, word_len, best_classifier, best_classifier_size, ks_const, target_digit):
    bit_mask = 0x1
    fit_vec = []
    stat_matrix = []

    tp = [0] * word_len  # true positive
    tn = [0] * word_len  # true negative
    fp = [0] * word_len  # false positive
    fn = [0] * word_len  # false negative
    pdata = 0  # positives in the data set
    ndata = 0  # negatives in the data set

    # Evaluate func over the entire data set
    for j in range(0, row_num):
        ARG0 = data_matrix[j][0]
        label = data_matrix[j][1]

        tree_result = best_classifier(ARG0)

        if label == target_digit:
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

    # # evaluate the fitness on each stat vector
    # use of map() for speed
    fit_vec = list(map(lambda a, b: numpy.sqrt(50 * (a ** 2 + b ** 2) / ((pdata + ndata) ** 2)) +
                                    best_classifier_size * ks_const, fp, fn))

    # Save stats and fitness
    for i in range(0, word_len):
        stat_matrix.append([tp[i], tn[i], fp[i], fn[i]])

    return fit_vec, stat_matrix, pdata, ndata


def evaluate_classifier(data_set, data_set_type, dump_file, word_len, best_classifier, best_classifier_size,
                        ks_const, target_digit):
    data_matrix = []

    with open(data_set, newline='\n') as bigall3test_file:
        bigall3test_data = csv.reader(bigall3test_file, delimiter=';')
        for row in bigall3test_data:
            data_matrix.append([int(row[0]), int(row[1])])

    row_num = bigall3test_data.line_num

    fit_vec, stat_matrix, pdata, ndata = fitness_best(data_matrix, row_num, word_len, best_classifier,
                                                      best_classifier_size,
                                                      ks_const, target_digit)
    # Extract stats of the best tree
    best_bit = fit_vec.index(min(fit_vec))

    tp_best = stat_matrix[best_bit][0]
    tn_best = stat_matrix[best_bit][1]
    fp_best = stat_matrix[best_bit][2]
    fn_best = stat_matrix[best_bit][3]
    fit_best = min(fit_vec)
    TPR = tp_best / pdata  # sensitivity or true positive rate
    TNR = tn_best / ndata  # specificity or true negative rate
    FPR = fp_best / ndata  # false positive rate
    FNR = fn_best / pdata  # false negative rate

    results_str = data_set_type + ":\t" + \
                  "FPR = " + str(fp_best) + "/" + str(ndata) + " (TPR% " + str(round(100 * TPR, 2)) + ")\t" + \
                  "FNR = " + str(fn_best) + "/" + str(pdata) + " (TNR% " + str(round(100 * TNR, 2)) + ")\t" + \
                  "fit\t" + str(round(fit_best, 6)) + "\t" \
                  "size\t" + str(best_classifier_size) + "\t" \
                  "bit\t" + str(best_bit) + "\n"

    print(results_str)

    # open the dump file or create it
    f = open(dump_file, "a")
    f.write(results_str)
    f.close()
    return


def stampaValori(toolbox, hof, eval_=True):

    if eval_:
        bigall3_matrix = []
        with open(dir_path+"\\dataset\\bigall_ul104(reduced).csv", newline='\n') as bigall3_file:
            bigall3_data = csv.reader(bigall3_file, delimiter=';')
            for row in bigall3_data:
                bigall3_matrix.append([int(row[0]),int(row[1]),int(row[2])])

        a = hof[0]
        b = hof[0]
        c = hof[0]
        for ind in hof:
            if (ind.fitness.weights[0] < 0):
                if (ind.fitness.values[0] < a.fitness.values[0]):       # restituisce l'ind in hof con il primo parametro minore
                    a = ind
            else:
                if (ind.fitness.values[0] > a.fitness.values[0]):
                    a = ind
            if (ind.fitness.weights[1] < 0):
                if (ind.fitness.values[1] < b.fitness.values[1]):       # restituisce l'ind in hof con il secondo parametro maggiore
                    b = ind
            else:
                if (ind.fitness.values[1] > b.fitness.values[1]):
                    b = ind
            
            if (abs(ind.fitness.values[1] - ind.fitness.values[0]) > abs(c.fitness.values[1] - c.fitness.values[0])):       # restituisce l'ind in hof con la distanza fra i due maggiore
                c = ind

        for ind in [a, b, c]:
            print(evaluate_differentiator(toolbox, ind, bigall3_matrix, eval_))


def evaluate_differentiator(toolbox, best_embedding, bigall3_matrix, onlyMed=False):

    embed = toolbox.compile(best_embedding)
    trasformata = []
    # lista di liste: [[label, HDsameclass, countsameclass, HDdiffclass, countdiffclass],...]
    list_counterHD = [[0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [2, 0, 0, 0, 0],
                      [3, 0, 0, 0, 0],
                      [4, 0, 0, 0, 0],
                      [5, 0, 0, 0, 0],
                      [6, 0, 0, 0, 0],
                      [7, 0, 0, 0, 0],
                      [8, 0, 0, 0, 0],
                      [9, 0, 0, 0, 0],
                      ["Medium", 0, 0, 0, 0]]
    row_num = len(bigall3_matrix)
    for j in range(0, row_num):
        ARG0 = bigall3_matrix[j][0]
        ARG1 = bigall3_matrix[j][1]
        label = bigall3_matrix[j][2]

        tree_result = embed(ARG0, ARG1)
        trasformata.append([tree_result, label])

    for i in range(0, row_num):
        for j in range(i+1, row_num):
            x = trasformata[i][0] ^ trasformata[j][0]
            h_d = bin(x).count("1")
            if (trasformata[i][1] == trasformata[j][1]):
                list_counterHD[trasformata[i][1]][1] *= list_counterHD[trasformata[i][1]][2]
                list_counterHD[trasformata[i][1]][1] += h_d
                list_counterHD[trasformata[i][1]][2] += 1
                list_counterHD[trasformata[i][1]][1] /= list_counterHD[trasformata[i][1]][2]
            else:
                list_counterHD[trasformata[i][1]][3] *= list_counterHD[trasformata[i][1]][4]
                list_counterHD[trasformata[i][1]][3] += h_d
                list_counterHD[trasformata[i][1]][4] += 1
                list_counterHD[trasformata[i][1]][3] /= list_counterHD[trasformata[i][1]][4]

                list_counterHD[trasformata[j][1]][3] *= list_counterHD[trasformata[j][1]][4]
                list_counterHD[trasformata[j][1]][3] += h_d
                list_counterHD[trasformata[j][1]][4] += 1
                list_counterHD[trasformata[j][1]][3] /= list_counterHD[trasformata[j][1]][4]
    
    for l in list_counterHD:
        if(l[0]=="Medium"):
            break
        list_counterHD[10][1] += (l[1]*l[2])
        list_counterHD[10][2] += l[2]
        list_counterHD[10][3] += (l[3]*l[4])
        list_counterHD[10][4] += l[4]
    list_counterHD[10][1] /= list_counterHD[10][2]
    list_counterHD[10][3] /= list_counterHD[10][4]
    result_string = ""
    result_string += "-"*50+"\n"
    result_string += "Distanze di Hamming: \n"
    for l in list_counterHD:
        if ((not onlyMed) or (l[0] == "Medium")):
            result_string += "Class {} =>\n           Hamming Distance (same class) = {}\n           Hamming Distance (diff class) = {}\n".format(l[0], round(l[1], 2), round(l[3], 2))
            try:
                result_string += "           Ratio = {}\n".format(round(l[1]/l[3], 2))
            except:
                result_string += "           Ratio = +oo\n"
    

    result_string += "-"*50+"\n"
    result_string += "\n\n"
    return result_string


def plot(toolbox, pop):
    
    bigall3_matrix = []
    with open(dir_path+"\\dataset\\bigall_ul104(reduced).csv", newline='\n') as bigall3_file:
        bigall3_data = csv.reader(bigall3_file, delimiter=';')
        for row in bigall3_data:
            bigall3_matrix.append([int(row[0]),int(row[1]),int(row[2])])
    row_num = len(bigall3_matrix)
    
    x, y, a, b= [], [], [], []
    print("Lunghezza pop: {}".format(len(pop)))
    for ind in pop:
        
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
        a.append(ind.fitness.values[0])
        b.append(ind.fitness.values[1])

    seaborn.set(style='whitegrid')
    seaborn.set_context('notebook')
    plt.subplot(121)
    plt.plot(x, y, 'r.', label='Pop', alpha=0.7)
    plt.xlabel('HDs')
    plt.ylabel('HDd')
    plt.title("Valori HD")
    mxy = max(y)+2
    mxx = max(x)+2
    plt.axis([0, mxx, 0, mxy])
    plt.legend()

    plt.subplot(122)
    plt.plot(a, b, 'r.', label='Funz', alpha=0.7)
    plt.xlabel('$f_1(\mathbf{x})$')
    plt.ylabel('$f_2(\mathbf{x})$')
    #plt.xlabel('f1')
    #plt.ylabel('f2')
    plt.title("Valori Population")
    mxy = max(b)+2
    mxx = max(a)+2
    mny = min(b)-2
    mnx = min(a)-2
    plt.axis([mnx, mxx, mny, mxy])
    plt.legend()
    plt.show()