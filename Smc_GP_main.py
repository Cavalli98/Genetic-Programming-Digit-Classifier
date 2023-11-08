import os
import pickle
import time

start_time = time.time()
elapsed_time = time.strftime('\nPartito Main - %H:%M:%S', time.gmtime(start_time))
print(elapsed_time)
print("-----------------------")

########################################################################################################################
# PARAMETRI
########################################################################################################################
params_dict = {
    'gen_n_diff': 500,        # numero di generazioni differenziatore
    'population_diff': 200,  # dimensione della popolazione iniziale differenziatore
    'gen_n_class': 1000,        # numero di generazioni classificatore
    'population_class': 1000,  # dimensione della popolazione iniziale classificatore
    'evolution': 0,   # 0 se solo traformazione dataset, 1 per evoluzione differenziatore

    'dataset_training_diff' : "\\dataset\\bigall_ul104(reduced).csv",
                        # nome dataset per training differenziatore
    'dataset_training_class' : "\\dataset\\embedded_bigall_ul104.csv",
                        # nome dataset per training classificatore

    'word_len': 64,     # dimensione della parola in uscita
    'hof_size' : 1,     # dimensione dell'halloffame, usata anche per Elitismo
    'fgen_minh': 1,     # minima altezza degli individui della prima generazione
    'fgen_maxh': 7,     # massima altezza degli individui della prima generazione
    'tournsize': 7,     # valore del parametro tournsize quando usato il metodo selTournament
    'mut_app_minh': 1,  # altezza minima appendibile di un albero durante la mutazione
    'mut_app_maxh': 6,  # altezza massima appendibile di un albero durante la mutazione
    'mate_maxh': 12,    # altezza massima raggiungibile durante un crossover
    'mut_maxh': 12,     # altezza massima raggiungibile durante una mutazione
    'mate_maxs': 4096,  # dimensione massima raggiungibile durante un crossover
    'mut_maxs': 4096,   # dimensione massima raggiungibile durante la mutazione
    'cxpb': 0.7,        # probabilità crossover
    'mutpb': 0.2,       # probabilità mutazione
    'use_erc': 1,       # variabile booleana: 1 usa le EphemeralConstant come terminali
    'use_const1': 1,    # variabile booleana: 1 usa la costante 0 come terminale
    'use_const0': 1,    # variabile booleana: 1 usa la costante 1 come terminale
    'k_const': 0.1,     # valore della penalità per individui di classi diverse ma simili
    'ks_const': 0.000001,
    'target_digit': 0
}

# save as pickle since dictionary can not be passed as sys.argv
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path+'\\pkl_dump\\params.pkl', 'wb') as output:
    pickle.dump(params_dict, output, -1)


if os.name == 'nt':
    # launching script on windows

    cmd = '"'+dir_path+"\\embedder_Training.py"+'"'
    os.system('{} {}'.format('python', cmd))

    cmd = '"'+dir_path+"\\Smc_GP_training.py"+'"'
    os.system('{} {}'.format('python', cmd))

    cmd = '"'+dir_path+"\\eval_best_tree.py"+'"'
    os.system('{} {}'.format('python', cmd))
