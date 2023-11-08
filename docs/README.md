# SmcGP_deap
DEAP implementation of Sub-machine Code GP

A presentation of the project can be found in the /docs folder.

DEAP homepage and sources at https://pypi.org/project/deap/

To install the latest version:
pip install git+https://github.com/DEAP/deap@master

DEAP documentation at https://deap.readthedocs.io/en/master/

###

### Needed packages

#### DEAP
see [deap installation guide]
(https://deap.readthedocs.io/en/master/installation.html).

#### Graphics
To plot the tree `pygraphviz' is needed. It has some dependecies
(assuming python3)

```bash
sudo apt-get install python3-dev graphviz libgraphviz-dev pkg-config
sudo pip install pygraphviz
```

### Datasets

Datasets of digits taken from license plate images are provided, in
which each line represents a different pattern along with the
corresponding digit/label (last number in each row). The patterns are
encoded as:

- four 32-bit words (as in "": scanning the patterns row-wise, the
  first 3 words encode lines 1-3, 4-6, and 7-9 of the input pattern,
  respectively, into the least significant 24 bits; the fourth 32-bit
  word encodes the last 4 rows of(10-13) of the input pattern)
  bigall32c.csv and bigalltest32c.csv

- two 64-bit words (scanning the pattern row-wise, the first word
  encodes lines 1-6 of the input pattern into its least significant 48
  bits, the second word encodes lines 7-13 into its least significant
  56 bits).
  bigall32c_64_int.csv  and  bigalltest32c_64_int.csv

- 104 bits (13 rows x 8 columns of the input pattern scanned row-wise)
  These files may be helpful to examine/plot the patterns with no need
  to convert the integers back into bits.
  bigall32c_104.csv    and    bigalltest32c_104.csv

### Run training and test

`Smc_GP_main.py`: this script performs the following

1. Initilizes a dictionary which holds the parameters.

2. Runs `Smc_GP_test.py`, which trains the classifier.

3. Runs `eval_best_tree.py`, which runs the best classifier on the
   test set and dumps the results.

Each of these scripts is autonomous and data are passed between them
using `.pkl` files.

### NOTES

The file `eaSimple_cp.py` contains a modified version of
[eaSimple](https://github.com/DEAP/deap/blob/master/deap/algorithms.py),
which dumps periodic checkpoints of the evolution.

##############################################################
NB The scripts provided work ONLY with 64-bit encoded data!!!!
##############################################################

To use 32-bit data it should be enough (not guaranteed) to modify the
files Smc_GP_training.py, fitness_fun.py, and eval_best_tree.py as
follows:

- change 'word_len' into 32 in Smc_GP_main.py

- change the fitness function prototype to require 4 parameters
  pset = gp.PrimitiveSetTyped("MAIN", [int, int, int, int], int)
  in Smc_GP_training.py and eval_best_tree.py

- change the input file names
  in Smc_GP_training.py and eval_best_tree.py

- read 4 integers per pattern from the input files instead of 2
     for j in range(0, row_num):
        ARG0 = bigall3_matrix[j][0]
        ARG1 = bigall3_matrix[j][1]
        ARG2 = bigall3_matrix[j][2]
        ARG3 = bigall3_matrix[j][3]
        label = bigall3_matrix[j][4]
  in Smc_GP_training.py and fitness_fun.py

