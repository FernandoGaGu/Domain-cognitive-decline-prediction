import os
from pathlib import Path 

# path to the lib directory 
PATH_TO_LIB = Path('..').resolve() 

# version of the gojo library used internally by the library
GOJO_VERSION = 'gojo_v0_1_6'

# key used to identify the models
FFN_MODEL_KEY = 'FFN'
CNN_MODEL_KEY = 'CNN'
GNN_MODEL_KEY = 'GNN'
