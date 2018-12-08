def import_statements():
    return 'import os;' + \
           'import numpy as np;' + \
           'import enn_v1_0;' + \
           'import simple_nn;' + \
           'import nn_visualizer as nv;' + \
           'import feature_visualization as fv;' + \
           'from importlib import reload;' + \
           'reload(enn_v1_0);' + \
           'reload(simple_nn);' + \
           'from enn_v1_0 import ENN;' + \
           'from simple_nn import NN;' + \
           'reload(fv);' + \
           'reload(nv);'
           