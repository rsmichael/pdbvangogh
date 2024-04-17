import pytest
import sys
import os
from pdbvangogh import api
import tensorflow as tf
import random
import numpy as np
from pdbvangogh.data_models import *

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def test_test_func():
    assert(True)
    assert(api.test_func())
def test_pdbvangogh():
    api.pdbvangogh(background_image = 'tests/in/pdx.png', 
                   content_image = 'tests/in/2l1v.png', 
                   style_image = 'tests/in/starry_night.png',
                   save_prefix = 'tests/out/2l1v_pdx_starry_night',
                   content_size=100,
                   background_size = 200,
                   background_hyperparameters=gatys_transfer_parameters(style_weight=1e-2, epochs=1, steps_per_epoch=50),
                   content_hyperparameters=gatys_transfer_parameters(style_weight=1e-4, epochs=1, steps_per_epoch=50)
                   )
    api.pdbvangogh(background_image = 'tests/in/pdx.png', 
                   pdb_id = '2HYY',
                   out_dir = 'tests/out/',
                   style_image = 'tests/in/starry_night.png',
                   save_prefix = 'tests/out/2HYY_pdx_starry_night',
                   content_size=100,
                   background_size = 200,
                   background_hyperparameters=gatys_transfer_parameters(style_weight=1e-2, epochs=1, steps_per_epoch=50),
                   content_hyperparameters=gatys_transfer_parameters(style_weight=1e-4, epochs=1, steps_per_epoch=50)
                   )
    assert(True)
