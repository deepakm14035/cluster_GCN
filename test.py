#Deepak's code start

import networkx as nx
import numpy as np
from tensorflow.compat.v1 import gfile

dataset_path = 'data'
dataset_str = 'ppi'
feats = np.load(
    gfile.Open(
        '{}/{}/{}-feats.npy'.format(dataset_path, dataset_str, dataset_str),
        'rb')).astype(np.float32)

print("feats.npy", feats[0])
print("feats.npy", len(feats[0]))

#Deepak's code end
