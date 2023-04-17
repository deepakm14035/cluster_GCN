import networkx as nx
import utils
from networkx.readwrite import json_graph
import json
from tensorflow.compat.v1 import gfile
import tensorflow.compat.v1 as tf
import numpy as np
import random

def generate_sample_graph(G, path, name):
    print(G.nodes)
    sampled_nodes = random.sample(G.nodes, 20)
    sampled_graph = G.subgraph(sampled_nodes)
    print(sampled_graph.nodes)
    name = name+'_new'
    print(json_graph.node_link_data(sampled_graph))
    


dataset_path = 'data'
dataset_str = 'ppi'

graph_json = json.load(
    gfile.Open('{}/{}/{}-G.json'.format(dataset_path, dataset_str,
                                        dataset_str)))
graph_nx = json_graph.node_link_graph(graph_json)

id_map = json.load(
    gfile.Open('{}/{}/{}-id_map.json'.format(dataset_path, dataset_str,
                                            dataset_str)))
is_digit = list(id_map.keys())[0].isdigit()
id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}
class_map = json.load(
    gfile.Open('{}/{}/{}-class_map.json'.format(dataset_path, dataset_str,
                                                dataset_str)))

is_instance = isinstance(list(class_map.values())[0], list)
class_map = {(int(k) if is_digit else k): (v if is_instance else int(v))
            for k, v in class_map.items()}

broken_count = 0
to_remove = [ ]
for node in graph_nx.nodes():
    if node not in id_map:
        to_remove.append(node)
        broken_count += 1
for node in to_remove:
    graph_nx.remove_node(node)


feats = np.load(
    gfile.Open(
        '{}/{}/{}-feats.npy'.format(dataset_path, dataset_str, dataset_str),
        'rb')).astype(np.float32)


edges = []
for edge in graph_nx.edges():
    if edge[0] in id_map and edge[1] in id_map:
        edges.append((id_map[edge[0]], id_map[edge[1]]))
num_data = len(id_map)

generate_sample_graph(graph_nx, dataset_path, dataset_str)