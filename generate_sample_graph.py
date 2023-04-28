import networkx as nx
import utils
from networkx.readwrite import json_graph
import json
from tensorflow.compat.v1 import gfile
import tensorflow.compat.v1 as tf
import numpy as np
import os
import random

def get_largest_component(G):
    sorted_components_list = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    sampled_graph = G.subgraph(sorted_components_list[2])
    return sampled_graph


def generate_sample_graph(G, path, name, class_map, feats):
    #print(feats)
    #print(G.nodes)

    sampled_nodes = random.sample(G.nodes, 100)
    sampled_graph = G.subgraph(sampled_nodes)
    
    test_nodes = np.array(
      [id_map[n] for n in sampled_graph.nodes() if graph_nx.nodes[n]['test'] == True],
      dtype=np.int32)
    print("test_nodes", test_nodes)
    name = name+'_new'
    nodes_dict={}
    new_feats = np.zeros((len(sampled_nodes), feats.shape[1]))
    #print("new_feats", new_feats)
    i=0
    for node in sampled_graph:
        if(node in test_nodes):
            continue
        nodes_dict[node] = i
        new_feats[i] = feats[node]
        i=i+1

    for node in test_nodes:
        nodes_dict[node] = i
        new_feats[i] = feats[node]
        i=i+1
    #print("nodes_dict", nodes_dict)

    new_nodes_dict={}
    for node in sampled_graph:
        new_nodes_dict[str(nodes_dict[node])]=nodes_dict[node]
    sampled_graph = nx.relabel_nodes(sampled_graph, nodes_dict)

    if not os.path.exists(path+"/"+name):
        os.makedirs(path+"/"+name)
    f = open(path+"/"+name+"/" + name+'-id_map.json', 'w')
    json.dump(new_nodes_dict, f)


    #sampled_graph = get_largest_component(G)
    #print(sampled_graph.nodes)
    #print(json_graph.node_link_data(sampled_graph))
    json.dump(json_graph.node_link_data(sampled_graph), open(path+"/"+name+"/" + name+'-G.json', 'w'))


    new_class_map={}
    for node in sampled_graph:
        new_class_map[str(node)] = class_map[str(node)]
    f = open(path+"/"+name+"/" + name+'-class_map.json', 'w')
    json.dump(new_class_map, f)

    print("feats", new_feats.shape)
    np.save(path+"/"+name+"/" + name+'-feats.npy', new_feats)



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

generate_sample_graph(graph_nx, dataset_path, dataset_str, class_map, feats)