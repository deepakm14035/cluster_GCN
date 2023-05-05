import networkx as nx
import utils
from networkx.readwrite import json_graph
import json
from tensorflow.compat.v1 import gfile
import tensorflow.compat.v1 as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt

def draw_colored_graph(G, communities):
    print(communities)
    color_map=[]
    colors=['red', 'green', 'blue', 'maroon', 'black', 'magenta', 'yellow', 'orange', 'gray', 'brown', 'cyan', 'purple', 'pink']
    for node in G:
        #print(node)
        i=0
        for community in communities:
            if node in community:
                #print(node, colors[i], community)
                color_map.append(colors[i])
            i=i+1
    nx.draw(G, node_color=color_map, with_labels=True)
    plt.show()


def num_of_self_loops(graph):
    count=0
    for edge in graph.edges():
        if(edge[0] ==edge[1]):
            count=count+1

    return count

def get_counts(G):
    return [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=False)]

def get_num_of_test_nodes(G):
    test_data = np.array(
      [n for n in G.nodes() if G.nodes[n]['test'] == True],
      dtype=np.int32)
    return len(test_data)

def get_graph(file):
    graph_json = json.load(gfile.Open(file))
    graph_nx = json_graph.node_link_graph(graph_json)
    print("number of nodes: ", len(graph_nx.nodes))
    print("number of edges: ", len(graph_nx.edges))
    print("number of self loops: ", num_of_self_loops(graph_nx))
    print("number of connected components: ", nx.number_connected_components(graph_nx))
    print("size of connected components: ", get_counts(graph_nx))
    print("number of test nodes: ", get_num_of_test_nodes(graph_nx))
    #print("nodes", graph_nx.nodes)
    
    return graph_nx

graph = get_graph("data/ppi/ppi-G.json")
#graph = get_graph("data/ppi/ppi-G.json")
#nx.draw_networkx(graph, node_size=10, with_labels=True)
#plt.show()