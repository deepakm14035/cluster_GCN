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
      [n for n in G.nodes() if 'test' in G.nodes[n] and G.nodes[n]['test'] == True],
      dtype=np.str)
    return len(test_data)

def get_num_of_validation_nodes(G):
    val_data = np.array(
      [n for n in G.nodes() if 'val' in G.nodes[n] and G.nodes[n]['val'] == True],
      dtype=np.str)
    return len(val_data)

def get_cen(G):
    p = nx.closeness_centrality(G)        
    for a,b in p:
        s.append(b)
    plt.bar(G.nodes,s, align='center', alpha=1)
    plt.show()
    #cen_deg = nx.degree_centrality(G)
    #print(cen_deg)
    #cen_betw = nx.betweenness_centrality(G)
    #print(cen_betw)
    #cen_eig = nx.eigenvector_centrality(G)
    #print(cen_eig)
    #cen_katz = nx.katz_centrality(G, alpha=0.1, beta=1.0, max_iter=1000, tol=1e-02, nstart=None, normalized=True, weight=None)
    #print(cen_katz)


def plot_centralities(graph):
    fig = plt.figure()
    #ax = fig.add_axes([0,0,1,1])
    langs = ['C', 'C++', 'Java', 'Python', 'PHP']
    students = [23,17,35,29,12]
    ax.bar(langs,students)
    plt.show()

def get_diameters(G):
    components = [c for c in sorted(nx.connected_components(G), key=len, reverse=False)]
    for c in components:
        new_graph = G.subgraph(c)
        print("diameter: ", nx.diameter(new_graph))

def plot_dict(d):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(list(d.keys()),list(d.values()))
    plt.show()

def get_connected_components_freq(G):
    components = [c for c in sorted(nx.connected_components(G), key=len, reverse=False)]
    freq={}
    for c in components:
        component_size = len(c)
        if component_size in freq:
            freq[component_size] = freq[component_size] + 1
        else:
            freq[component_size] = 1

    #plot_dict(freq)
    print("connected_components_freq", freq)

def get_graph(file):
    graph_json = json.load(gfile.Open(file))
    graph_nx = json_graph.node_link_graph(graph_json)
    print("number of nodes: ", len(graph_nx.nodes))
    print("number of edges: ", len(graph_nx.edges))
    print("number of self loops: ", num_of_self_loops(graph_nx))
    #print("number of connected components: ", nx.number_connected_components(graph_nx))
    print("size of connected components: ", get_counts(graph_nx))
    #get_diameters(graph_nx)
    print("number of test nodes: ", get_num_of_test_nodes(graph_nx))
    print("number of validation nodes: ", get_num_of_validation_nodes(graph_nx))
    get_connected_components_freq(graph_nx)
    #print("nodes", graph_nx.nodes)
    
    return graph_nx




graph = get_graph("data/ppi/ppi-G.json")
#get_cen(graph)
graph = get_graph("data/ppi/ppi-G.json")
nx.draw_networkx(graph, node_size=10, with_labels=True)
plt.show()