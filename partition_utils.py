# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collections of partitioning functions."""

import time
import metis
import scipy.sparse as sp
import scipy
import tensorflow.compat.v1 as tf
import numpy as np
import networkx as nx
import networkx.convert_matrix
#import community
from networkx.algorithms import community
import random
"""
adj:          visible data adj
idx_nodes:    visible data
num_cluster:  partition clusters
"""

#Deepak's code start

def dfs_partition(G, num_partitions):

	size_per_partition = len(G.nodes)/num_partitions
	total_num_nodes = len(G.nodes)
	partitions = []
	stack = []
	visited_nodes_count = 0
	visited_nodes_set = set()
	partition=[]
	unvisited_nodes_set = set(G.nodes.keys())
	while visited_nodes_count < total_num_nodes:
		random_node = random.sample(unvisited_nodes_set, 1)
		stack.append(random_node[0])
		while len(stack)>0:
			next_node = stack[len(stack)-1]
			stack.pop(len(stack)-1)
			visited_nodes_set.add(next_node)
			visited_nodes_count = visited_nodes_count + 1
			unvisited_nodes_set.discard(next_node)
			partition.append(next_node)
			if len(partition) >= size_per_partition and len(partitions) < num_partitions:
				partitions.append(partition)
				partition=[]
				stack=[]
				break
			random_neighbor = -1
			for neighbor in list(G[next_node].keys()):
				#print("neighbor", neighbor)
				if neighbor in unvisited_nodes_set:
					random_neighbor = neighbor
					stack.append(random_neighbor)
			
			if random_neighbor == -1:
				continue

	if len(partition) > 0:
		partitions.append(partition)
	#assign each node to partition
	node_partitions = [0] * len(G.nodes)
	i=0
	for partition in partitions:
		for node in partition:
			node_partitions[node] = i
		i = i+1
	return node_partitions


def partition_graph1(G, num_partitions):
  mode=1
  res={}
  if mode == 0:
    #louvain
    communities = community.louvain_communities(G, threshold=10000)
    print("communities", sorted([len(c) for c in communities]))
    groups = [0] * len(adj[0].toarray())
  else:
    #greeding community detection algorithm
    c = nx.community.greedy_modularity_communities(G, cutoff = num_partitions)
    print("len(c)", len(c))
    groups = [0] * len(G.nodes)
    i=0

    for partition in c:
      for node in partition:
        groups[node] = i
      i = i+1
    a=0

  return groups

def get_freq(groups):
  freq={}
  for g in groups:
    component_size = g
    if component_size in freq:
      freq[component_size] = freq[component_size] + 1
    else:
      freq[component_size] = 1
  print("groups", freq)
  #print(np.array(train_adj_lists).shape)

#Deepak's code end

def partition_graph(adj, idx_nodes, num_clusters):
  """partition a graph by METIS."""

  start_time = time.time()
  num_nodes = len(idx_nodes)    # visible nodes
  num_all_nodes = adj.shape[0]  # all nodes

  neighbor_intervals = []
  neighbors = []
  edge_cnt = 0
  neighbor_intervals.append(0)
  print("idx_nodes", idx_nodes)
  train_adj_lil = adj[idx_nodes, :][:, idx_nodes].tolil() # get diag matrix in incresing order
  train_ord_map = dict()
  train_adj_lists = [[] for _ in range(num_nodes)]
  for i in range(num_nodes):
    rows = train_adj_lil[i].rows[0]
    # self-edge needs to be removed for valid format of METIS
    if i in rows:
      rows.remove(i)
    train_adj_lists[i] = rows
    neighbors += rows
    edge_cnt += len(rows)
    neighbor_intervals.append(edge_cnt)
    train_ord_map[idx_nodes[i]] = i # (old_idx, new_idx)

  if num_clusters > 1:
    #_, groups = metis.part_graph(train_adj_lists, num_clusters)
    groups = dfs_partition(nx.from_scipy_sparse_array(train_adj_lil), num_clusters)
    #print("groups", set(groups))
    #print("groups1", set(groups1))
    #get_freq(groups1)
  else:
    groups = [0] * num_nodes

	
  part_row = []
  part_col = []
  part_data = []
  #print("train_adj_lists-", train_adj_lists[0:10])
  parts = [[] for _ in range(num_clusters)]
  for nd_idx in range(num_nodes):
    gp_idx = groups[nd_idx]
    nd_orig_idx = idx_nodes[nd_idx]
    parts[gp_idx].append(nd_orig_idx)
    for nb_orig_idx in adj[nd_orig_idx].indices:  # neighbors in orig adj
      nb_idx = train_ord_map[nb_orig_idx]
      if groups[nb_idx] == gp_idx:  # retain intra-cluster edges
        part_data.append(1)
        part_row.append(nd_orig_idx)
        part_col.append(nb_orig_idx)
  # padding zero for unvisible nodes
  part_data.append(0)
  part_row.append(num_all_nodes - 1)
  part_col.append(num_all_nodes - 1)
  part_adj = sp.coo_matrix((part_data, (part_row, part_col))).tocsr()

  tf.logging.info('Partitioning done. %f seconds.', time.time() - start_time)
  return part_adj, parts
