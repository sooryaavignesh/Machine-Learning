import numpy as np
import numpy.random as rn
import pandas as pd
import math
import random
import sys 
import networkx as nx
import matplotlib.pyplot as plt  # to plot
import matplotlib as mpl

from pprint import pprint    
from scipy import optimize 
d1 = pd.read_csv("traffic_complement.csv",header=None)
d2 = pd.read_csv("traffic_rand.csv",header=None)
d3 = pd.read_csv("traffic_uniform.csv",header=None)
np.set_printoptions(threshold=sys.maxsize)

def hop_count(src,trg,G):
    
    Path=nx.dijkstra_path(G, src, trg, weight='weight')   #using dijkstra's path to calculate the hop count
    hops=len(Path)-1
    #print(hops)
    #print(A)
    
    return hops

def adjacency_matrix(N):
    
    G=nx.grid_graph(dim=[N,N], periodic=False)
    adjmat=nx.to_numpy_matrix(G, nodelist=None, dtype=int, order=None, multigraph_weight= sum, weight='weight')
    G=nx.from_numpy_matrix(adjmat, create_using=None)
    
    return adjmat,G
    
    
def link_swap(G1):
    a1=[0]
    a=[0]
    while len(a1)==1:                                # making sure the source node has atleast 1 connection
        x=[n for n in range(64)]
        src=random.choice(x)
        a=[n for n in G1.neighbors(src)]
        for i in a:                                  # making sure the neighbours of source node has atleast 1 connection
            a1=[n for n in G1.neighbors(i)]
            if len(a1)>1:
                removal_node=i
                break
        
            
    x.remove(src)                                    #choosing the destination nodes such that it is not alredy connected with the source node
    snh=[n for n in G1.neighbors(src)]
    for i in range(len(snh)):
        x.remove(snh[i])
    dest=random.choice(x)
    
    
    hops=5
        
    while hops>4:                                    #maximum link length(4) constraint
        hops=hop_count(src,dest,G1)
        if hops<5:
             
            break
        else:
            dest=random.choice(x)     
  
        
    if len(a)>7:                                     # maximum connections to a node can only be 7
        adj_mat,G= link_swap(G1)
    else:   
      
        
        G1.add_edge(src,dest)                           
        G1.remove_edge(src,removal_node)            # as we are removing and adding 1 edge at a time the total links in the mesh is always 112.
        
        if nx.has_path(G1, src, removal_node):      # checking if a path still exists between the source node and the neighbour node after removal of the edge between them
            adj_mat=nx.to_numpy_matrix(G1, nodelist=None, dtype=int, order=None, multigraph_weight= sum, weight='weight')
            G=nx.from_numpy_matrix(adj_mat, create_using=None)
        else:                                       #if there is no path replace the edges and call link_swap again
            G1.remove_edge(src,dest)
            G1.add_edge(src,removal_node)
            adj_mat , G= link_swap(G1)
            
        
    return adj_mat,G
    
def tile_swap(G2):
    
    a=[i for i in G2.nodes()]                      #getting the nodes of the a graph as the input and stores it in a list
    node1= random.choice(a)                        #random values are selected from the list
    node2= random.choice(a)                        
    while node1==node2:                            #condition for node1 not equal to node2
        node2= random.choice(a)
    a[node1],a[node2]=a[node2],a[node1]            #swapping nodes and storing in the same node
    #print(node1,node2,a)
    mp=dict(zip(G2, a))
    p=nx.relabel_nodes(G2, mp)                     #giving the list as lables to the graph
    adj_mat=nx.to_numpy_matrix(p, nodelist=None, dtype=int, order=None, multigraph_weight= sum, weight='weight')
    #G=nx.from_numpy_matrix(adj_mat, create_using=None)
    
    return adj_mat, p
    
def linklength(src,trg,G):
    
    a=[n for n in G.nodes()]                        #getting the nodes of the a graph as the input and stores it in a list
    src1=a.index(src)
    trg1=a.index(trg)
    xs,ys= divmod(a[src1],8)                        #divmod is used to get the quotiont and remainder of the source and target nodes
    xt,yt= divmod(a[trg1],8)                        
    xlen=abs(xs-xt)
    ylen=abs(ys-yt)
    l=xlen**2 + ylen**2
    l=l**0.5
    l=math.ceil(l)                                   #finding the link length between 2 nodes
    
    return l
    
def zero_load_latency(df,gph,m):
    #m=8
    f=df    #print(f)
    r, c = f.shape
    lat=0
    for i in range(c):
        for j in range(r):
            hops=hop_count(i,j,gph)
            l= linklength(i,j,gph)
            lat=lat+(m*hops+l)*(f.at[i,j]) 
            #print(lat)
    return lat
    
def annealing(t0,Tth,alpha,num_iter,df,N):
    
    t=t0
    m=3
    scurr, gphcurr=adjacency_matrix(N)
    laten=zero_load_latency(df,gphcurr,m)
    print("Annealing Started------->Starting Latency :",laten)
    j=0
    while t>Tth:
        for i in range(num_iter):
            o=random.randint(0,1)
            if o==0 and j<(0.40*num_iter):             #does tile swap for only 40 percent of number of iterations
                sneigh, gphneigh=tile_swap(gphcurr)   #tile swap
            else:
                sneigh, gphneigh=link_swap(gphcurr)   #link swap
            d1=zero_load_latency(df,gphneigh,m)       #Cost function (zero load latency)
            
            d2=zero_load_latency(df,gphcurr,m)        
           
            sys.stdout.write("\r{0}>%dPercent".format("="*i) % ((((i+1)/num_iter)*100)))
            sys.stdout.flush()
            
            delta= d1 - d2                          
            if d1 < d2:                                #Minimize the cost function
                gphcurr=gphneigh
            else:
                prob=math.exp(-delta/t)                #accept the bad configuration with a probability decreases with increase in temp
                if prob > random.random():
                    gphcurr=gphneigh
            j=j+1
        j=0
        print("\nTemparature-",t," latency-",d2)
        t=t*alpha
    scurr=nx.to_numpy_matrix(gphcurr, nodelist=None, dtype=int, order=None, multigraph_weight= sum, weight='weight')
    return scurr, gphcurr,d2
 
T=100.00
Tstop=0.1
alp=0.9
iterations=100
N=8
ADJACENCY_MATRIX, GRAPH, latency = annealing(T,Tstop,alp,iterations,d1,N)
print("*******************************")
print("minmum latency for complement traffic :\r",latency)
print(ADJACENCY_MATRIX)
print("*******************************")

pos = nx.spring_layout(GRAPH,scale=1,iterations=200) 
plt.figure()
nx.draw_networkx(GRAPH, pos,with_labels=True,node_size = 200)
tile_placement=[n for n in GRAPH.nodes()]
print("Tile placement vector :\n",tile_placement)

np.savetxt("Adjacency Matrix for Complement traffic.csv", ADJACENCY_MATRIX, delimiter=",")

T=100
Tstop=0.1
alp=0.9
iterations=100
N=8
ADJACENCY_MATRIX, GRAPH, latency = annealing(T,Tstop,alp,iterations,d2,N)
print("*******************************")
print("minmum latency for random trafic :\r",latency)
print(ADJACENCY_MATRIX)
print("*******************************")

pos = nx.spring_layout(GRAPH,scale=1,iterations=200) 
plt.figure()
nx.draw_networkx(GRAPH, pos,with_labels=True,node_size = 200)
tile_placement=[n for n in GRAPH.nodes()]
print("Tile placement vector :\n",tile_placement)

np.savetxt("Adjacency Matrix for Random traffic.csv", ADJACENCY_MATRIX, delimiter=",")

T=100
Tstop=0.1
alp=0.9
iterations=100
N=8
ADJACENCY_MATRIX, GRAPH, latency = annealing(T,Tstop,alp,iterations,d3,N)
print("*******************************")
print("minmum latency for uniform traffic :\r",latency)
print(ADJACENCY_MATRIX)
print("*******************************")


pos = nx.spring_layout(GRAPH,scale=1,iterations=200) 
plt.figure()
nx.draw_networkx(GRAPH, pos,with_labels=True,node_size = 200)
tile_placement=[n for n in GRAPH.nodes()]
print("Tile placement vector :\n",tile_placement)

np.savetxt("Adjacency Matrix for Uniform traffic.csv", ADJACENCY_MATRIX, delimiter=",")