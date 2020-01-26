#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy.random as rn
import pandas as pd
import math
import random
import sys 
import networkx as nx
import matplotlib.pyplot as plt  # to plot
import matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor
import collections
from dataclasses import dataclass
import time
from pprint import pprint    
from scipy import optimize 


# In[2]:


d1 = pd.read_csv("traffic_complement.csv",header=None)
d2 = pd.read_csv("traffic_rand.csv",header=None)
d3 = pd.read_csv("traffic_uniform.csv",header=None)
#np.set_printoptions(threshold=sys.maxsize)


# In[3]:


def hop_count(src,trg,G):
    
    Path=nx.dijkstra_path(G, src, trg, weight='weight')   #usign dijkstra's path to calculaate the hop count
    hops=len(Path)-1
    #print(hops)
    #print(A)
    
    return hops


# In[4]:


def adjacency_matrix(N):
    
    G=nx.grid_graph(dim=[N,N], periodic=False)
    adjmat=nx.to_numpy_matrix(G, nodelist=None, dtype=None, order=None, multigraph_weight= sum, weight='weight')
    G=nx.from_numpy_matrix(adjmat, create_using=None)
    
    return adjmat,G
    
    


# In[5]:


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
        adj_mat,G, t_mat= link_swap(G1)
    else:   
      
        
        G1.add_edge(src,dest)                           
        G1.remove_edge(src,removal_node)            # as we are removing and adding 1 edge at a time the total links in the mesh is always 112.
        
        if nx.has_path(G1, src, removal_node):      # checking if a path still exists between the source node and the neighbour node after removal of the edge between them
            adj_mat=nx.to_numpy_matrix(G1, nodelist=None, dtype=int, order=None, multigraph_weight= sum, weight='weight')
            G=nx.from_numpy_matrix(adj_mat, create_using=None)
            t_mat=[i for i in G.nodes()]
        else:                                       #if there is no path replace the edges and call link_swap again
            G1.remove_edge(src,dest)
            G1.add_edge(src,removal_node)
            adj_mat , G,t_mat = link_swap(G1)
            
        
    return adj_mat,G, t_mat
   


# In[8]:


def tile_swap(G2,t_mat):
    
    a=t_mat
    node1= random.choice(a)
    node2= random.choice(a)
    while node1==node2:
        node2= random.choice(a)
    a[node1],a[node2]=a[node2],a[node1]
    #print(node1,node2,a)
    mp=dict(zip(G2, a))
    G2=nx.relabel_nodes(G2, mp)
    t_mat=[i for i in G2.nodes()]
    adj_mat=nx.to_numpy_matrix(G2, nodelist=None, dtype=int, order=None, multigraph_weight= sum, weight='weight')
    #G=nx.from_numpy_matrix(adj_mat, create_using=None)
    
    return adj_mat, G2, t_mat
    


# In[9]:


def linklength(src,trg,G):
    
    a=[n for n in G.nodes()]
    src1=a.index(src)
    trg1=a.index(trg)
    xs,ys= divmod(a[src1],8)
    xt,yt= divmod(a[trg1],8)
    xlen=abs(xs-xt)
    ylen=abs(ys-yt)
    l=xlen**2 + ylen**2
    l=l**0.5
    l=math.ceil(l)
    
    return l


# In[10]:


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


# In[11]:


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
            if o==0 and j<(0.2*num_iter):             #does tile swap for only 20 percent of number of iterations
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
                
    


def STAGE(df,num_iter,counter):
    
    '''STAGE algorithm '''
    start=time.time()
    N=8;
    m=3;
    
    Xmemory=[]                                                            #the training data and lables are appended to Xmemory and Ymemory 
    Ymemory=[]                                                            # for each iteration of STAGE
    
    scurr, gphcurr=adjacency_matrix(N)
    
    d2=zero_load_latency(df,gphcurr,m)
   
    Tcurr=[i for i in range(64)]
    
    print("STAGE in progress --------> Starting latency: ",d2)
    
    for i in range(num_iter):
        
        # Xmemory and Ymemory is appended for each iteration with the previous data and the last design
        # is fed as base to the new design
        costq=collections.deque(maxlen=counter)
        costq.clear()                                                    # clear the deque for each run of stage
        
        predictq=collections.deque(maxlen=counter)
        predictq.clear()                                                 # clear the deque for each run of stage
        
        training_data=[]
        
        
        for i in range(num_iter):
            o=random.randint(0,1)
            
            if o==0:                                         
                sneigh, gphneigh, Tneigh=tile_swap(gphcurr, Tcurr)             #tile swap
                                                                
            else:
                sneigh, gphneigh, Tneigh=link_swap(gphcurr)                    #link swap

            d1=zero_load_latency(df,gphneigh,m)                                #Cost function (zero load latency)

            delta= d1 - d2                          
            if d1 < d2:                                                        #Minimize the cost function
                gphcurr=gphneigh
                scurr=sneigh
                Tcurr=Tneigh
                d2=d1
            
            current_feature=[]
            feature=[]
            
            adjmat=np.asarray(scurr)
                                                                            
            upr_tngle=adjmat[np.triu_indices(64, k = 1)]                      # Get the upper triangular of adjacency matrix
            input_features=[upr_tngle,Tcurr]
            feature=[x for y in input_features for x in y]
            for i in feature:
                current_feature.append(i)
            
            Xmemory.append(current_feature)                                   # append the database for the regressor
            training_data.append(current_feature)        
            costq.append(d2)
            if len(costq)==counter:
                val=costq[0]
                min_val=val-100
                max_val=val+100
                count=0
                for i in range(len(costq)):
                    if (min_val < costq[i]) and (max_val > costq[i]):
                        count=count+1
                if count==counter:
                    d2 = costq[counter-1]
                    break
        
        xTrain=np.array(Xmemory)
        len_training=len(training_data)
       
        label = [d2]*len_training                                           #the cost value of the last design are kept as the labes for the dictionary
        for i in label:
            Ymemory.append(i)
       
        yTrain = Ymemory
        regressor = RandomForestRegressor(n_estimators=20, random_state=0)  #Training the regeressor
        regressor.fit(xTrain, yTrain)  
        
        Pcurr=d2
        
        
        for i in range(num_iter):
           
            training_Sneigh=[]
            current_feature1=[]
            feature1=[]
            o=random.randint(0,1)
            
            if o==0 :                                                        
                sneigh, gphneigh, Tneigh=tile_swap(gphcurr, Tcurr)           #tile swap
                  
            else:
                sneigh, gphneigh, Tneigh=link_swap(gphcurr)                  #link swap


            adjmat=np.asarray(sneigh)
            
            upr_tngle1=adjmat[np.triu_indices(64, k = 1)]
            input_features1=[upr_tngle1,Tneigh]
            feature1=[x for y in input_features1 for x in y]
            
            for i in feature1:
                current_feature1.append(i)
            training_Sneigh.append(current_feature1)
            
            Pneigh = regressor.predict(training_Sneigh)
            
            if Pneigh < Pcurr:
                scurr=sneigh
                Pcurr=Pneigh
                Tcurr=Tneigh
            
            predictq.append(Pcurr)
            if len(predictq)==counter:

                predict_val=predictq[0]
                min_predict_val=predict_val-100
                max_predict_val=predict_val+100
                countP=0
                for i in range(len(predictq)):
                    if (min_predict_val < predictq[i]) and (max_predict_val > predictq[i]):
                        countP=countP+1
                if countP==counter:
                    Pcurr = predictq[counter-1]
                    break   
        
        print("New Design's latency",d2)
    end=time.time()
    print("Total time taken ",end-start)
    return scurr,Tcurr,d2


# In[23]:


Adjacency_mat,Task_mat,Latency=STAGE(d1,100,50)
print(Task_mat)
print(Adjacency_mat)
np.savetxt("Adjacency Matrix for Complement traffic STAGE.csv", Adjacency_mat, delimiter=",")


# In[23]:


Adjacency_mat,Task_mat,Latency=STAGE(d2,100,50)
print(Task_mat)
print(Adjacency_mat)
np.savetxt("Adjacency Matrix for Random traffic STAGE.csv", Adjacency_mat, delimiter=",")


# In[23]:


Adjacency_mat,Task_mat,Latency=STAGE(d3,100,50)
print(Task_mat)
print(Adjacency_mat)
np.savetxt("Adjacency Matrix for Uniform traffic STAGE.csv", Adjacency_mat, delimiter=",")





