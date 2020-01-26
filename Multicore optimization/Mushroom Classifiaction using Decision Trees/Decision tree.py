#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import pandas as pd

from pprint import pprint


# In[75]:


#df = pd.read_csv("mushrooms_train_updated.csv")
#df = df.drop("stalk-root", axis=1)

#dt = pd.read_csv("mushrooms_test_updated.csv")
#df = df.drop("stalk-root", axis=1)
#l=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,0] # index order
#dt=dt[[dt.columns[i] for i in l]]
#dt= dt.rename(columns={"class":"label"})


#data= df.values


# In[76]:


#df.head()


# In[77]:


#df.info()
#len("mushrooms_test_updated.csv")


# In[78]:



def puritychk(data):                     #to check whether the tree has reached to pure nodes (stopping criteria)
    unique_cls = np.unique(data[:,-1])
    if len(unique_cls) == 1:
        return True
    else:
        return False
    
#puritychk(df.values)


# In[79]:


def classify_data(data):

    unique_cls, counts_unique = np.unique(data[:,-1],return_counts=True)
    index = counts_unique.argmax()
    classification = unique_cls[index]
    
    return classification


# In[80]:


def split(data):              # to get the possible splits in the data
    
    splits={}
    r, cols = data.shape
    for i in range(cols -1):
        splits[i]=[]
        values= data[:,i]
        unique_values = np.unique(values)
        
        for index in range(len(unique_values)):
            splits[i].append(unique_values[index])
  
    return splits


# In[81]:


#mushroom_training.values
#splited_data=split(data)
#splited_data


# In[82]:


def split_data(data,split_column,split_value):
    
    split_column_values = data[:,split_column]
    
    data_below = data[split_column_values == split_value]
    data_above = data[split_column_values != split_value]
    
    return data_below, data_above
            


# In[83]:


#data_below, data_above = split_data(data, 1, 'f')
#data_above


# In[84]:


def entropy_calc(data):
    label_column= data[:,-1]
    r,counts = np.unique(label_column,return_counts=True)
    total = counts.sum()
    p = counts/total
    entropy = sum(-p * np.log2(p))
    return entropy


# In[85]:


#e = entropy_calc(data_below)
#e


# In[86]:


def entropy_overall(data_below, data_above):
    total_pts = len(data_below) + len(data_above)
    
    p_data_below = len(data_below)/total_pts
    p_data_above = len(data_above)/total_pts

    overall_entropy = (p_data_below * entropy_calc(data_below) + p_data_above * entropy_calc(data_above))
    return overall_entropy


# In[87]:


#entropy_overall(data_below, data_above)


# In[88]:


#splited_data


# In[89]:


def best_split(data, possible_splits):
    overall_entropy=999
    
    for i in possible_splits:
        for values in possible_splits[i]:
            data_below , data_above = split_data(data, split_column=i, split_value=values)
            current_overall_entropy=entropy_overall(data_below,data_above)

            if current_overall_entropy < overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column= i
                best_split_value= values
    return best_split_column, best_split_value


# In[90]:


#best_split(data,split)


# In[91]:


def decision_tree(df, counter=0):      #generating the tree
    
    if counter == 0:
        global header
        header = df.columns 
        data = df.values         #if its the first instance of running the program the data frame(df) is converted to a numpy 2d array of data
    else:
        data = df
        
    if puritychk(data):           #checking for purity (i.e) the the tree has reached to its bottom most node
        classification = classify_data(data)
        return classification
    
    else:                         # recursive calling to form the tree
        counter+=1
        
        possible_splits = split(data)
        split_column , split_value = best_split(data, possible_splits)
        data_below, data_above = split_data(data,split_column,split_value)
        
        #initialising the sub tree
        question="{} <= {}". format(header[split_column],split_value)
        subtree = {question:[]}
        
        #recursion
        left_side  = decision_tree(data_below, counter)
        right_side = decision_tree(data_above, counter)
        
        if left_side == right_side:
            subtree = left_side
        else:
            subtree[question].append(left_side)
            subtree[question].append(right_side)
        
        #print(subtree)
        
        return subtree
    
    


# In[92]:


#mushroom_training= pd.read_csv("mushrooms_train_updated.csv")
#l=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,0] # index order
#mushroom_training= mushroom_training[[mushroom_training.columns[i] for i in l]]
#mushroom_training= mushroom_training.rename(columns={"class":"label"})

#tree= decision_tree(mushroom_training)
#pprint(tree)


# In[93]:


#header = mushroom_training.columns
#header


# In[94]:


#mushroom_testing = pd.read_csv("mushrooms_test_updated.csv")
#l=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,0] # index order
#mushroom_testing= mushroom_testing[[mushroom_testing.columns[i] for i in l]]
#mushroom_testing= mushroom_testing.rename(columns={"class":"label"})

#example = mushroom_testing.iloc[0]
#question=list(tree.keys())[0]
#question.split()


# In[95]:


def classify(dt,tree):            
    question=list(tree.keys())[0]
    feature_name,_,value = question.split()

    if dt[feature_name] == value:
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    if not isinstance(answer,dict):
        return answer
    else:
        return classify(dt, answer)


# In[96]:


#    classify(example,tree)


# In[100]:


def accuracy(dt, tree):             #finding the accuracy of the tree
    
    dt["classification"] = dt.apply(classify , axis=1, args=(tree,))
    dt["classification_correct"] = dt.classification == dt.label
    
    accuracy = dt.classification_correct.mean()
    
    return accuracy , dt


# In[121]:


def conf_mat(dt):          #confusion matrix generation
    c1=0
    c2=0
    c3=0
    c4=0
    label = dt.label
    classification = dt.classification
    for i in range(len(dt)):
        if classification[i] == label[i]:
            if label[i] == 'p':
                c4+=1
            else:
                c1+=1
        else:
            if classification[i] == 'p':
                c2+=1
            else:
                c3+=1
    array=[[c1,c2], [c3,c4]]
    return array


# In[111]:



label[0]


# In[122]:


mushroom_testing = pd.read_csv("mushrooms_test_updated.csv")
l=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,0] # index order
mushroom_testing= mushroom_testing[[mushroom_testing.columns[i] for i in l]]
mushroom_testing= mushroom_testing.rename(columns={"class":"label"})

mushroom_training = pd.read_csv("mushrooms_test_updated.csv")
l=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,0] # index order
mushroom_training = mushroom_training[[mushroom_training.columns[i] for i in l]]
mushroom_training = mushroom_training.rename(columns={"class":"label"})

tree = decision_tree(mushroom_training)
accuracy1, updated_test_data = accuracy(mushroom_testing, tree)
confusion_matrix = conf_mat(updated_test_data)

pprint(tree)
print(accuracy1)
print(confusion_matrix)


# In[ ]:




