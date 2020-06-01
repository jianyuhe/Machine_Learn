
# coding: utf-8

import math

featureNames = ['stream', 'slope', 'elevation', 'vegetation']

featureLevels = {'stream':['false','true'],
                  'slope':['flat','moderate','steep'], 
                  'elevation':['low','medium','high','highest'], 
                  'vegetation':['chaparral','riparian','conifer']}

dataset1=[['false','steep','high','chaparral'],
          ['true','moderate','low','riparian'],
          ['true','steep','medium','riparian'],
          ['false','steep','medium','chaparral'],
          ['false','flat','high','conifer'],
          ['true','steep','highest','conifer'],
          ['true','steep','high','chaparral']]



def entropy(values):
   from math import log
   val_counts={}
   for v in values:
        if v in val_counts.keys():
            count = val_counts[v]
            val_counts[v]=count+1
        else:
            val_counts[v]=1
   entropy=0.0
   for v in val_counts.keys():
      p=float(val_counts[v])/len(values)
      entropy=entropy+(p*log(p,2))
   if entropy!=0:
        entropy=entropy*-1
   return (entropy)


#The following function extracts a column of values from a table and returns it as a list
def getFeatureColumn(featureindex=-1, dataset=None):
    featureColumn = []
    for i in range(0,len(dataset)):
        featureColumn.append(dataset[i][featureindex])
    return(featureColumn)

#This function splits a dataset (arg2) up based on the
#levels of a particular feature in the dataset (arg1)
#The functon returns a dictionary object where
#the keys are the levels of a specified feature
#the values are lists containing the instances in a dataset
#where the feature takes the level specified by the dictionary key
def createPartitions(featureIndex=-1,dataset=None):
    partitions={}
    for i in range(0, len(dataset)):
        tmpValue = dataset[i][featureIndex]
        tmpList = []
        if tmpValue in partitions.keys():
            tmpList = partitions[tmpValue]
            tmpList.append(dataset[i])
        else:
            tmpList.append(dataset[i])
        partitions[tmpValue]=tmpList
    return(partitions)


#This function takes a dictionary object as an argument
#and sums the weighted entropy of the sets defined by the lists
#contained in the values of the dictionary
def calculateRemainder(partitions):
    remainder=0
    #we assume that the target feature is the rightmost column in an instance
    #so we can get the target index by retrieveing an one of the value from the
    #partitions datasture (next(iter(partitions.values()))) and subtracting 1
    #from the length of this instance.
    exampleInstance=(next(iter(partitions.values())))[0]
    targetIndex=len(exampleInstance)-1
    #in order to be able to weight the entropy of each partition
    #we need to know the total number of examples across all the partitions
    #this number defines the denominator in the weight term
    #we store this number in the variable size_dataset
    size_dataset =0
    for k in partitions.keys():
        size_dataset = size_dataset + len(partitions[k])
    #we are no ready to calculate the remaining entropy by calculating a
    #the weighted sum of the entropy for each partition
    for k in partitions.keys():
        #calculate the entropy for each partition
        tmpPartition = partitions[k]
        targetColumn = getFeatureColumn(targetIndex,tmpPartition)
        ent = entropy(targetColumn)
        #calculate the weight for each partition
        weight = len(tmpPartition)/size_dataset
        #sum the weighting remaining entropy for each partition
        remainder = remainder + (weight * ent)
    return(remainder)

def informationGain(featureIndex=-1, dataset=[]):
    #calculate the entropy of the dataset before we partition it using the feature 
    targetIndex = len(dataset[0])-1
    targetColumn = getFeatureColumn(targetIndex,dataset)
    H=entropy(targetColumn)
    #calculate the remaining entropy after we partition the dataset using the feature
    partitions=createPartitions(featureIndex,dataset)
    rem=calculateRemainder(partitions)
    #calculate the information gain for the feature
    ig=H-rem
    return(ig)

# ###The Decision Tree Representation
class tree_node:
  def __init__(self,featureName='',featureIndex='',branches={},instances=[],prediction=''):
    self.featureName=featureName #stores the name of the feature tested at this node
    self.featureIndex=featureIndex #stores the index of the feature column in the dataset
    self.branches=branches #a dictionary object: each key=level of test feature, each value=child node of this node
    self.instances=instances #in a leaf node this list stores the set of instances that ended up at the leaf
    self.prediction=prediction #in a leaf node this variable stores the target level returned as a prediction by the node
    
    


# ###Implementing the ID3 Algorithm
# To keep the implementation of the **ID3** algorithm readable we define two helper functions:
# 1. **allSame** this function takes a list of instances and returns tree if the list is non-empty and all the instances have the same target feature level.
# 2. **majorityTargetLevel** which will take a dataset as a parameter and returns the majority target level in this dataset.

#return true if all the instances in the dataset D have the same target level
#a handy way to check for this condition is by checking if the entropy of the 
#dataset with respect to the target feature ==0
def allSame(D=[]):
    if len(D) > 0:
        targetIndex = len(D[0])-1
        targetColumn = getFeatureColumn(targetIndex,D)
        if entropy(targetColumn) == 0:
            return True
    return False
    
#return the majority target level in the instances list
def majorityTargetLevel(D):
        #assume the target feature is the last feature in each instance
        targetIndex = len(D[0])-1
        #extract the set of target levels in the instances at this node
        targetColumn = getFeatureColumn(targetIndex,D)
        #create a dictionary object that records the count for each target level
        levelsCount = {}
        for l in targetColumn:
            if l in levelsCount.keys():
                count = levelsCount[l]
                levelsCount[l]=count+1
            else:
                levelsCount[l]=1
        #find the target level with the max count
        #for ease of implementation we break ties in max counts
        #by symply returning the first level we find with the max count
        maxCount=-999999
        majorityLevel=''
        for k in levelsCount.keys():
            if levelsCount[k] > maxCount:
                maxCount=levelsCount[k]
                majorityLevel=k
        return(majorityLevel)

#This **ID3** implementation takes 5 parameters. The first 2 (d and D) are described in the book, 
#the last 3 (parentD, featureLevels, and featureNames) are including to ease the translation from
#the book pseudocode to Python implementation. The 5 parameters are as follows:
#1. d = list of descriptive features not yet used on the path from the root to the current node 
#2. D = the set of training instances that have descended the path to this node
#3. parenttD = the set of training instances at the parent of this node
#4. featureLevels = a dictionary object that lists for each feature (key) the set of levels in the domain of the feature (value)
#5. featureNames = a list of the names of the features in the dataset
def id3(d=[], D=[], parentD=[], featureLevels={}, featureNames=[]):
    if allSame(D):
        return tree_node(featureName='',featureIndex=-1,branches={},instances=D,prediction=D[0][len(D[0])-1])
    elif len(d) == 0:
        return tree_node(featureName='',featureIndex=-1,branches={},instances=D,prediction=majorityTargetLevel(D))
    elif len(D) == 0:
        return tree_node(featureName='',featureIndex=-1,branches={},instances=D,prediction=majorityTargetLevel(parentD))
    else:
        dBest = ""
        bestIndex = -1
        maxIG = -9999
        for f in d:
            featureIndex = featureNames.index(f)
            tmpIG = informationGain(featureIndex,D)
            if tmpIG > maxIG:
                maxIG=tmpIG
                dBest=f
                bestIndex=featureIndex
        node = tree_node(featureName=dBest,featureIndex=bestIndex,branches={},instances=[],prediction='')
        #partition the dataset using the feature with the highest information gain
        partitions=createPartitions(bestIndex,D)
        #remove dBest from the list of features passed down to the children of this node
        dNew = [ f for f in d if not f.startswith(dBest) ]
        #iterate across all the levels of the feature 
        #and create a branch for each level
        #we use arg4 for this because it may be that one or more of the
        #levels of the feature do not appears in D
        for level in featureLevels[dBest]:
            if level in partitions.keys():
                DNew = partitions[level]
            else:
                #if there is a feature level that does not occur in D
                #then create a child node where the set of training instances
                #at the node is empty
                DNew = []
            node.branches[level]=id3(dNew,DNew,D,featureLevels,featureNames)
        return(node)
        
#This function prints out the tree in text format
def printtree(tree,indent='-'):
    if tree.prediction == '':
        indent=indent+"--"
        for level in tree.branches.keys():
            print(indent+tree.featureName + ':' + str(level))
            printtree(tree.branches[level],indent)
    else:
        s=''
        for c in indent:
            s=s+' '
        print(s+" prediction = " + tree.prediction)


#This function returns a prediction from a tree for a query instance
def makePrediction(query,tree,featureLevels):
    if tree.prediction != '':
        #if we have reached a leaf node return the prediction
        return tree.prediction
    else:
        #otherwise descend the tree.
        #1. get the level of the query instance for the node test feature
        level = query[tree.featureIndex]
        for l in featureLevels[tree.featureName]:
            if l.startswith(level):
                #2. find the branch that matchs this level and desencd the branch
                return makePrediction(query,tree.branches[level],featureLevels)
        print("No prediction!")

if __name__ == "__main__":
    # Create a decision tree.
    tree = id3(featureNames[:-1], dataset1, dataset1, featureLevels, featureNames)
    # See what the tree looks like
    printtree(tree,"")
    # Using the Tree to Make Predictions
    query1 = ['true','moderate','low','?']
    print("Query: " + str(query1) + " Prediction: " + makePrediction(query1, tree, featureLevels))
    query2 = ['true','moderate','medium','?']
    print("Query: " + str(query2) + " Prediction: " + makePrediction(query2, tree, featureLevels))
    query3 = ['true','moderate','highest','?']
    print("Query: " + str(query3) + " Prediction: " + makePrediction(query3, tree, featureLevels))
    query4 = ['true','moderate','high','?']
    print("Query: " + str(query4) + " Prediction: " + makePrediction(query4, tree, featureLevels))
    query5 = ['true','steep','high','?']
    print("Query: " + str(query5) + " Prediction: " + makePrediction(query5, tree, featureLevels))
    query6 = ['true','flat','high','?']
    print("Query: " + str(query6) + " Prediction: " + makePrediction(query6, tree, featureLevels))

