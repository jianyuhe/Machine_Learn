# -*- coding: utf-8 -*-

# This weeks labs builds on the previous lab. For conveniences, I have
# included the code from last weeks lab at the top of the file.
# This weeks lab starts further down the file.
# To begin, scroll down to the start of this weeks lab.


from random import random,randint
import math

def wineprice(rating,age):
  peak_age=rating-50
  
  # Calculate price based on rating
  price=rating/2
  if age>peak_age:
    # Past its peak, goes bad in 10 years
    price=price*(5-(age-peak_age)/2)
  else:
    # Increases to 5x original value as it
    # approaches its peak
    price=price*(5*((age+1)/peak_age))
  if price<0: price=0
  return price

def wineset1():
  rows=[]
  for i in range(300):
    # Create a random age and rating
    rating=random()*50+50
    age=random()*50

    # Get reference price
    price=wineprice(rating,age)
    
    # Add some noise
    price*=(random()*0.2+0.9)

    # Add to the dataset
    rows.append({'input':(rating,age),
                 'result':price})
  return rows


def euclidean(v1,v2):
  d=0.0
  for i in range(len(v1)):
    d+=(v1[i]-v2[i])**2
  return math.sqrt(d)


def getdistances(data,vec1):
  distancelist=[]
  
  # Loop over every item in the dataset
  for i in range(len(data)):
    vec2=data[i]['input']
    
    # Add the distance and the index
    distancelist.append((euclidean(vec1,vec2),i))
  
  # Sort by distance
  distancelist.sort()
  return distancelist

def knnestimate(data,vec1,k=5):
  # Get sorted distances
  dlist=getdistances(data,vec1)
  avg=0.0
  
  # Take the average of the top k results
  for i in range(k):
    idx=dlist[i][1]
    avg+=data[idx]['result']
  avg=avg/k
  return avg

def inverseweight(dist,num=1.0,const=0.1):
  return num/(dist+const)


def subtractweight(dist,const=1.0):
  if dist>const: 
    return 0
  else: 
    return const-dist

def gaussian(dist,sigma=5.0):
  return math.e**(-dist**2/(2*sigma**2))


def weightedknn(data,vec1,k=5,weightf=gaussian):
  # Get distances
  dlist=getdistances(data,vec1)
  avg=0.0
  totalweight=0.0
  
  # Get weighted average
  for i in range(k):
    dist=dlist[i][0]
    idx=dlist[i][1]
    weight=weightf(dist)
    avg+=weight*data[idx]['result']
    totalweight+=weight
  if totalweight==0: return 0
  avg=avg/totalweight
  return avg


#
#--------------------------------------------------------------------------------
#
# In our last lab we looked at:
# - Nearest Neighbour classifiers
# - - k-NN: 
# - - setting K
# - - similarity measures: euclidean distance, ...
# - Weighted Neighbours
# - - weight functions: inverse, subtraction, gaussian
# - - weighted k-NN
#
# We will begin today's lecture by looking at how to handle the data.
#
#--------------------------------------------------------------------------------
#
# Cross Validation
#
# We have been given some data related to a problem and we are going to create a 
# classifier for the problem. It is important that we are able to judge how well 
# our classifier will generalise before we deploy it.
#
# A simple way to to this would be to:
# (1) Randomly choose a small subset of a data and set that aside as a test set
# (2) Take the remaing parts of out data and use it as a training set
# (3) Create our classifer using the training set
# (4) Estimate how well our classifier will generalise by testing it on the test set.
#
# This approach is called: 'test-set estimator of performance'
#
# This approach seems sensible and straight forward, so whats the problem:
# (1) Wastes data: the test set is never used during training
# (2) If we don't have much data, our test-set might just be lucky or unlucky;
# in other words, test-set estimator of performance has a high variance.
#
# Cross-validation is the name given to a set of techniques that repeatdely divides 
# up data into training sets and test sets.
# 
# During cross-validation we repeatedly do the follow sequence of steps:
#
# 1 Divide the data into a training set and test set.
# 1 The training set is given to the algorithm, along with the correct answers 
# (in this case, prices), and becomes the set used to make predictions.
# 2 The algorithm is then asked to make predictions for each item in the test set. 
# 3 The answers it gives are compared to the correct answers, and an overall score for how
# well the algorithm did is calculated.
#
# Usually this procedure is performed several times, dividing the data up differently
# each time. 
#
# Typically, the test set will be a small portion, perhaps 5 percent of the all
# the data, with the remaining 95 percent making up the training set. 
#
# We can then report the average accuracy.
#
# In order to cross-validate our k-nn classifier we need to create a function that
# will divide up the dataset into two smaller sets given a ratio that we specify:

def dividedata(data,test=0.05):
  trainset=[]
  testset=[]
  for row in data:
    if random()<test:
      testset.append(row)
    else:
      trainset.append(row)
  return trainset,testset
  
# The next step is to test the algorithm by giving it a training set and calling it with
# each item in the test set. The function calculates the differences and combines them
# to create an aggregate score for how far off it was in general. This is usually done by
# adding up the squares of all the differences.  
#
#
# testalgorithm takes an algorithm, algf, which accepts a dataset and query. It loops
# over every row in the test set and then calculates the best guess by applying algf. It
# then subtracts the guess from the real result.

def testalgorithm(algf,trainset,testset):
  error=0.0
  for row in testset:
    guess=algf(trainset,row['input'])
    error+=(row['result']-guess)**2
    #print row['result'],guess
  #print error/len(testset)
  return error/len(testset)

# The final step is to create a function that makes several different divisions of data and
# runs testalgorithm on each, adding up all the results to get a final score.

def crossvalidate(algf,data,trials=100,test=0.1):
  error=0.0
  for i in range(trials):
    trainset,testset=dividedata(data,test)
    error+=testalgorithm(algf,trainset,testset)
  return error/trials

# Lets try to generate a cross validated accuracy score for our k-nn
# classifier using different values of k
#
##>>> reload(knn2lab)
##<module 'knn2lab' from 'knn2lab.py'>
##>>> data=knn2lab.wineset1( )

##>>> knn2lab.crossvalidate(knn2lab.knnestimate,data)
##254.06864176819553
##>>> def knn3(d,v): return knn2lab.knnestimate(d,v,k=3)
##    (press return)
##>>> knn2lab.crossvalidate(knn3,data)
##166.97339783733005
##>>> def knn1(d,v): return knn2lab.knnestimate(d,v,k=1)
##    (press return)
##>>> knn2lab.crossvalidate(knn1,data)
##209.54500183486215
#
# Looking at the results I got (your results may be different because the data was generated using
# some randomness) we see that: using too few neighbors or too many neighbors 
# leads to poor results. 
# In this example, a value of 3 performs better than a value of 1 or 5. 
#
# How does our weighted knn classifer do when its cross validated:
#
##>>> knn2lab.crossvalidate(knn2lab.weightedknn,data)
##200.34187674254176
##>>> def knninverse(d,v):
##        return knn2lab.weightedknn(d,v,weightf=knn2lab.inverseweight)
##    (press return)
##>>> knn2lab.crossvalidate(knninverse,data)
##148.8594770266061
#
# When the parameters are set properly, weighted kNN seems to give better results for
# this dataset. 
#
#--------------------------------------------------------------------------------
#
# Heterogeneous Variables
# 
# The dataset we have used so far is artificially simple—specifically, 
# (1) all the variables used to predict the price are roughly comparable 
# (2) and are all important to the final result.
#
# Since all the variables fall within the same range, it’s meaningful to calculate distances
# using all of them at once. However, if we add a new variable that influenced the price to
# the dataset which had a bigger range than age or rating, this new variable would have a far greater
# impact on the calculated distances than the original ones do—it will overwhelm any distance
# calculation, which essentially means that the other variables are not taken into account.
#
# We can simulate this issue in our wineset domain using the size of the bottle in milliliters. 
# Unlike age and raing, which range between 0 and 100, this variable ranges up to 1,500. 
#
# A different problem is the introduction of entirely irrelevant variables. If the dataset
# also included the number of the aisle in which you found the wine, this variable
# would be included in the distance calculations. Two items identical in every respect
# but with very different aisles would be considered very far apart, which would badly
# hinder the ability of the algorithms to make accurate predictions.
#
# Let update our dataset to simulate these issues:
# The wineset2() function is the same as wineset1() except for the inclusion of the # lines

def wineset2():
  rows=[]
  for i in range(300):
    rating=random()*50+50
    age=random()*50
    aisle=float(randint(1,20)) #
    bottlesize=[375.0,750.0,1500.0][randint(0,2)] #
    price=wineprice(rating,age)
    price*=(bottlesize/750) #
    price*=(random()*0.2+0.9)
    rows.append({'input':(rating,age,aisle,bottlesize),
                 'result':price})
  return rows

# We can now create new datasets with aisles and bottle sizes:
#
##>>> reload(knn2lab)
##<module 'knn2lab' from 'knn2lab.py'>
##>>> data=knn2lab.wineset2( )
#
# To see how this affects the kNN predictors, try them out on the new datasets with
# the best parameters you managed to find earlier:
##>>> knn2lab.crossvalidate(knn3,data)
##1427.3377833596137
##>>> knn2lab.crossvalidate(knn2lab.weightedknn,data)
##1195.0421231227463
#
# Even though the dataset now contains even more information 
# than it did before (which should theoretically lead to better predictions),
# the values returned by crossvalidate have actually gotten a lot worse. 
#
# This is because the algorithms do not yet know how to treat the variables differently.
#
#--------------------------------------------------------------------------------
#
# Scaling Dimensions
#
# If we have variables with different ranges we need to normalize their values
# so that it makes sense to consider them all in the same space.
#
# It would also be helpful to find a way to eliminate the superfluous variables
# or to at least reduce their impact on the calculations.
#
# One way to accomplish both things is to rescale the dimensions before
# performing any of the calculations. The simplest form of rescaling is multiplying
# the values in each dimension by a constant for that dimension.
#
# The rescale function takes a list of items and a parameter called scale,
# which is a list of real numbers. It returns a new dataset with all the
# values multiplied by the values in scale.

def rescale(data,scale):
  scaleddata=[]
  for row in data:
    scaled=[scale[i]*row['input'][i] for i in range(len(scale))]
    scaleddata.append({'input':scaled,'result':row['result']})
  return scaleddata

#
# Using the rescale function we can address both of our problems by:
# (1) scaling down the bottle size variable by a factor of 0.5
# (2) multiplying all the irrelevant variables (e.g., aisle) by 0
#
##>>> reload(knn2lab)
##<module 'knn2lab' from 'knn2lab.py'>
##>>> sdata=knn2lab.rescale(data,[10,10,0,0.5])
##>>> knn2lab.crossvalidate(knn3,sdata)
##660.9964024835578
##>>> knn2lab.crossvalidate(knn2lab.weightedknn,sdata)
##852.32254222973802
#
# The results are pretty good for those few examples; certainly better than before
#
# In this case, it’s not difficult to choose good parameters for scaling because
# you know in advance which variables are important. However, most of the time
# you’ll be working with datasets that you didn’t build yourself, and you won’t
# necessarily know which variables are unimportant and which ones have a
# significant impact.
#
# In theory, you could try out a lot of numbers in different combinations until
# you found one that worked well enough, but there might be hundreds of
# variables to consider and it would be very tedious.
#
# Typically you have to automatically find a good solution when there are many
# input variables to consider—by using optimization techniques. This basically
# involve automatically searching though a combination of ranges of scaling
# factos and cross-validating the results for each combination. You then use
# the best scaling parameters in your classifier. (we will post-pone a more
# detailed description of optimization for another day).
#
#-----------------------------------------------------------------------------
#
# Uneven Distributions
#
# So far we’ve been assuming that if you take an average or weighted average of the
# data, you’ll get a pretty good estimate of the final price. In many cases this will
# be accurate, but in some situations there may be an unmeasured variable that can have
# a big effect on the outcome.
#
# Imagine that in the wine example there were buyers from two separate groups:
# (1) people who bought from the liquor store,
# (2) and people who bought from a discount store and received a 50 percent discount.
#
# Unfortunately, this information isn’t tracked in the dataset.
#
# What will happen if you ask for an estimate of the price of a different item
# using the kNN or weighted kNN algorithms?
# - It will bring in the nearest neighbors regardless of where the purchase was made.
# The result is that it will give the average of items from both groups, perhaps
# representing a 25 percent discount. The problem here is that the estimate does not
# accurately reflect what someone will actually end up paying for an item.
# In order to get beyond averages, you need a way to look closer at the data at that point.
#
# To examine the impact of uneven distributions on classification and how to address
# it we will extend our dataset creation function to create uneven distributed data.
# The wineset3 function creates a dataset that simulates these properties.
# It drops some of the complicating variables and just focuses on the original ones. 

def wineset3():
  rows=wineset1()
  for row in rows:
    if random()<0.5:
      # Wine was bought at a discount store
      row['result']*=0.6
  return rows

#
#----------------------------------------------------------------------------------
#
# Estimating the Probability Density
#
# Rather than taking the weighted average of the neighbors and getting a single price
# estimate, it might be interesting in this case to know the probability that an item falls
# within a certain price range.
#
# In the example, given inputs of 99 percent and 20 years, you’d like a function
# that tells you there’s a 50 percent chance that the price is
# between $40 and $80, and a 50 percent chance that it’s between $80 and $100.
#
# To do this, you need a function that returns a value between 0 and 1 representing the
# probability of a particular bottle of wine being within a specified range.
# - The function first calculates the weights of the neighbors within the range
# - It then calculates the weights of all the neighbors.
# - The probability is the sum of the neighbor weights within the range divided by
# the sum of all the weights. 

def probguess(data,vec1,low,high,k=5,weightf=gaussian):
  dlist=getdistances(data,vec1)
  nweight=0.0
  tweight=0.0
  
  for i in range(k):
    dist=dlist[i][0]
    idx=dlist[i][1]
    weight=weightf(dist)
    v=data[idx]['result']
    
    # Is this point in the range?
    if v>=low and v<=high:
      nweight+=weight
    tweight+=weight
  if tweight==0: return 0
  
  # The probability is the weights in the range
  # divided by all the weights
  return nweight/tweight

# Like kNN, this function sorts the data by the distance from vec1 and determines the
# weights of the nearest neighbors.
# It adds the weights of all the neighbors together to get tweight.
# It also considers whether each neighbor’s price is within the range (between low and high);
# if so, it adds the weight to nweight.
# The probability that the price for vec1 is between low and high is
# nweight divided by tweight.
#
# Lets try it out on our dataset:
#
##>>> import knn2lab
##>>> data=knn2lab.wineset3()
##>>> knn2lab.wineprice(99.0,20.0)
##106.07142857142857
##>>> knn2lab.probguess(data,[99.0,20.0],10,40)
##0.0
##>>> knn2lab.probguess(data,[99.0,20.0],40,80)
##0.5556375402316277
##>>> knn2lab.probguess(data,[99.0,20.0],80,120)
##0.44436245976837219
##>>> knn2lab.probguess(data,[99.0,20.0],120,160)
##0.0
##>>> knn2lab.probguess(data,[99.0,20.0],40,120)
##1.0
#
# The function gives good results. The ranges that are well outside the actual prices
# have probabilities of 0, and the ones that capture the full range of possibilities are
# close to 1.
#
# By breaking it down into smaller buckets, you can determine the actual
# ranges in which things tend to cluster. However, this requires that you guess at the
# ranges and enter them until you have a clear picture of the structure of the data.
#
