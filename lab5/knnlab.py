# -*- coding: utf-8 -*-
#
# --------------------------------------------------------------------
# How to use these notes:
#
# The notes for todays lab are in the form of code with
# comments and examples interspersed. You should read down
# through the code and examples and try out the examples in
# the Python window as you work down through the notes.
#
# --------------------------------------------------------------------
# Lab Overview:     
#
# The lab will focus on applying kNN classificaiton to the problem of
# prices prediction.  
# When predicting prices it’s more interesting to look at a dataset
# where price doesn’t simply increase in proportion to size or the
# number of characteristics. One domain where this is the case is
# predicting the prices of wine.
#
# TOC:
#    Generating a data set
#    kNN Price Modelling Overview
#    Defining Similarity
#    k-NN Code
#    Weighted Neighbours
#    Weighted KNN
#    Solutions
#
# --------------------------------------------------------------------
# Generating a data set:
#
# The first thing we need to do is to get a data set.
# Normally you will gather a dataset together from different sources.
# However, for today's lab we will create a function that artificially
# simulates a data set.
#
# The wineprice/2 function below computes a price for a wine:
# (1) based on its age and rating.
# (2) using the assumption that wine has a peak age, which is:
#     - older for good wines
#     - and almost immediate for bad wines.
#

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

#           ***********************************
#                       Your Turn!
#           ***********************************
#
# Lets start a Python session and test some wine prices.
# I suggest that you start Python from the same directory as
# you have saved these notes. This will make it easier to import
# the code you are reading into your Python session.
#
# On the next few lines which start with ## I have listed the
# inputs and outputs that occurred in my Python session when
# I tested the wineprice/2 function.
# Note that the function is inside the module knnlab so
# to invoke the function I used knnlab.wineprice/2
#
## $ python
## >>> import knnlab
## >>> knnlab.wineprice(95.0,3.0)
## 21.111111111111114
## >>> knnlab.wineprice(95.0,8.0)
## 47.5
## >>> knnlab.wineprice(99.0,1.0)
## 10.102040816326529
#
#           ***********************************
#
# We can use the wineprice/2 function to build of data set of wines.
#
# The wineset1/0 function generates 300 random bottles of wine and
# calculates their prices using the wineprice/2 function above.
#
# It then randomly adds or subtracts 20 percent to capture things like
# taxes and local variations in prices, and also to make the numerical
# prediction a bit more difficult.
#

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

#           ***********************************
#                       Your Turn!
#           ***********************************
#
# Lets build use wineset1/0 to build a new dataset.
#
# Remember that we use a random function in the dataset generation
# so the data set you generate will be different from the one I
# used when creating the examples. So you will see slightly different
# numbers in the examples.
#
## >>> data=knnlab.wineset1( )
## >>> data[0]
## {'input': (63.602840187200407, 21.574120872184949), 'result': 34.565257353086487}
## >>> data[1]
## {'input': (74.994980945756794, 48.052051269308649), 'result': 0.0}
#
#           ***********************************
#
# --------------------------------------------------------------------
# k-NN Prices Modelling
#
# Approach find a few of the most similar items and assume the prices
# will be roughly the same.
#
# By finding a set of items similar to the item that interests you, the
# algorithm can average their prices and make a guess at what the price
# should be for this item.
#
# If the data were perfect, you could use k=1, meaning you would just
# pick the nearest neighbor and use its price as the answer.
#
# But there are always aberrations so it is best to average over a few
# neighbours to reduce any noise.
#
# Selecting k is difficult: to low and the prediction is overly
# sensitive to noise, to high and the prediction is effected by
# examples that are not similar.
#
# --------------------------------------------------------------------
# Defining Similarity
#
# There are many ways to define similarity.
#
# Euclidean distance is one of the most popular
# 

def euclidean(v1,v2):
  d=0.0
  for i in range(len(v1)):
    d+=(v1[i]-v2[i])**2
  return math.sqrt(d)

# Note: this function treats both age and rating the same when calculating
# distance, even though in almost any real problem, some of variables have
# a greater impact on the final price than others. This is a well-known
# weakness of kNN.
#
# *********** Python Note ********************************************
# If you do need to iterate over a sequence of numbers, the built-in
# function range() comes in handy. It generates lists containing
# arithmetic progressions, e.g.:
##    >>> range(10)
##    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# The given end point is never part of the generated list;
# *********** End Python Note ****************************************
#
#           ***********************************
#                       Your Turn!
#           ***********************************
#
# In your Python session, try the function on some of the points in
# your dataset, along with a new data point:
#
## >>> data[0]['input']
## (82.720398223643514, 49.21295829683897)
## >>> data[1]['input']
## (98.942698715228076, 25.702723509372749)
## >>> knnlab.euclidean(data[0]['input'],data[1]['input'])
## 28.56386131112269
#
# During the lecture we talk about other direct similarity metrics such
# as the Manhattan distance and Cosine.
#
# In the next to lab tasks you will implement functions to compute these
# similarity metrics. Note, you should use methods from the Python math
# module to define these functions. You can look up the Python math module
# online. To include it in your Python session simply type:
#
## >>> import math
#
# Define a method to that implements the Manhattan Distance. Example
# interactions with this function are:
#
## >>> manhattan([1,1],[1,0])
## 1.0
## >>> manhattan([0,1],[1,0])
## 2.0
#
# Define a method to that implements the cosinse similarity measure. Example
# interactions with this function are:
#
## >>> cosval = cosSim([1,1],[2,2])
## >>> print cosval
## 1.0
## >>> cosval = cosSim([1,0],[0,1])
## >>> print cosval
## 0.0
## >>> cosval = cosSim([-1,0],[1,0])
## >>> print cosval
## -1.0
## >>> cosval = cosSim([0,-1],[1,0])
## >>> print cosval
## 0.0
## >>>
#
# If you want to check your implemention of the cosinse similarity using other
# vectors pairs, there is a cosine similarity calculator online at:
#
# http://www.appliedsoftwaredesign.com/cosineSimilarityCalculator.php
#
# A solution implementation for the manhattan distance and cosine similarity
# metrics are listed at the bottom of these class notes. You should try to
# implement your own solutions first before you look at the provided solutions.
#
#           ***********************************
#
#
# --------------------------------------------------------------------
# k-NN Code
#
# Implementing the k-NN algorithm is simple. We need two functions:
# (1) one getdistances/2 function computes the distances between a
# query and the examples in the data set
# (2) the knnestimate/3 function averages over the nearest neighbours
#
# The getdistances/2 function calls the distance function on the vector
# given against every other vector in the dataset and puts them in a
# big list. The list is sorted so that the closest item is at the top.

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

#
# The kNN function uses the list of distances and
# averages the top k results
#

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

#
#           ***********************************
#                       Your Turn!
#           ***********************************
#
# You can now get a price estimate for a new item:
#
## >>> knnlab.knnestimate(data,(95.0,3.0))
## 29.176138546872018
## >>> knnlab.knnestimate(data,(99.0,3.0))
## 22.356856188108672
## >>> knnlab.knnestimate(data,(99.0,5.0))
## 37.610888778473793
## >>> knnlab.wineprice(99.0,5.0) # Get the actual price
## 30.306122448979593
## >>> knnlab.knnestimate(data,(99.0,5.0),k=1) # Try with fewer neighbors
## 38.078819347238685
#
# Try different parameters and different values for k to see how the
# results are affected
#
#           ***********************************
#
# --------------------------------------------------------------------
# Weighted Neighbours
# 
# One way to compensate for the fact that the algorithm may be using
# neighbors that are too far away is to weight them according to their
# distance. 
#
# So we need a way of converting distances to weights. 
#
# * Inverse Function 
# Simplest form returns 1 divided by the distance.
# However, if items are exactly the same will result in infinite weight.
# Avoid this by adding a small number to the distance before inverting it.

def inverseweight(dist,num=1.0,const=0.1):
  return num/(dist+const)

# Pros: simple to implement and fast
# Cons: applies very heavy weights to items close by and falls of quickly
# after that. This can make the algorithm sensitive to noise.
#
# * Subtract Function 
# Subtract the distance from a constant. The weight is:
# (1) the result of this s ubtraction if the result is greater than zero;
# (2) otherwise, the result is zero.

def subtractweight(dist,const=1.0):
  if dist>const: 
    return 0
  else: 
    return const-dist

# Pros: simple to implement and fast
# Pros: overcomes the issue of overweighting close items
# Cons: Because the weight eventually falls to 0, it’s possible that
# there will be nothing close enough to be considered a close neighbor,
# which means that for some items the algorithm won’t make a prediction
# at all.
#
# * Gaussian Function
# The weight in this function is 1 when the distance is 0,
# and the weight declines as the distance increases.

def gaussian(dist,sigma=5.0):
  return math.e**(-dist**2/(2*sigma**2))

# Pros: overcomes the issue of overweighting close items
# Pros: overcomes the issue of weight going to zero resulting in no prediction
# Cons: The code for this function is more complex and will not
# evaluate as quickly as the other two functions.
#
#
#           ***********************************
#                       Your Turn!
#           ***********************************
#
# Try out these distance weighting methods on a range of values to
# get a feel for how they work.
#
## >>> knnlab.inverseweight(0.1)
## 5.0
## >>> knnlab.subtractweight(0.1)
## 0.9
## >>> knnlab.gaussian(0.1)
## 0.9998000199986667
## >>> knnlab.inverseweight(1.0)
## 0.9090909090909091
## >>> knnlab.subtractweight(1.0)
## 0.0
## >>> knnlab.gaussian(1.0)
## 0.9801986733067553
## >>> 
#
#           ***********************************
#
# *********** Python Note ********************************************
# Python allows you to pass methods as argument to other methods.
# For example
#
## >>> def method1():
## >>>     return 'hello world'
## >>> 
## >>> def method2(methodToRun):
## >>>     result = methodToRun()
## >>>     return result
## >>> 
## >>> method2(method1)
# *********** End Python Note ****************************************
#
#           ***********************************
#                       Your Turn!
#           ***********************************
#
# Define a function that takes a weight method as an argument and applies
# it to an array of number. For example, assuming the function you defined
# is called applyWeightMethod you should be able to use it as follows:
#
## >>> distances = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
## >>> applyWeigthMethod(knnlab.inverseweight,distances)
## [5.0, 3.333333333333333, 2.5, 2.0, 1.6666666666666667, 1.4285714285714286, 1.25, 1.1111111111111112, 1.0, 0.9090909090909091]
## >>> applyWeigthMethod(knnlab.subtractweight,distances)
## [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.30000000000000004, 0.19999999999999996, 0.09999999999999998, 0.0]
## >>> applyWeigthMethod(knnlab.gaussian,distances)
## [0.9998000199986667, 0.9992003199146837, 0.9982016190284373, 0.9968051145430329, 0.9950124791926823, 0.9928258579038134, 0.9902478635182347, 0.9872815715902905, 0.9839305142725083, 0.9801986733067553]
## >>> 
#
# I have listed a solution to this at the end of the labs notes. Try to solve this
# youself before you look at the solution.
#
#           ***********************************
#
# --------------------------------------------------------------------
# Weighted kNN
#
# The code for doing weighted kNN works the same way as the regular
# kNN function by:
# (1) getting the sorted distances
# (2) and taking the k closest elements.
#
# The important difference is that instead of just averaging them,
# the weighted kNN calculates a weighted average.
#
# The weighted average is calculated by:
# (1) multiplying each item’s weight by its value
# (2) summing the results of these multiplicationsbefore adding them together.
# (3) dividing the sum from (2) by the sum of all the weights.
#

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

# The function loops over the k nearest neighbors and passes each of
# their distances to one of the weight functions you defined earlier.
#
# The avg variable is calculated by multiplying these weights by the
# neighbor’s value.
#
# The totalweight variable is the sum of the weights.
#
# At the end, avg is divided by totalweight.
#
#           ***********************************
#                       Your Turn!
#           ***********************************
#
# Lets try this function in our python session and compare its
# performance to that of the regular kNN function:
#
## >>> knnlab.weightedknn(data,(99.0,5.0))
## 32.640981119354301
#
#           ***********************************
#
# --------------------------------------------------------------------
# Solutions:
#
## >>> def manhattan(v1,v2):
## 	d=0.0
## 	for i in range(len(v1)):
## 		d+=math.fabs(v1[i]-v2[i])
## 	return d
#
## >>> def cosSim(v1,v2):
## 	dot=0.0
## 	for i in range(len(v1)):
## 		dot+=(v1[i]*v2[i])
## 	len1=0.0
## 	len2=0.0
## 	for i in range(len(v1)):
## 		len1+=(v1[i]*v1[i])
## 		len2+=(v2[i]*v2[i])
## 	cos = dot/(math.sqrt(len1)*math.sqrt(len2))
## 	return cos
#
## >>> def applyWeigthMethod(weightMethod,inputs):
## 	  weights=[]
## 	  for i in range(len(inputs)):
## 	      weights.append(weightMethod(inputs[i]))
## 	  return weights
#
#         




