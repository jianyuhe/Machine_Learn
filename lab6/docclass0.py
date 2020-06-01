# Document Filtering using Naive Bayes
#--------------------------------------------------
# 
# Knowing how to classify documents based on their contents is a very
# practical application of machine learning. Perhaps the most useful
# and well-known application of document filtering is spam detection.
# 
# We will develop a probability based classifier that will identify
# spam. You should remember that the naieve bayes algorithms we will
# look at today is not specific to dealing with spam. Because it
# solves the more general problem of learning to recognize whether
# a document belongs in one category or another, it can be used
# for less unsavory purposes. 
#
#
# Documents, Words and Features
#--------------------------------------------------
# 
# The classifier that we will be building needs features to use for
# classifying different items.
# 
# A feature is anything that you can determine as being either present
# or absent in the item.
# 
# When considering documents for classification, the items are the
# documents and the features are the words in the document.
# 
# When using words as features, the assumption is that some words are
# more likely to appear in spam than in nonspam, which is the basic
# premise underlying most spam filters.
# 
# Features don’t have to be individual words, however; they can be
# word pairs or phrases or anything else that can be classified
# as absent or present in a particular document.
#
# They could even be whole documents or single letters. Can you think
# why using whole documents or single letters as features typically
# isn't a good idea?
#
# ------------------Python Interlude------------------------------------
# As part of the preprocessing of the text data we will
# use the Python re module. This module allows us to create
# regular expressions that you can use to split up text.
#
# The following illustrates how to use re to split up text
# into a list of words:
#
## >>> import re
## >>> splitter=re.compile('\\W*')
## >>> s1="The quick brown fox jumped over the lazy dog."
## >>> s2=splitter.split(s1)
## >>> s2
## ['The', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', '']
## >>> 
# 
# re.compile() creates a regular expression.
# The parameter \ introduces a special sequence
# The parameter \W is a special sequence that matches any non-alphanumeric character
# 
# See http://docs.python.org/library/re.html for doc on re
# 
# ----------------------------------------------------------------------


import re #regular expression library
import math


# The first thing we need is a function that will extract the features (words)
# from a text. The getwords features will do this for us.
# This function breaks up the text into words by dividing the text on
# any character that isn’t a letter. This leaves only actual words,
# all converted to lowercase.
def getwords(doc):
  splitter=re.compile('\\W*') 
  print doc
  # Split the words by non-alpha characters
  # Exclude words with a length 2 character or less or
  # greater than 20 (words longer the 20 letters are
  # likely to be either errors in the splitting or
  # so rare as to be useless for classifying
  words=[s.lower() for s in splitter.split(doc) 
          if len(s)>2 and len(s)<20]
  
  # Return the unique set of words only
  return dict([(w,1) for w in words])

# 
# Determining which features to use is both very tricky and very important.
# - The features must be common enough that they appear frequently,
# - but not so common that they appear in every single document.
# 
# The other thing to consider when deciding on features is how well they
# will divide the set of documents into the target categories.
# 
# For example, the code for getwords above reduces the total number of
# features by converting them to lowercase. This means it will recognize
# that a capitalized word at the start of a sentence is the same
# as when that word is all lowercase in the middle of a sentence—a good
# thing, since words with different capitalization usually have the same
# meaning. However, this function will completely miss the SHOUTING style
# used in many spam messages, which may be vital for dividing the set into
# spam and nonspam. 
# 
# As you can see, the choice of feature set involves many tradeoffs and
# is subject to endless tweaking. For now, we can use the simple
# getwords function given.
# 
# Lets try out getwords in our python session:
# 
# >>> import docclass
# >>> docclass.getwords('this is a test sentence')
# this is a test sentence
# {'this': 1, 'test': 1, 'sentence': 1}
# >>> docclass.getwords('this is this')
# this is this
# {'this': 1}
# >>> 
#
#
# Training the Classifier
#--------------------------------------------------
# 
# We need to make a class to represent the classifier. 

class classifier:
  def __init__(self,getfeatures,filename=None):
    # Counts of feature/category combinations
    self.fc={}
    # Counts of documents in each category
    self.cc={}
    self.getfeatures=getfeatures

# The fc variable will store the counts for different features in
# different classifications.
# - For example:
#   {'python': {'bad': 0, 'good': 6}, 'the': {'bad': 3, 'good': 3}}
# This indicates that the word “the” has appeared in documents
# classified as bad three times, and in documents that were
# classified as good three times. The word “Python”
# has only appeared in good documents.
# 
# The cc variable is a dictionary of how many times every
# classification has been used. This is needed for the probability
# calculations that we’ll discuss shortly.
# 
# The final instance variable, getfeatures, is the function that will
# be used to extract the features from the items being classified
# —in this example, it is the getwords function wejust defined.

# 
# We also need helper methods to increment and get the counts of
# the features
# 
    
  # Increase the count of a feature/category pair
  def incf(self,f,cat):
    self.fc.setdefault(f,{}) 
    self.fc[f].setdefault(cat,0)
    self.fc[f][cat]+=1

  # Increase the count of a category
  def incc(self,cat):
    self.cc.setdefault(cat,0)
    self.cc[cat]+=1

  # The number of times a feature has appeared in a category
  def fcount(self,f,cat):
    if f in self.fc and cat in self.fc[f]:
      return float(self.fc[f][cat])
    return 0.0

  # The number of items in a category
  def catcount(self,cat):
    if cat in self.cc:
      return float(self.cc[cat])
    return 0

  # The total number of items
  def totalcount(self):
    return sum(self.cc.values( ))

  # The list of all categories
  def categories(self):
    return self.cc.keys( )


# The train method takes an item (a document in this case) and a
# classification.
# 
# It uses the getfeatures function of the class to break the item
# into its separate features.
# 
# It then calls incf to increase the counts for this classification
# for every feature.
# 
# Finally, it increases the total count for this classification:

  def train(self,item,cat):
    features=self.getfeatures(item)
    # Increment the count for every feature with this category
    for f in features:
      self.incf(f,cat)
    # Increment the count for this category
    self.incc(cat)



# Lets check if the train method updates the counts correctly:
#
# >>> cl=docclass.classifier(docclass.getwords)
# >>> cl.train('the quick brown fox jumped over the lazy dog', 'good')
# the quick brown fox jumped over the lazy dog
# >>> cl.train('make quick money in the online casino', 'bad')
# make quick money in the online casino
# >>> cl.fcount('quick','good')
# 1.0
# >>> cl.fcount('quick','bad')
# 1.0
# >>> cl.fcount('jumped','bad')
# 0.0
# >>> 
#
# The fcount function returns the number of times a feature has
# appeared in a category.


# To save ourselves the bother of continually typing in the training
# data we can add a method to do this automatically.
# The sampletrain() method at the end of the file does this for us.
#


# Calculating Probabilities
#--------------------------------------------------
# 
# We now have counts for how often an feature appears in each category,
# so the next step is to convert these numbers into probabilities.
# 
# A probability is a number between 0 and 1, indicating how likely an
# event is.
# 
# We can calculate the probability that a word (feature)
# is in a particular category by dividing the number of times
# the word appears in a document in that category by the total number
# of documents in that category.
# 
# The method fprob does this calculation for us:

  def fprob(self,f,cat):
    if self.catcount(cat)==0: return 0

    # The total number of times this feature appeared in this 
    # category divided by the total number of items in this category
    return self.fcount(f,cat)/self.catcount(cat)

# The number returned by fprob is called a conditional probability.
# Conditional probabilities are usually written as Pr(A | B) and read as
# “the probability of A given B.”
# 
# In this example, the numbers we have now are Pr(word | classification);
# that is, for a given classification we calculate the probability
# that a particular word appears.
# 
# We can test this in our Python session:
#   
# >>> reload(docclass)
# <module 'docclass' from '/users/shensman/teaching/ml/labs/2019_20/Lab_ Bayes/docclass.py'>
# >>> cl=docclass.classifier(docclass.getwords)
# >>> docclass.sampletrain(cl)
# Nobody owns the water.
# the quick rabbit jumps fences
# buy pharmaceuticals now
# make quick money at the online casino
# the quick brown fox jumps
# >>> cl.fprob('quick','good')
# 0.66666666666666663
# >>> 
# 
# The word “quick” appears in two of the three documents classified
# as good, which means there’s a probability of
# Pr(quick| good) = 0.666 (a 2/3 chance)
# that a good document will contain that word.


# 
# Starting with a Reasonable Guess
#--------------------------------------------------
# 
# The fprob method gives an accurate result for the features and
# classifications it has seen so far, but it has a slight problem
#   —using only the information it has seen so far makes it
#   incredibly sensitive during early training and to words that
#   appear very rarely.
# 
#   In the sample training data, the word “money” only appears in
#   one document and is classified as bad because it is a casino ad.
#   Since the word “money” is in one bad document and no good ones,
#   the probability that it will appear in the good category using
#   fprob is now 0.
#   This is a bit extreme, since “money” might be a perfectly neutral
#   word that just happens to appear first in a bad document. It
#   would be much more realistic for the value to gradually approach
#   zero as a word is found in more and more documents with the
#   same category.
# 
# To get around this, we need to decide on an assumed probability,
# which will be used when we have very little information about
# the feature in question.
# 
# A good number to start with is 0.5.
# 
# We also need to decide how much to weight the assumed probability
#   —a weight of 1 means the assumed probability is weighted the
#   same as one word.
# 
# The weighted probability returns a weighted average of
# getprobability and the assumed probability.
#   -  In the “money” example, the weighted probability for the
#   word “money” starts at 0.5 for all categories.
#   After the classifier is trained with one bad document and finds
#   that “money” fits into the bad category, its probability becomes
#   0.75 for bad. This is because:
# 
#   (weight*assumedprob + count*fprob)/(count+weight)
#   = (1*1.0+1*0.5+)/(1.0 + 1.0)
#   = 0.75
# 
# >>> reload(docclass)
# <module 'docclass' from '/users/shensman/teaching/ml/labs/2019_20/Lab_ Bayes/docclass.py'>
# >>> cl=docclass.classifier(docclass.getwords)
# >>> cl.train('make quick money in the online casino','bad')
# make quick money in the online casino
# >>> cl.fcount('money','bad')
# 1.0
# >>> cl.catcount('bad')
# 1.0
# >>> cl.fprob('money','bad')
# 1.0
# >>> cl.weightedprob('money','bad',cl.fprob,1.0,0.5)
# 0.75
# >>> 

  def weightedprob(self,f,cat,prf,weight=1.0,ap=0.5):
    # Calculate current probability
    basicprob=prf(f,cat)

    # Count the number of times this feature has appeared in
    # all categories
    totals=sum([self.fcount(f,c) for c in self.categories()])

    # Calculate the weighted average
    bp=((weight*ap)+(totals*basicprob))/(weight+totals)
    return bp

# We can see how the probabilities change as more examples are
# given to the classifier by rerunning the sampletrain method.
# 
# >>> cl.weightedprob('money','bad',cl.fprob,1.0,0.5)
# 0.75
# >>> reload(docclass)
# <module 'docclass' from '/users/shensman/teaching/ml/labs/2019_20/Lab_ Bayes/docclass.py'>
# >>> cl=docclass.classifier(docclass.getwords)
# >>> docclass.sampletrain(cl)
# Nobody owns the water.
# the quick rabbit jumps fences
# buy pharmaceuticals now
# make quick money at the online casino
# the quick brown fox jumps
# >>> cl.weightedprob('money','good',cl.fprob)
# 0.25
# >>> docclass.sampletrain(cl)
# Nobody owns the water.
# the quick rabbit jumps fences
# buy pharmaceuticals now
# make quick money at the online casino
# the quick brown fox jumps
# >>> cl.weightedprob('money','good',cl.fprob)
# 0.16666666666666666
# >>> 
# 
# You can see how the classifier becomes more confident of the
# various word probabilities as they get pulled further from their
# assumed probability.
# 
# The assumed probability of 0.5 was chosen simply because it is
# halfway between 0 and 1. However, it’s possible that you might
# have better background information than that, even on a completely
# untrained classifier.
# 


# A Naïve Classifier
#--------------------------------------------------
# 
# 
# Once we have the probabilities of a document in a category
# containing a particular word, you need a way to combine the
# individual word probabilities to get the probability that
# an entire document belongs in a given category.
# 
# The classifier we will use to do this is called a naïve
# Bayesian classifier.
# 
# It is called naïve because it assumes that the probabilities being
# combinedare independent of each other.
#   - That is, the probability of one word in the document being in a
#   specific category is unrelated to the probability of the other words
#   being in that category.
# 
# This is actually a false assumption:
#   -you’ll probably find that documents containing the word “casino”
#   are much more likely to contain the word “money” than documents
#   about Python programming are.
# 
# This means that you can’t actually use the probability created by
# the naïve Bayesian classifier as the actual probability that a
# document belongs in a category, because the assumption of
# independence makes it inaccurate.
# 
# However, you can compare the results for different categories and
# see which one has the highest probability. In real life, despite
# the underlying flawed assumption, this has proven to be a
# surprisingly effective method for classifying documents.

# Probability of a Whole Document
#--------------------------------------------------
# 
# To use the naïve Bayesian classifier, we first have to determine
# the probability of an entire document being given a classification.
# 
# We are going to assume the probabilities are independent,
# which means we can calculate the probability of all of them by
# multiplying them together.
# 
#   - For example, suppose you’ve noticed that the word “Python”
#   appears in 20 percent of your bad documents
# 
#     Pr(Python | Bad) = 0.2
#     
#   and that the word “casino” appears in 80 percent of your bad
#   documents
# 
#   (Pr(Casino | Bad) = 0.8)
# 
#   You would then expect the independent probability of both
#   words appearing in a bad document to be:
#     
#   Pr(Python & Casino | Bad)—to be 0.8 × 0.2 = 0.16.
# 
# From this you can see that calculating the entire document
# probability is just a matter of multiplying together all
# the probabilities of the individual words in that document.
# 
# We will create a subclass of classifier called naivebayes,
# and create a docprob method that extracts the features (words)
# and multiplies all their probabilities together to get an
# overall probability:


class naivebayes(classifier):
  
  def __init__(self,getfeatures):
    classifier.__init__(self,getfeatures)
    #used the hold the thresholds for each category
    #(this is explained later in the lab)
    self.thresholds={}
  
  def docprob(self,item,cat):
    features=self.getfeatures(item)   

    # Multiply the probabilities of all the features together
    p=1
    for f in features: p*=self.weightedprob(f,cat,self.fprob)
    return p

# We know how to calculate Pr(Document | Category), but this
# isn’t very useful by itself.
# 
# In order to classify documents, we really need:
#   Pr(Category | Document).
#   
# In other words, given a specific document, what’s the
# probability that it fits into this category?
# 
# Fortunately, a British mathematician named Thomas Bayes
# figured out how to do this about 250 years ago.
# 
# Bayes’ Theorem is a way of flipping around conditional
# probabilities. It’s usually written as:
#   
#   Pr(A | B) = Pr(B | A) x Pr(A)/Pr(B)
# 
# In the example, this becomes:
# 
#   Pr(Category | Document) =
#     Pr(Document | Category) x Pr(Category) / Pr(Document)
# 
# We know how to calculate Pr(Document | Category),
# but what about the other two values in the equation?
# 
# Well, Pr(Category) is the probability that a randomly
# selected document will be in this category, so it’s just
# the number of documents in the category divided by the
# total number of documents.
# 
# As for Pr(Document), we could calculate it, but that
# would be unnecessary effort. Remember that the results of
# this calculation will not be used as a real probability.
# Instead, the probability for each category will be
# calculated separately, and then all the results will be
# compared.
# Since Pr(Document) is the same no matter what category
# the calculation is being done for, it will scale the
# results by the exact same amount, so we can safely
# ignore this term.
# 
# The prob method calculates the probability of the
# category, and returns the product of
# Pr(Document | Category) and Pr(Category). 

  def prob(self,item,cat):
    catprob=self.catcount(cat)/self.totalcount()
    docprob=self.docprob(item,cat)
    return docprob*catprob


# Lets try our Naieve Bayes classifier out in out
# Python session:
# 
# >>> reload(docclass)
# <module 'docclass' from '/users/shensman/teaching/ml/labs/2019_20/Lab_ Bayes/docclass.py'>
# >>> cl=docclass.naivebayes(docclass.getwords)
# >>> docclass.sampletrain(cl)
# Nobody owns the water.
# the quick rabbit jumps fences
# buy pharmaceuticals now
# make quick money at the online casino
# the quick brown fox jumps
# >>> cl.prob('quick rabbit','good')
# quick rabbit
# 0.15624999999999997
# >>> cl.prob('quick rabbit','bad')
# quick rabbit
# 0.050000000000000003
# >>> 
# 
# Based on the training data, the phrase “quick rabbit”
# is considered a much better candidate for the good
# category than the bad.


# Choosing a Category
# 
# The final step in building the naïve Bayes classifier is
# actually deciding in which category a new item belongs.
# 
# The simplest approach would be to calculate the probability
# of this item being in each of the different categories and
# to choose the category with the best probability.
# 
# If you were just trying to decide the best place to put something,
# this would be a feasible strategy, but
#   - in many applications the categories can’t be considered equal,
#   - and in some applications it’s better for the classifier to
#   admit that it doesn’t know the answer than to decide that the
#   answer is the category with a marginally higher probability.
# 
# In the case of spam filtering, it’s much more important to avoid
# having good email messages classified as spam than it is to catch
# every single spam message.
# 
# To deal with this problem, we can set up a minimum threshold for
# each category. For a new item to be classified into a particular
# category, its probability must be a specified amount larger than
# the probability for any other category. This specified
# amount is the threshold.
#   - For spam filtering, the threshold to be filtered to bad could
#   be 3, so that the probability for bad would have to be 3 times
#   higher than the probability for good.
#   - The threshold for good could be set to 1, so anything would
#   be good if the probability were at all better than for the bad
#   category.
#   - Any message where the probability for bad is higher, but not
#   3 times higher, would be classified as unknown.
# 
# We will use the thresholds instance variable declared in the class
# init function to hold these threshols.
# 
# We will also add some methods to set and get the threshold values,
# returning 1.0 as default:
  
  def setthreshold(self,cat,t):
    self.thresholds[cat]=t
    
  def getthreshold(self,cat):
    if cat not in self.thresholds: return 1.0
    return self.thresholds[cat]

# We can now implement a classify method.
# 
# This method will:
#   (1) calculate the probability for each category,
#   (2) determine which one is the largest
#   (3) determine whether it exceeds the next largest
#   by more than its threshold.
#   (4) If none of the categories can accomplish this, the
#   method just returns the default values.
 
  def classify(self,item,default=None):
    probs={}
    # Find the category with the highest probability
    max=0.0
    for cat in self.categories():
      probs[cat]=self.prob(item,cat)
      if probs[cat]>max: 
        max=probs[cat]
        best=cat

    # Make sure the probability exceeds threshold*next best
    for cat in probs:
      if cat==best: continue
      if probs[cat]*self.getthreshold(best)>probs[best]: return default
    return best

# Thats everything we need for a Naieve Bayes classifier!
# 
# Lets try it out in our Python session:
# 
# >>> reload(docclass)
# <module 'docclass' from 'docclass.pyc'>
# >>> cl=docclass.naivebayes(docclass.getwords)
# >>> docclass.sampletrain(cl)
# >>> cl.classify('quick rabbit',default='unknown')
# 'good'
# >>> cl.classify('quick money',default='unknown')
# 'bad'
# >>> cl.setthreshold('bad',3.0)
# >>> cl.classify('quick money',default='unknown')
# 'unknown'
# >>> for i in range(10): docclass.sampletrain(cl)
# ...
# >>> cl.classify('quick money',default='unknown')
# 'bad'
# 
# 
# We could extend our classifier to use different features.
# 
# We can also alter the thresholds and see how
# the results are affected.
# 
# The thresholds will also be different for other
# applications that involve document filtering; sometimes a
# ll categories will be equal, or filtering to “unknown” will
# be unacceptable.


# dump some sample training data in a classifier
def sampletrain(cl):
  cl.train('Nobody owns the water.','good')
  cl.train('the quick rabbit jumps fences','good')
  cl.train('buy pharmaceuticals now','bad')
  cl.train('make quick money at the online casino','bad')
  cl.train('the quick brown fox jumps','good')
