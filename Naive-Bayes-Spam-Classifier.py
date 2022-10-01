"""
Matthew Twete

A basic program to classify spam e-mails using a naive bayes classifier.
"""

#Import needed libraries
import numpy as np
import math
from sklearn.metrics import confusion_matrix

#Import data and randomly suffle to before separating into test and training set
allData = np.genfromtxt('spambase.data',delimiter=',')
np.random.shuffle(allData)

#Split into test and training sets, the training set will have 2301 instances
#and the test set will have 2300
trainData = allData[0:2301,:]
testData = allData[2301:,:]



#Class for running a naive bayes spam classifier on email data. It will take training and test
#email data to set up a classifier and then run the classifier on the test data.  
class nBayes:

    
    #Naive bayes classifer class constructor. The arguments are:
    #testData, the test data to run the classifier on
    #trainData, the training data to set up the classifier with
    def __init__(self, testData, trainData):
        #Training data array
        self.test = testData
        #Test data array
        self.train = trainData
        #Small value to use for the standard deviation of features that end up being 0,
        #to avoid dividing by zero errors
        self.epsilon = 0.00001
        #Spam class prior probability, initialized to 0 
        self.spamPrior = 0
        #Real email class prior probability, initialized to 0 
        self.realPrior = 0
        #Number of features in the data, the minus one is because the last value in each input
        #is the class label, so the number of features is the number of columns minus one
        self.numFeatures = testData.shape[1] - 1
        #Array to hold the average value of the features in the spam class from the training data
        self.spamAvg = np.zeros(self.numFeatures)
        #Array to hold the average value of the features in the real class from the training data
        self.realAvg = np.zeros(self.numFeatures)
        #Arrays to hold the standard deviations for the features in the two classes
        #set up with an initial value of 0.00001 which will be used if the calculated
        #value for the standard deviations is 0
        self.spamStd = np.full(self.numFeatures, self.epsilon)
        self.realStd = np.full(self.numFeatures, self.epsilon)
        #Array to hold the predicted classes of each input in the test set
        self.predictions = np.zeros(self.test.shape[0])
     
        
    #Function to calculate and initialize the values for the classifier based on the training data.
    #It will separate the training data into the two classes to calculate the class priors and the 
    #means/standard deviations of each feature in the training data for each of the classes.
    def initClassifier(self):
        #Separate test data into classes
        spam = self.train[np.where(self.train[:,-1] == 1)]
        real = self.train[np.where(self.train[:,-1] == 0)]
        #Calculate the class priors
        self.spamPrior = spam.shape[0]/self.train.shape[0]
        self.realPrior = real.shape[0]/self.train.shape[0]
        #Loop over the features in each class to get the averages and standard deviations 
        for i in range(self.numFeatures):
            #Calculate and store feature averages
            spamave = np.mean(spam[:,i])
            realave = np.mean(real[:,i])
            self.spamAvg[i] = spamave
            self.realAvg[i] = realave
            #Calculate and store feature stds, only if they are non-zero
            #otherwise the default value of 0.0001 will be used
            spamstd = np.std(spam[:,i])
            realstd = np.std(real[:,i])
            if (spamstd != 0):
                self.spamStd[i] = spamstd
            if (realstd != 0):
                self.realStd[i] = realstd
    
    
    #Function that represents a guassian distribution. The arguments are:
    #x, the variable to calculate the guassian distribution value of
    #mean, the mean to use for the guassian distribution
    #std, the standard deviation for the guassian distribution
    #The funcion will return the value of the guassian distribution for x.
    def gaussian(self, x, mean, std):
        return ((1/(math.sqrt(2*np.pi)*std))*np.exp(-((x-mean)**2)/(2*std**2)))
    
    
    #Function to classify an input with the naive bayes classifier. The only argument is:
    #x, the email input vector to classify
    #The function will return 1 if the input is classified as spam and 0 if not classified as spam.
    def classify(self,x):
        #Get the prior probabilities for the two classes, they will be used with the sum of probabilites 
        #for the given input features for each class
        spam = self.spamPrior
        real = self.realPrior
        #Iterate over the features and calculate the log of the gaussian value for that feature for each
        #class, adding that value to the probability sum of each class
        for i in range(self.numFeatures):
            spam += np.log(self.gaussian(x[i],self.spamAvg[i],self.spamStd[i]))
            real += np.log(self.gaussian(x[i],self.realAvg[i],self.realStd[i]))
        #If the probablity (argmax) of the input being spam is higher, return 1 to indicate that
        #input was classified as spam, otherwise return 0 to indicate it was classified as not spam
        if (spam > real):
            return 1
        else:
            return 0
    
    
    #Function to run the classifier on the test set and display the results, including the 
    #confusion matrix, classifier accuracy, precision and recall. 
    def run(self):
        #Calculate the values needed to run the classifier
        self.initClassifier()
        #iterate over the test set and classify each input
        for i in range(self.test.shape[0]):
            self.predictions[i] = self.classify(self.test[i])
        #Create the confusion matrix, (Note: Prof. Rhodes said it was fine to use
        #an imported function for this)
        self.confmat = confusion_matrix(self.test[:,-1],self.predictions)
        #Get the true negative, false positive, false negative and true positive values from the 
        #confusion matrix
        tn, fp, fn, tp = self.confmat.ravel()
        #Print the confusion matrix, accuracy, precision and recall of the classifier on the test set 
        print("")
        print("")
        print(self.confmat)
        print("Accuracy on test set: ", (tp+tn)/(tn + fp + fn + tp))
        print("Precision on test set: ",tp/(tp+fp))
        print("Recall on test set: ",tp/(tp+fn))
        
            
    
    
            
#Create the class instance and run the classifier
c = nBayes(testData,trainData)
c.run()
        
        