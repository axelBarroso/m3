#!/usr/bin/python
#Copyright 2015 CVC-UAB

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

__author__ = "Miquel Ferrarons, David Vazquez"
__copyright__ = "Copyright 2015, CVC-UAB"
__credits__ = ["Miquel Ferrarons", "David Vazquez"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Miquel Ferrarons"

import numpy as np
import Config as cfg
import pickle
from sklearn import svm
from sklearn import linear_model

import os

def loadImageFeatures(featuresPath):
    # print 'Loading features from '+str(featuresPath)
    file = open(featuresPath, 'r')
    return pickle.load(file)

def run():

    for thisClass in range(cfg.num_classes):
        curSignalsToTrain = cfg.labels[thisClass]
        featuresFolder = '-'.join(cfg.featuresToExtract)
        curPositiveFeaturesPath = 'Features/'+featuresFolder+'/'+cfg.labels[thisClass]
        print 'Training model: '+cfg.labels[thisClass]


        containsSpecificModel = curSignalsToTrain+'.feat'

        #List all the files .feat in the positives directory
        positiveList = os.listdir(curPositiveFeaturesPath)
        positiveList = filter(lambda element: containsSpecificModel in element, positiveList)


        negativeSamplesCount = 0
        # Count features of negative classes: all but signalsToTrain. (One vs all)
        for label in cfg.labels:
            if label is not curSignalsToTrain:
                        negativeList = os.listdir(cfg.negativeFeaturesPath + label)
                        negativeList = filter(lambda element: '.feat' in element, negativeList)
                        negativeSamplesCount = negativeSamplesCount + len(negativeList)


        #Count how many samples we have
        positiveSamplesCount = len(positiveList)
        samplesCount = positiveSamplesCount + negativeSamplesCount


        #Load the features of the first element, to obtain the size of the feature vectors.
        filepath = curPositiveFeaturesPath + '/'+positiveList[0]
        file = open(filepath, 'r')
        feats = pickle.load(file)
        featuresLength = len(feats)

        #Initialize the structure that we will pass to the model for training
        # X will be the samples for training
        # y will be the labels
        X = np.zeros(shape=(samplesCount, featuresLength))
        y = np.append(np.ones(shape=(1, positiveSamplesCount)),
                      -1*np.ones(shape=(1, negativeSamplesCount)))

        # Load all the positive feature vectors in X
        count = 0
        for filename in positiveList:
            filepath = curPositiveFeaturesPath+'/'+filename
            X[count] = loadImageFeatures(filepath)
            count += 1


        # Load features of negative classes: all but signalsToTrain. (One vs all)
        for label in cfg.labels:
            if label is not curSignalsToTrain:
                negativeList = os.listdir(cfg.negativeFeaturesPath + label)
                negativeList = filter(lambda element: '.feat' in element, negativeList)
                for filename in negativeList:
                    filepath = cfg.negativeFeaturesPath+'/' + label + '/' + filename
                    print filepath
                    X[count] = loadImageFeatures(filepath)
                    count += 1



        # Train the classifier with X and y, and some given parameters
        if cfg.model == 'SVM_Linear':
            print 'Training Linear SVM....'
            model = svm.LinearSVC(C=cfg.svm_Linear_C,
                                    #loss='hinge',# loss
                                    penalty=cfg.svm_Linear_penalty,
                                    dual=cfg.svm_Linear_dual,
                                    tol=cfg.svm_Linear_tol,
                                    fit_intercept=cfg.svm_Linear_fit_intercept,
                                    intercept_scaling=cfg.svm_Linear_intercept_scaling)
        elif cfg.model == 'SVM_RBF':
            print 'Training SVM with RBF kernel....'
            #Default kernel on SVC is RBF
            model = svm.SVC(C=cfg.svm_RBF_C)
        else:
            print 'ERROR: Model can only be SVM_Linear or SVM_RBF'
            exit(0)

        model.fit(X, y)
        #Obtain the model score for the training set
        print 'MODEL score'
        print model.score(X, y)

        #Save the model
        modelPath = 'Models/' + cfg.model + '--' + cfg.modelFeatures + '_' + curSignalsToTrain + '.model'
        if not os.path.exists('Models/'):
            os.makedirs('Models/')

        outputModelFile = open(modelPath, 'wb')
        pickle.dump(model, outputModelFile)

if __name__ == '__main__':
    run()