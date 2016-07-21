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

#############################
# FEATURE extraction settings
#############################

#Signals: Each image filename to be tested should end with its class ID: Ex.: 'im00.000948_2' is a Rectangle


# 6 CLASSES --  6 CLASSES --  6 CLASSES --  6 CLASSES --  6 CLASSES --
#Classes ID's:'Background': 0, 'GiveWay':1, 'Rectangles':2, 'Circles':3, 'Squares':4, 'Triangles':5
# labels = ['Background', 'GiveWay', 'Rectangles', 'Circles', 'Squares', 'Triangles']


# 15 CLASSES --    15 CLASSES --   15 CLASSES --   15 CLASSES --
labels = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14']
index_background = '14'

num_classes = len(labels)

# featuresToExtract = ['HOG']
# featuresToExtract = ['LBP']
# featuresToExtract = ['HOG', 'LBP']
# featuresToExtract = ['HOGRGB', 'LBP']
featuresToExtract = ['HOGRGB', 'LBP', 'CANNY']

# LBP Parameters
lbp_win_shape = (16, 16)
lbp_win_step = lbp_win_shape[0]/2
lbp_radius = 1
lbp_n_points = 8 * lbp_radius
lbp_METHOD = 'nri_uniform'
lbp_n_bins = 59 # NRI uniform LBP has 59 values

# HOG Parameters
hog_orientations = 9
hog_pixels_per_cell = (8, 8)
hog_cells_per_block = (2, 2)
hog_normalise = True

##################
# DATASET Settings
##################

datasetRoot = '../DataSetMulticlass/Train'


#Location to store the features of of the positive and negative sample images
featuresFolder = '-'.join(featuresToExtract)
# positiveFeaturesPath = 'Features/'+featuresFolder+'/'+positive_folder
negativeFeaturesPath = 'Features/'+featuresFolder+'/'

testFolderPath = '../DataSetMulticlass/Test'
annotationsFolderPath = '../DataSet/Test/gt'
resultsFolder = 'ResultsMulticlass/'

##############################
# MODEL TO TEST SETTINGS
##############################

# model = 'SVM_Linear'
model = 'SVM_RBF'

# Location of the model
modelFeatures = '-'.join(featuresToExtract)

# signalsToTrain. Kind of Signal to evaluate. Give way, Square, Rectangle, Circle or Triangle.
# modelPath = 'Models/'+model+'_'+modelFeatures+'_'+signalsToTrain+'_' + str(signalsToTrainIndex) + '.model'
modelRootPath = 'Models/'

# SVM_Linear parameters
svm_Linear_C = 1
svm_Linear_penalty = 'l2'
svm_Linear_dual = False
svm_Linear_tol = 0.0001
svm_Linear_fit_intercept = True
svm_Linear_intercept_scaling = 100

#SVM_RBF parameters
svm_RBF_C = 1

#Min maximum score to be considered a signal
min_score = 0
