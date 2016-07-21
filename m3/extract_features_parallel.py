#!/usr/bin/python
#Copyright 2015 CVC-UAB

# extract_features.py: Extract features for a given dataset, and stores them.
# TODO: Modify Config.py to point to the location where you stored the dataset.

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

from Tools import feature_extractor

from skimage import io
import os
import Config as cfg
from skimage import util
from skimage.transform import resize
from joblib import Parallel, delayed
import time
import pickle
import random

def main():

    # Create necessary directories to save the features of the images
    if not os.path.exists(cfg.positiveFeaturesPath):
        os.makedirs(cfg.positiveFeaturesPath)
    if not os.path.exists(cfg.negativeFeaturesPath):
        os.makedirs(cfg.negativeFeaturesPath)

    #Extract features for positive samples
    print 'Extracting features from images in '+cfg.positiveInputPath
    fileList = os.listdir(cfg.positiveInputPath)
    imagesList = filter(lambda element: '.jpg' in element, fileList)


    Parallel(n_jobs=-1)(delayed(extractAndStoreFeaturesPositives)(filename) for filename in imagesList)

    #Extract features for negative samples
    print 'Extracting features from images in '+cfg.negativeInputPath
    fileList = os.listdir(cfg.negativeInputPath)
    imagesList = filter(lambda element: '.jpg' in element, fileList)

    Parallel(n_jobs=-1)(delayed(extractAndStoreFeaturesNegatives)(filename) for filename in imagesList)

def extractAndStoreFeaturesPositives(filename):

    imagepath = cfg.positiveInputPath + '/' + filename

    print 'Extracting features for ' + imagepath

    imageOriginal = io.imread(imagepath, as_grey=True)

    outputpath = cfg.positiveFeaturesPath+'/'+filename+'_'+cfg.signalsToTrain+'.feat'

    if os.path.exists(outputpath):
        print 'Features for ' + imagepath + '. Delete the file if you want to replace.'
    else:
        extractFeaturesSingleImage(imageOriginal, outputpath)


def extractAndStoreFeaturesNegatives(filename):

    imagepath = cfg.negativeInputPath + '/' + filename

    print 'Extracting features for ' + imagepath

    imageOriginal = io.imread(imagepath, as_grey=True)

    outputpath = cfg.negativeFeaturesPath+'/'+filename+'.feat'

    if os.path.exists(outputpath):
        print 'Features for ' + imagepath + '. Delete the file if you want to replace.'
    else:
        extractFeaturesSingleImage(imageOriginal, outputpath)


def extractFeaturesSingleImage(imageOriginal, outputpath):

    #Read the image as bytes (pixels with values 0-255)
    image = util.img_as_ubyte(imageOriginal)

    #Extract the features
    feats = feature_extractor.extractFeatures(image)

    #Save the features to a file
    outputFile = open(outputpath, "wb")
    pickle.dump(feats, outputFile)
    outputFile.close()


if __name__ == '__main__':
    main()

