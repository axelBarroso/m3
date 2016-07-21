#!/usr/bin/python
#Copyright 2015 CVC-UAB

# extract_features.py: Extract features for a given dataset, and stores them.

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
import pickle
import random

def run():


    #Extract features for positive samples
    featuresFolder = '-'.join(cfg.featuresToExtract)

    for i in range(cfg.num_classes):

        curPositiveFeaturesPath = 'Features/'+featuresFolder+'/'+cfg.labels[i]
        curPositiveInputPath = cfg.datasetRoot+'/'+cfg.labels[i]
        if not os.path.exists(curPositiveFeaturesPath):
            os.makedirs(curPositiveFeaturesPath)
        print 'Extracting features from images in: '+cfg.labels[i]
        extractAndStoreFeatures(curPositiveInputPath, curPositiveFeaturesPath, '.jpg', cfg.labels[i])




def extractAndStoreFeatures(inputFolder, outputFolder, extension, curClass):


    #List all files
    fileList = os.listdir(inputFolder)
    #Select only files that end with .png
    #imagesList = filter(lambda element: '.ppm' in element, fileList)
    imagesList = filter(lambda element: extension in element, fileList)

    #Imposed size for training crops

    for filename in imagesList:

        imagepath = inputFolder + '/' + filename

        # print 'Extracting features for ' + imagepath

        image = io.imread(imagepath, as_grey=True)

        outputpath = outputFolder+'/'+filename+'_'+curClass+'.feat'

        if os.path.exists(outputpath):
            print 'Features for ' + imagepath + '. Delete the file if you want to replace.'
            continue

        extractFeaturesSingleImage(image, outputpath, imagepath)



def extractFeaturesSingleImage(imageOriginal, outputpath, imagepath):

    #Read the image as bytes (pixels with values 0-255)
    image = util.img_as_ubyte(imageOriginal)

    #Extract the features
    feats = feature_extractor.extractFeatures(image, imagepath)

    #Save the features to a file
    outputFile = open(outputpath, "wb")
    pickle.dump(feats, outputFile)
    outputFile.close()




if __name__ == '__main__':
    run()

