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

import extract_features
import train_model
import test_image
import test_folder
import test_folder_parallel
import show_results
import evaluate_results


# ---------------------------------------
# Extracts the features for all the images
# extract_features.run()

# Extracts the features for all the images parallel
# execfile("extract_features_parallel.py")

# ---------------------------------------
# Train the classifier
# train_model.run()

# ---------------------------------------
# Tests a single image and shows the results
# test_image.run()

# ---------------------------------------
# Test a whole folder
# test_folder.run()

# Test a whole folder parallel
execfile("test_folder_parallel.py")

# ---------------------------------------
# Saves the resulting images
# show_results.run()

# ---------------------------------------
# Runs the evaluation
# evaluate_results.run()

# Runs the evaluation parallel
# execfile("evaluate_results_parallel.py")

# Runs the evaluation parallel Compensatio modes
# execfile("evaluate_results_parallel_comp.py")

# ---------------------------------------
# Extracts the features for all the images for multi class classification
# extract_features_multiclass.run()

# ---------------------------------------
# Test a whole folder for multiclass
# multiClassification.run()

# ---------------------------------------
# Runs the evaluation for multiclass
# evaluate_results_multiclass.run()
