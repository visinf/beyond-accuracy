from os.path import join as pjoin
import numpy as np
from shutil import copyfile
import os
import linecache as lc

import quba_constants as c

categories = None
WNIDs = None
IMAGENET_CATEGORIES_FILE = pjoin(c._CODE_DIR, "helper", "categories.txt")


def get_ilsvrc2012_categories():
    """
        Return the first item of each synset of the ilsvrc2012 categories.
        Categories are lazy-loaded the first time they are needed.
    """

    global categories
    if categories is None:
        categories = []
        with open(IMAGENET_CATEGORIES_FILE) as f:
            for line in f:
                categories.append(get_category_from_line(line))

    return categories


def get_ilsvrc2012_WNIDs():
    """
        Return the first item of each synset of the ilsvrc2012 categories.
        Categories are lazy-loaded the first time they are needed.
    """

    global WNIDs
    if WNIDs is None:
        WNIDs = []
        with open(IMAGENET_CATEGORIES_FILE) as f:
            for line in f:
                WNIDs.append(get_WNID_from_line(line))

    return WNIDs


def get_category_from_line(line):
    """Return the category without anything else from categories.txt"""

    category = line.split(",")[0][10:]
    category = category.replace(" ", "_")
    category = category.replace("\n", "")
    return category


def get_WNID_from_line(line):
    """Return the WNID without anything else from categories.txt"""
    
    WNID = line.split(" ")[0]
    return WNID


def get_WNID_from_index(index):
    """Return WNID given an index of categories.txt"""
    assert (index >= 0 and index < 1000), "index needs to be within [0, 999]"

    file_path = IMAGENET_CATEGORIES_FILE
    assert(os.path.exists(file_path)), "path to categories.txt wrong!"
    line = lc.getline(file_path, index+1)
    return line.split(" ")[0]

