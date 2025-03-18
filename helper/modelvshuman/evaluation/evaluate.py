"""
Generic evaluation functionality: evaluate on several datasets.
"""

import csv
import os
from os.path import join as pjoin

import quba_constants as c
import shutil
import numpy as np
from math import isclose

IMAGENET_LABEL_FILE = pjoin(c._CODE_DIR, "evaluation", "imagenet_labels.txt")

class ResultPrinter():

    def __init__(self, model_name, dataset,
                 data_parent_dir=c._RAW_DATA_DIR):

        self.model_name = model_name
        self.dataset = dataset
        self.data_dir = pjoin(data_parent_dir, dataset.name)
        self.decision_mapping = self.dataset.decision_mapping
        self.info_mapping = self.dataset.info_mapping
        self.session_list = []

    def create_session_csv(self, session):

        self.csv_file_path = pjoin(self.data_dir,
                                   self.dataset.name + "_" +
                                   self.model_name.replace("_", "-") + "_" +
                                   session + ".csv")

        if os.path.exists(self.csv_file_path):
            # print("Warning: the following file will be overwritten: "+self.csv_file_path)
            os.remove(self.csv_file_path)

        directory = os.path.dirname(self.csv_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.index = 0

        # write csv file header row
        with open(self.csv_file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["subj", "session", "trial",
                             "rt", "object_response", "category",
                             "condition", "imagename"])


    def print_batch_to_csv(self, object_response,
                           batch_targets, paths):

        for response, target, path in zip(object_response, batch_targets, paths):
            session_name, img_name, condition, category = self.info_mapping(path)
            session_num = int(session_name.split("-")[-1])

            if not session_num in self.session_list:
                self.session_list.append(session_num)
                self.create_session_csv(session_name)

            with open(self.csv_file_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow([self.model_name,
                                 str(session_num), str(self.index+1),
                                 "NaN", response[0], category,
                                 condition, img_name])
            self.index += 1
