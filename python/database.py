"""Class for Liver Tumore Segmentation database management."""

import os
import natsort as ns
import numpy as np


class LiTSdb(object):
    """LiTSdb class for data path and protocol split definitions.

    Attributes:
        db_path (str): path to the LiTS database
        train_db_batches: list of folders containing training data
        test_batch: list of folders containing testing data

        split_ratios: list of possible training data split ratios
            into develop-valid-eval subsets
        split_ratio: selected training data split ratio

        training_subjects: vector for storing subjects' names
           from the training batches
        development_subjects: vector for storing subjects' names
           for algorithm development/training
        validation_subjects: vector for storing subjects' names
           for algorithm validation
        evaluation_subjects: vector for storing subjects' names
           for algorithm evaluation
        testing_subjects: vector for storing subjects' names
           from the testing batch

        n_train: the total number of training subjects
        n_develop: the total number of development subjects
        n_valid: the total number of validation subjects
        n_eval: the total number of evaluation subjects
        n_test: the total number of testing subjects

    Methods:

        load_train_subjects_names: loading train subjects' names
        load_test_subjects_names: loading test subjects' names

        train_data_split: splitting data into development, validation
            and evaluation parts
        empty_split: reseting data split

        get_number_of_training: get the total number of training subjects
        get_number_of_development: get the total number of development subjects
        get_number_of_validation: get the total number of validation subjects
        get_number_of_evaluation: get the total number of evaluation subjects
        get_number_of_testing: get the total number of testing subjects

        get_train_subject_name: get subject's name from the training set
            at required position
        get_develop_subject_name: get subject's name from the development set
            at required position
        get_valid_subject_name: get subject's name from the validation set
            at required position
        get_eval_subject_name: get subject's name from the evaluation set
            at required position
        get_test_subject_name: get subject's name from the testing set
            at required position

        get_train_paths: get training subject's volume and segmentation paths
        get_train_volume_path: get training subject's volume path
        get_train_segmentation_path: get training subject's segmentation path
        get_train_meta_segmentation_path: get training subject's
          meta segmentation path
        get_test_volume_path: get testing subject's volume path
        get_test_segmentation_path: get testing subject's segmentation path
    """

    train_db_batches = ['Training Batch 1', 'Training Batch 2']
    test_batch = "Testing Batch"
    split_ratios = [[60, 20, 20], [50, 25, 25], [80, 10, 10]]

    def __init__(self, db_path):
        """Initialization method for LiTSdb object.

        Arguments:
            db_path (str): path to the LiTSdb database
        """
        self.db_path = db_path
        self.split_ratio = 0

        self.training_subjects = []
        self.development_subjects = []
        self.validation_subjects = []
        self.evaluation_subjects = []
        self.testing_subjects = []

        self.n_train = 0
        self.n_develop = 0
        self.n_valid = 0
        self.n_eval = 0
        self.n_test = 0

    def load_train_subjects_names(self):
        """Create list of training subjects."""
        for tf in self.train_db_batches:
            files = ns.natsorted(os.listdir(os.path.join(self.db_path, tf)))
            for f in files:
                if f.startswith('volume'):
                    s_name = str.split(str.split(f, '.')[0], '-')[-1]
                    self.training_subjects.append(s_name)
        np.random.seed(1)
        np.random.shuffle(self.training_subjects)
        self.n_train = len(self.training_subjects)

    def load_test_subjects_names(self):
        """Create list of testing subjects."""
        files = os.listdir(os.path.join(self.db_path, self.test_batch))
        for f in files:
            if f.startswith('test-volume'):
                s_name = str.split(str.split(f, '.')[0], '-')[-1]
                self.testing_subjects.append(s_name)
        self.n_test = len(self.testing_subjects)

    def train_data_split(self, selected_sr, selected_ss):
        """Split training data into develop, valid and eval subsets."""
        """
            Arguments:
                selected_sr: ordinal number of the selected split ratio
                selected_ss: ordinal number of split shift
        """
        assert selected_sr < len(self.split_ratios),\
            "The total number of possible split ratios is: %d"\
            % len(self.split_ratios)

        max_shifts = 100 / self.split_ratios[selected_sr][-1]

        assert selected_ss < max_shifts,\
            "The total number of split shifts is: %d" % max_shifts

        self.empty_split()

        n = float(self.n_train) / max_shifts
        self.n_develop = int(self.split_ratios[selected_sr][0] /
                             (100 / max_shifts) * n)

        self.n_valid = int(self.split_ratios[selected_sr][1] /
                           (100 / max_shifts) * n)

        self.n_eval = self.n_train - self.n_develop - self.n_valid

        for i in range(self.n_develop):
            self.development_subjects.\
                append(self.training_subjects[(selected_ss * self.n_eval + i) %
                                              self.n_train])

        for i in range(self.n_valid):
            self.validation_subjects.\
                append(self.training_subjects[(selected_ss * self.n_eval +
                                               self.n_develop + i) %
                                              self.n_train])

        for i in range(self.n_eval):
            self.evaluation_subjects.\
                append(self.training_subjects[(selected_ss * self.n_eval +
                                               self.n_develop +
                                               self.n_valid + i) %
                                              self.n_train])

    def empty_split(self):
        """Empty dev-valid-eval split."""
        self.n_develop = 0
        self.n_valid = 0
        self.n_eval = 0
        self.development_subjects = []
        self.validation_subjects = []
        self.evaluation_subjects = []

    def get_number_of_training(self):
        """Return number of training samples."""
        return self.n_train

    def get_number_of_development(self):
        """Return number of development samples."""
        return self.n_develop

    def get_number_of_validation(self):
        """Return number of validation samples."""
        return self.n_valid

    def get_number_of_evaluation(self):
        """Return number of evaluation samples."""
        return self.n_eval

    def get_number_of_testing(self):
        """Return number of test samples."""
        return self.n_test

    def get_train_subject_name(self, position):
        """Get train subject name from the training list."""
        """
            Arguments:
                position: position of the name in the list
            Returns:
                train subject's name
        """
        assert position < self.n_train,\
            "The total number of training samples is: %d" % self.n_train
        return self.training_subjects[position]

    def get_develop_subject_name(self, position):
        """Get develop subject name from the development list."""
        """
            Arguments:
                position: position of the name in the list
            Returns:
                develop subject's name
        """
        assert position < self.n_develop,\
            "The total number of development samples is: %d" % self.n_develop
        return self.development_subjects[position]

    def get_valid_subject_name(self, position):
        """Get valid subject name from the validation list."""
        """
            Arguments:
                position: position of the name in the list
            Returns:
                valid subject's name
        """
        assert position < self.n_valid,\
            "The total number of validation samples is: %d" % self.n_valid
        return self.validation_subjects[position]

    def get_eval_subject_name(self, position):
        """Get eval subject name from the evaluation list."""
        """
            Arguments:
                position: position of the name in the list
            Returns:
                eval subject's name
        """
        assert position < self.n_eval,\
            "The total number of evaluation samples is: %d" % self.n_eval
        return self.evaluation_subjects[position]

    def get_test_subject_name(self, position):
        """Get test subject name from the testing list."""
        """
            Arguments:
                position: position of the name in the list
            Returns:
                test subject's name
        """
        assert position < self.n_test,\
            "The total number of testing samples is: %d" % self.n_test
        return self.testing_subjects[position]

    def get_train_paths(self, subject_name):
        """Create train volume and segmentation path."""
        """
        Arguments:
            subject_name: name of the subject
        Returns:
            full train volume and segmentation paths
        """
        if (int(subject_name) < 28):
            db_batch = "/Training Batch 1"
        else:
            db_batch = "/Training Batch 2"

        volume_path = self.db_path + db_batch + "/volume-" +\
            subject_name + ".nii"

        segmentation_path = self.db_path + db_batch + "/segmentation-" +\
            subject_name + ".nii"

        return volume_path, segmentation_path

    def get_train_volume_path(self, subject_name):
        """Create train volume path."""
        """
        Arguments:
            subject_name: name of the subject
        Returns:
            full train volume path
        """
        if (int(subject_name) < 28):
            db_batch = "/Training Batch 1"
        else:
            db_batch = "/Training Batch 2"

        volume_path = self.db_path + db_batch + "/volume-" +\
            subject_name + ".nii"

        return volume_path

    def get_train_segmentation_path(self, subject_name):
        """Create train segmentation path."""
        """
        Arguments:
            subject_name: name of the subject
        Returns:
            full train segmentation path
        """
        if (int(subject_name) < 28):
            db_batch = "/Training Batch 1"
        else:
            db_batch = "/Training Batch 2"

        segmentation_path = self.db_path + db_batch + "/segmentation-" +\
            subject_name + ".nii"

        return segmentation_path

    def get_train_meta_segmentation_path(self, subject_name):
        """Create train meta segmentation path."""
        """
        Arguments:
            subject_name: name of the subject
        Returns:
            full train meta segmentation path
        """
        meta_segment_path = self.db_path + "/Training Meta Segmentations" +\
            "/meta-segmentation-" + subject_name + ".nii"

        return meta_segment_path

    def get_test_volume_path(self, subject_name, volume_path):
        """Create test segmentation path."""
        """
        Arguments:
            subject_name: name of the subject
        Returns:
            full test volume path
        """
        volume_path = self.db_path + "/Testing Batch" + "/test-volume-" +\
            subject_name + ".nii"

        return volume_path

    def get_test_segmentation_path(self, subject_name):
        """Create test segmentation path."""
        """
        Arguments:
            subject_name: name of the subject
        Returns:
            full test segmentation path
        """
        db_batch = "/Testing Results"
        segmentation_path = self.db_path + db_batch + "/test-segmentation-" +\
            subject_name + ".nii"

        return segmentation_path
