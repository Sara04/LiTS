"""Class for Liver Tumore Segmentation database management."""

import os
import natsort
import numpy as np

# random shuffling of patients' ordinal numbers in case they are ordered
# according to the institutions where the recordings took place

np.random.seed(1)
RANDOM_SHUFFLE = np.arange(0, 130)
np.random.shuffle(RANDOM_SHUFFLE)


class LiTSdb(object):
    """LiTSdb class for data path and protocol split definitions.

    Attributes:
        db_path (str): path to the LiTS database
        db_batches (list): names of the training batches
        split_ratios (list): possible database split ratios into
                             train, valid and test subsets
        no_subjects (int): total number of patients/scans
        split_unit (int): the largest number that divides number of subjects
                          in train, valid and test subsets
        split_factors (list): multiplication factor used to determine the
                              number of subjects in train, valid and test
                              subsets

    Methods:
        _load_split_params: determining split_unit and split_factors based
                            on selected split_ratio
        get_volume_path: gets full path to the volume scan
        get_segmentation_path: gets full path to the segmentation file
        get_subjects_names: gets list of all subjects in the database
        get_data_split: gets the database split into train, valid and test
                        subsets

    """

    db_batches = ['Training Batch 1', 'Training Batch 2']
    split_ratios = ['60_20_20', '40_30_30', '80_10_10']

    def __init__(self, db_path, no_subjects=130, split_ratio='60_20_20'):
        """Initialization method for LiTSdb object.

        Args:
            db_path (str): path to the LiTSdb database
            no_subjects (int): total number of patients/scans, default 130
            split_ratio (str): database split ratios into train, valid and
                               test subsets, default '60_20_20'
        """
        assert split_ratio in self.split_ratios
        self.db_path = db_path
        self.no_subjects = no_subjects
        self.split_unit = None
        self.split_factors = None
        self._load_split_params(split_ratio)

    def _load_split_params(self, split_ratio):
        if split_ratio == '60_20_20':
            self.split_unit = 2 * self.no_subjects / 10
            self.split_factors = [3, 1, 1]
        elif split_ratio == '40_30_30':
            self.split_unit = self.no_subjects / 10
            self.split_factors = [4, 3, 3]
        elif split_ratio == '80_10_10':
            self.split_unit = self.no_subjects / 10
            self.split_factors = [8, 1, 1]

    def get_volume_path(self, s):
        """Return volume path.

        Args:
            s (str): subject/scan ordinal number
        """
        if int(s) > 27:
            sub_db = self.db_batches[1]
        else:
            sub_db = self.db_batches[0]
        return os.path.join(self.db_path,
                            sub_db, '-'.join(['volume', s]) + '.nii')

    def get_segmentation_path(self, s):
        """Return segmentation path.

        Args:
            s (str): subject/scan ordinal number
        """
        if int(s) > 27:
            sub_db = self.db_batches[1]
        else:
            sub_db = self.db_batches[0]
        return os.path.join(self.db_path,
                            sub_db, '-'.join(['segmentation', s]) + '.nii')

    def get_subjects_names(self):
        """Return list of all subjects in the database."""
        subjects = []
        for db_batch in self.db_batches:

            files = os.listdir(os.path.join(self.db_path, db_batch))

            for fi in files:
                if fi.startswith('.') or fi.startswith('volume'):
                    continue
                s = str.split(fi, '-')[1][:-4]
                if s not in subjects:
                    subjects.append(s)

        return natsort.natsorted(subjects)

    def get_data_split(self, subjects, protocol_no=1):
        """Return database split into train, valid and test subsets.

        Args:
            subjects (list): list of all subjects in the database
            protocol_no (int): number that determines at which point
                               in the list of subjects, selections for
                               training, valid and test subsets start
        """
        training = []
        validation = []
        testing = []

        for s_idx in range((protocol_no - 1) * self.split_unit,
                           ((protocol_no - 1) +
                            self.split_factors[0]) * self.split_unit):
            training.append(subjects[RANDOM_SHUFFLE[s_idx % self.no_subjects]])

        for s_idx in range(((protocol_no - 1) + self.split_factors[0]) *
                           self.split_unit,
                           ((protocol_no - 1) + sum(self.split_factors[0:2])) *
                           self.split_unit):
            validation.append(
                subjects[RANDOM_SHUFFLE[s_idx % self.no_subjects]])

        for s_idx in range(((protocol_no - 1) + sum(self.split_factors[0:2])) *
                           self.split_unit,
                           ((protocol_no - 1) + sum(self.split_factors[0:3])) *
                           self.split_unit):
            testing.append(subjects[RANDOM_SHUFFLE[s_idx % self.no_subjects]])

        return training, validation, testing
