"""Experiment LiTS."""
import os
import sys
import argparse
from database import LiTSdb
from scan import LiTSscan
from preprocessor import LiTSpreprocessor
from rough_detector import LiTSroughDetector
from tumor_detector import LiTStumorDetector


def main():
    """Main function for the liver tumor detection training and testing."""
    # Input arguments parsing
    # _______________________________________________________________________ #
    parser = argparse.ArgumentParser(description=''
                                     'Liver tumor segmentation experiment')
    parser.add_argument('-i', dest='db_path', required=True,
                        help='Path to the directory where folders '
                             '"Training Batch 1" and '
                             '"Training Batch 2" are placed')
    args = parser.parse_args()
    if not os.path.exists(args.db_path):
        print("\nInput database path does not exist!\n")
        sys.exit(1)
    if not os.path.exists(os.path.join(args.db_path, 'Training Batch 1')):
        print("\nTraining Batch 1 is not in the input database directory!\n")
        sys.exit(1)
    if not os.path.exists(os.path.join(args.db_path, 'Training Batch 2')):
        print("\nTraining Batch 2 is not in the input database direcory!\n")
    # _______________________________________________________________________ #

    # Database object creation and data split creation.
    # _______________________________________________________________________ #
    db = LiTSdb(args.db_path)
    subjects = db.get_subjects_names()
    # _______________________________________________________________________ #

    # Scan object initialization
    # _______________________________________________________________________ #
    scans = {}
    for s in subjects[0:]:
        if s not in scans:
            scans[s] = LiTSscan(s)
    # _______________________________________________________________________ #

    # Segmentation training, validation and testing
    # _______________________________________________________________________ #
    preprocess = LiTSpreprocessor()
    rough_detector = LiTSroughDetector()
    tumor_detector = LiTStumorDetector()

    for i in range(1, 6):
        training, validation, testing = db.get_data_split(subjects, 1)
        tumor_detector.train(db, scans, preprocess, rough_detector,
                             training, validation)
        tumor_detector.test(db, scans, preprocess, rough_detector,
                            testing)

if __name__ == '__main__':
    main()
