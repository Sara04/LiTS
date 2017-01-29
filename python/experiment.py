"""Experiment LiTS."""
import os
import sys
import argparse
from database import LiTSdb
from scan import LiTSscan
from preprocessor import LiTSpreprocessor
from rough_detector import LiTSroughDetector
from tumor_detector import LiTStumorDetector
import time


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
    training, validation, testing = db.get_data_split(subjects, 1)
    # _______________________________________________________________________ #

    # Data loading, preprocessing and liver bounding box detection
    # _______________________________________________________________________ #
    preprocess = LiTSpreprocessor()
    rough_detector = LiTSroughDetector()
    scans = {}
    time_start = time.time()
    for s in subjects[0:]:
        print("s:", s)
        if s not in scans:
            scans[s] = LiTSscan(s)

        # Loading CT volume, segmentation labels and voxel size,
        # volume intensity preprocessing and patient orientation
        # ___________________________________________________________________ #
        scans[s].load_volume(db.get_volume_path(s))
        scans[s].load_segmentation(db.get_segmentation_path(s))
        scans[s].load_info(db.get_volume_path(s))
        preprocess.preprocess(scans[s], s)
        rough_detector.detect_liver_bbox(scans[s])

    time_end = time.time()
    print("time elapsed:", time_end - time_start)
    # _______________________________________________________________________ #

    # Segmentation training and testing
    # _______________________________________________________________________ #
    tumor_detector = LiTStumorDetector()

    tumor_detector.train(scans, training, validation)
    tumor_detector.test(scans, testing)

if __name__ == '__main__':
    main()
