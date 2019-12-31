#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import sys

from SemanticKittiTool import SemanticKittiTool
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.laserscanvis import LaserScanVis


DATASET_PATH =  "E:\\DATASETS\\Kitti\\odometry_semantic_dataset"

                                #"E:\DATASETS\Kitti\odometry_semantic_dataset"
if __name__ == '__main__':
    parser = argparse.ArgumentParser("./semantic_to_kitti_3D_boxes_demo.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=False,
        default=DATASET_PATH,
        help='Dataset to visualize. Defaults to %(default)s',
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default = "config/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--obj', '-o',
        type=str,
        required=False,
        default ="config/objects.yaml",
        help='Dataset objects to detect. Defaults to %(default)s',
    )
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        default="-1",
        required=False,
        help='Sequence to visualize. Defaults to %(default)s',
    )

    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Dataset", FLAGS.dataset)
    print("Config", FLAGS.config)
    print("Objs", FLAGS.obj)
    print("Sequence", FLAGS.sequence)
    print("*" * 100)

    # open config file
    try:
        print("Opening config file %s" % FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    # if FLAGS.sequence == -1 -> go through all sequence
    if (int(FLAGS.sequence) == -1):
        # Get all available sequnces
        sequence_path = os.path.join(DATASET_PATH,'sequences')
        sequance_names  =  [f.name for f in os.scandir( sequence_path ) if f.is_dir()]
    else:
        sequance_names = FLAGS.sequence

    for seq in sequance_names[1:]:

        ## Point cloud
        # does sequence folder exist?
        scan_paths = os.path.join(FLAGS.dataset, "sequences",seq, "velodyne")

        if os.path.isdir(scan_paths):
            print("Sequence folder exists! Using sequence from %s" % scan_paths)
        else:
            print("Sequence folder doesn't exist! Exiting...")
            quit()

        # populate the pointclouds
        scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(scan_paths)) for f in fn]
        scan_names.sort()

        ## Semantic labels
        # does sequence folder exist?
        label_paths = os.path.join(FLAGS.dataset, "sequences",
                                    seq, "labels")

        if os.path.isdir(label_paths):
            print("Labels folder exists! Using labels from %s" % label_paths)
        else:
            print("Labels folder doesn't exist! Exiting...")
            quit()
        # populate the pointclouds
        label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn]
        label_names.sort()

        # check that there are same amount of labels and scans

        assert(len(label_names) == len(scan_names))

        ## Kitti labels (3D bounding boxes )
        # check if bbounding box directory exist
        label_2_paths = os.path.join(FLAGS.dataset,"sequences",seq,'labels_2')
        
        if os.path.isdir(label_2_paths):
            print("Labels 2 folder exists! Using labels from %s" % label_2_paths)
        else:
            try:
                os.makedirs(label_2_paths, 0x755)
            except OSError:
                
                print ("Creating directory %s failed" % label_2_paths)
            else:
                print ("Successfully created directory %s " % label_2_paths)


        ## Indicies of each object class
        
        object_class_path = os.path.join(FLAGS.dataset,"sequences",seq,
                                        'object_classes')
 
        if not os.path.isdir(object_class_path):
            try:
                os.makedirs(object_class_path,0x755)
            except OSError:
                print ("Creating directory %s failed" % object_class_path)
            else:
                print ("Successfully created directory %s " % object_class_path)

        # Load Semantic data

        color_dict = CFG["color_map"]
        nclasses = len(color_dict)
        scan = SemLaserScan(nclasses, color_dict, project=True)

        KITTItool = SemanticKittiTool(scan=scan, 
                                scan_names=scan_names,
                                label_names=label_names,
                                config=FLAGS.config,
                                obj = FLAGS.obj)

        KITTItool.ComputeAll3DBoundingBoxes(label_2_paths)

        KITTItool.SplitObjectClass(object_class_path)