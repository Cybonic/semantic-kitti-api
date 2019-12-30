#!/usr/bin/env python3

import argparse
import os
import yaml
import sys

from SemanticKittiTool import SemanticKittiTool
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.laserscanvis import LaserScanVis

DATASET_PATH =  "E:\\DATASETS\\Kitti\\odometry_semantic_dataset"

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./demo_save_class_idx_to_file.py")
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
      help=' Objects to detect in dataset. Defaults to %(default)s',
  )
  parser.add_argument(
      '--sequence', '-s',
      type=str,
      default="00",
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

  # does sequence folder exist?
  scan_paths = os.path.join(FLAGS.dataset, "sequences",FLAGS.sequence, "velodyne")
  #"E:\DATASETS\Kitti\odometry_semantic_dataset\sequences\00\velodyne"
  if os.path.isdir(scan_paths):
      print("Sequence folder exists! Using sequence from %s" % scan_paths)
  else:
      print("Sequence folder doesn't exist! Exiting...")
      quit()

  # populate the pointclouds
  scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(scan_paths)) for f in fn]
  scan_names.sort()

 # does sequence folder exist?
  label_paths = os.path.join(FLAGS.dataset, "sequences",
                                 FLAGS.sequence, "labels")
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

  # check if bbounding box directory exist

  object_class_path = os.path.join(FLAGS.dataset,"sequences",FLAGS.sequence,
                                'object_classes')
 
  if not os.path.isdir(object_class_path):
    try:
      os.makedirs(object_class_path,0x755)
    except OSError:
      print ("Directory %s failed" % object_class_path)
    else:
      print ("Successfully created directory %s " % object_class_path)

  color_dict = CFG["color_map"]
  nclasses = len(color_dict)
  scan = SemLaserScan(nclasses, color_dict, project=True)

  KITTItool = SemanticKittiTool(scan=scan, 
                                scan_names=scan_names,
                                label_names=label_names,
                                offset=0,
                                config=FLAGS.config,
                                bbox_path = "", # Have to change this to as
                                #input of the function
                                obj = FLAGS.obj)

  dataset_pts = KITTItool.LoadObjectClass(object_class_path)


  dataset_pts

