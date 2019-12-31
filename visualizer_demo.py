#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import sys


from SemanticKittiTool import SemanticKittiTool
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.laserscanvis import LaserScanVis


DATASET_PATH = "E:\\DATASETS\\Kitti\\odometry_semantic_dataset"


def getCurrentPath(filename):
        return os.path.dirname(os.path.abspath(__file__)) + '/' + filename

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./visualizer_demo.py")
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
      default= "config/semantic-kitti.yaml",
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
      default="00",
      required=False,
      help='Sequence to visualize. Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_semantics', '-i',
      dest='ignore_semantics',
      default=False,
      action='store_true',
      help='Ignore semantics. Visualizes uncolored pointclouds.'
      'Defaults to %(default)s',
  )
  parser.add_argument(
      '--do_instances', '-di',
      dest='do_instances',
      default=True,
      action='store_true',
      help='Visualize instances too. Defaults to %(default)s',
  )
  parser.add_argument(
      '--offset',
      type=int,
      default=0,
      required=False,
      help='Sequence to start. Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_safety',
      dest='ignore_safety',
      default=False,
      action='store_true',
      help='Normally you want the number of labels and ptcls to be the same,'
      ', but if you are not done inferring this is not the case, so this disables'
      ' that safety.'
      'Defaults to %(default)s',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Dataset", FLAGS.dataset)
  print("Config", FLAGS.config)
  print("Objs", FLAGS.obj)
  print("Sequence", FLAGS.sequence)
  print("ignore_semantics", FLAGS.ignore_semantics)
  print("do_instances", FLAGS.do_instances)
  print("ignore_safety", FLAGS.ignore_safety)
  print("offset", FLAGS.offset)
  print("*" * 80)

  # open config file
  try:
    print("Opening config file %s" % FLAGS.config)
    CFG = yaml.safe_load(open(FLAGS.config, 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # fix sequence name
  FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

  # does sequence folder exist?
  scan_paths = os.path.join(FLAGS.dataset, "sequences",
                            FLAGS.sequence, "velodyne")
  if os.path.isdir(scan_paths):
    print("Sequence folder exists! Using sequence from %s" % scan_paths)
  else:
    print("Sequence folder doesn't exist! Exiting...")
    quit()

  # E:\DATASETS\Kitti\odometry_semantic_dataset\sequences\00\velodyne
  # populate the pointclouds
  scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(scan_paths)) for f in fn]
  scan_names.sort()

  # does sequence folder exist?
  if not FLAGS.ignore_semantics:
   
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
    if not FLAGS.ignore_safety:
      assert(len(label_names) == len(scan_names))

  # check if bbounding box directory exist

  labels_2_path = os.path.join(DATASET_PATH,"sequences",FLAGS.sequence,"labels_2")
 
  if os.path.isdir(labels_2_path):
    print("Labels 2 folder exists! Using labels from %s" % labels_2_path)
  else:
    try:
        os.makedirs(labels_2_path, 0x755)
    except OSError:
        
        print ("Creating directory %s failed" % labels_2_path)
    else:
        print ("Successfully created directory %s " % labels_2_path)



  # create a scan
  if FLAGS.ignore_semantics:
    scan = LaserScan(project=True)  # project all opened scans to spheric proj
  else:
    color_dict = CFG["color_map"]
    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses, color_dict, project=True)

  # create a visualizer
  semantics = not FLAGS.ignore_semantics
  instances = FLAGS.do_instances
  if not semantics:
    label_names = None

  KITTItool = SemanticKittiTool(scan=scan, 
                                scan_names=scan_names,
                                label_names=label_names,
                                offset=FLAGS.offset,
                                config=FLAGS.config,
                                obj = getCurrentPath(FLAGS.obj),
                                semantics=semantics, instances=instances and semantics)
 


  
  #KITTItool.CreateAll3DBoundingBoxes(objectsOfInterrest)
                      
  # print instructions
  print("To navigate:")
  print("\tb: back (previous scan)")
  print("\tn: next (next scan)")
  print("\tq: quit (exit program)")

  KITTItool.run()
 
  #vis.run()