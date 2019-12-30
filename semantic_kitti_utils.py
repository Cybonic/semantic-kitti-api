
#!/usr/bin/env python3

import os
import numpy as np
import yaml
import math
from matplotlib import pyplot as plt
from auxiliary.laserscan import LaserScan, SemLaserScan
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R
from numpy import linalg as la

import kitti_utils as kitti
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class semantic_kitti_utils:
     """ interface from semantic kitti to Benchamerk kitti"""
     def __init__(self, scan, scan_names, label_names,config,  bbox_path, obj, offset=0,
                 semantics=True):

        self.scan = scan
        self.scan_names = scan_names
        self.label_names = label_names
        self.offset = offset
        self.config = config
        self.total = len(self.scan_names)
        self.semantics = semantics
        self.instances = instances
        self._labels_of_interest_name = yaml.load(open(obj))
        self._labels_of_interest_num = self.get_label_num(self._labels_of_interest_name)
        self.sizepoints = 1
        self._label_path = os.path.join(self._bbox_path,"labels_2") # path where bounding boxes will be stored

    def get_label_num(self,objects):

        config = yaml.load(open(self.config))
        convlabel = config['labels']
        objectlabels = list(objects.values())
        key = []
        for value in  objectlabels[0]:
            key.append(self.getKeysByValue(convlabel, value))
        return key

    def CreateAll3DBoundingBoxes(self):

        self.scan.reset()
        self.scan.open_scan(self.scan_names[self.offset])
        self.scan.open_label(self.label_names[self.offset])
        self.scan_labels = self.scan.sem_label
        self.scan_pts = self.scan.points

        # MergeColorFlag=1 - Merge all colors of all objects in same color frame
        # MergeColorFlag=0 - Create a color frame for each object
        bboxes = self.Create3DBoundingBoxes(self.scan_pts,self.scan_labels,MergeColorFlag=1)
        
        file_name = str(self.offset) + ".txt"
       
        self.Save3DBoundingBox(bboxes,file_name)

        # Plot objects & bounding boxes
        #shape = len(list(scan_color_all_obj))
        shape = len(scan_color_all_obj.shape)

    def save_3D_boundingboxes(self,bboxes,filename):

        label_path = self._label_path

        if os.path.isdir(label_path):
            true_label_path = label_path
        else:
            try:
                access_rights = 0x755
                os.makedirs(label_path,access_rights)
                true_label_path = label_path
            except OSError:
                print ("Creation of the directory %s failed" % label_path)
            else:
                print ("Successfully created the directory %s " % label_path)

        file_path = os.path.join(true_label_path,filename) # full file path

        for bbox in bboxes.items():

            kitti_labels = self.conv_to_kitti_format(bbox)
            kitti.write_label(kitti_labels,file_path)

    def Create3DBoundingBoxes(self,ScanPts,ScanLabels):

        # Check if objects of interest exist in the labels
        # ...
        # Do it here
        # ...

        # A segment may contain more than one object from the same class
        object_segment_idx = self.SplitIntoObjectSegments(ScanLabels)
        boundingboxes = self.ObjectClassClustering(ScanPts,object_segment_idx,MergeColorFlag)
        
        return(boundingboxes)

    def SplitIntoObjectSegments(self,ScanLabels):
        
        labels_of_interest_num = self._labels_of_interest_num
        labels_of_interest_name = self._labels_of_interest_name
        scan_semantic_labels = ScanLabels

        pointsLabel= dict()
       
        for label_num in labels_of_interest_num:
            pts_idx = np.where(scan_semantic_labels==label_num)[0]
            if pts_idx.size > 0:
                index = labels_of_interest_num.index(label_num)
                class_name= list(labels_of_interest_name.values())[0][index]
                pointsLabel.update({class_name: pts_idx})

        return pointsLabel

    def ObjectClassClustering(self,ScanPts,SegmentIdx):
        
        # Create 
        objects = dict()
        label_color_scan = []
        for name,segment in SegmentIdx.items():
        
            bbox,single_class_objects = self.SegmentClustering(ScanPts,segment)
            
            objects.update({name:bbox})
            
        return(objects)

    def SegmentClustering(self,ScanPts,SegmentIdx):
        # Split segment (with multiple objects of the same class) 
        # in different instances (point clouds)
        #
        # Input: Segment with same objects
        # Output: dict with all instances (objects) 

        
        # Clustering
        segment_points = ScanPts[SegmentIdx]
        db  = DBSCAN(eps=0.3, min_samples=5).fit(segment_points)
        
        # go through all clusters
        num_instances    = int(max(db.labels_))
        candidates       = np.array(db.labels_)
        instance_boundle = []
        bbox_list        = []
        for inst_label in  range(0,num_instances):

            inst_index = np.where(candidates == inst_label)
            instance_index  = SegmentIdx[inst_index]
            instance_points = segment_points[inst_index]

            bb = self.Campute3DBoundingBox(instance_points)
            bbox_list.append(bb)
            instance = dict([('bb', bb),('idx',instance_index)])
            instance_boundle.append(instance)

        return(bbox_list,instance_boundle)

    def Campute3DBoundingBox(self,points):

       
        return(obj)


class bounding_boxes:

    def __init__(self,otype,width,height,length,t,rz,score):
        self._width = width
        self._height = height
        self._length = length
        self._t = t   # 3x1 vector
        self._rz = rz # radians 
        self._type  = otype # object name
        self._score = score
        self.kitti_label_frame = kitti.Object3d()

    def compute_3D_boundingbox(self,points,score,otype):

         # Transform to object referencial 
        
        # translation
        t = np.mean(points,axis=0).reshape(3,1) # compute object's mass center 
        t[2] = 0 # object frame is mass center in the ground surface (as described kitti paper)
        pts = points.T - t

        # rotation 
        # ....
        rz = 0

        # Get 3D bounding box bounderies 
        x = pts[0,:]
        y = pts[1,:]
        z = pts[2,:]

        xminbound = float(min(x))
        xmaxbound = float(max(x))
        yminbound = float(min(y))
        ymaxbound = float(max(y))
        zminbound = float(min(z))
        zmaxbound = float(max(z))

        # height = np.abs(zmaxbound - zminbound) # height is z axis
        # Since object frame is located at the ground the heigth of the object 
        # is considered to be z
        height = np.abs(zmaxbound) # height is z axis 
        length = np.abs(xmaxbound - xminbound) # length
        width = np.abs(ymaxbound - yminbound)  # width
    
        #Rot = r.as_matrix()

        self._width = width
        self._height = height
        self._length = length
        self._t = t
        self._rz = rz
        self._type  = otype
        self._score = score

    def conv_to_kitti_format(self,bbox):

        w = self._width
        h = self._height
        l = self._length
        rz = self._rz
        t = self._t
        score = self._score
        kitti_label_frame.loadBox3D(objtype,h,w,l,t,rz,score)
        return(kitti_label_frame)

    


def getKeysByValue(dictOfElements, valueToFind):

        listOfKeys = []
        listOfItems = dictOfElements.items()
        for item  in listOfItems:
            if item[1] == valueToFind:
                listOfKeys = item[0]
        return  listOfKeys