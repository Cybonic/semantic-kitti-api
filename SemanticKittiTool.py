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

import vispy
from vispy.scene import visuals, SceneCanvas
from vispy.plot import Fig


class SemanticKittiTool:
    """ Class that creates and handles point cloud data for other application"""

    def __init__(self, scan, scan_names, label_names,config,  bbox_path, obj, offset=0,
                 semantics=True, instances=False):
        self.scan = scan
        self.scan_names = scan_names
        self.label_names = label_names
        self.offset = offset
        self.config = config
        self.total = len(self.scan_names)
        self.semantics = semantics
        self.instances = instances
        self._labels_of_interest_name = yaml.load(open(obj))
        self._labels_of_interest_num = self.GetLabelIdx(self._labels_of_interest_name)
        self.sizepoints = 1
        self._bbox_path = bbox_path # path where bounding boxes will be stored

        self._scan_labels = dict([('inst',[]),('sem',[])])
         # make instance colors
        max_inst_id = 100000
        self.inst_color_lut = np.random.uniform(low=0.0,
                                                high=1.0,
                                                size=(max_inst_id, 3))
        # force zero to a gray-ish color
        self.inst_color_lut[0] = np.full((3), 0.1)

        self.reset()
        
    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0

    def GetLabelIdx(self,objects):

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
        bboxes,scan_color_all_obj = self.Create3DBoundingBoxes(self.scan_pts,self.scan_labels,MergeColorFlag=1)
        
        file_name = str(self.offset) + ".txt"
       
        
        self.Save3DBoundingBox(bboxes,file_name)

        # Plot objects & bounding boxes
        #shape = len(list(scan_color_all_obj))
        shape = len(scan_color_all_obj.shape)

        if shape==1: # Plot all objects with different colors (MergeColorFlag=1 )
            scan_pts,scan_labels = self.Color3DBoundingBox(bboxes,scan_color_all_obj)
            self.colorizeObject(scan_labels)
            self.PlotPcl(scan_pts)

        #else: # Plot all instances of each object class at a time
        #    for scan_color in scan_color_all_obj:
        #        scan_pts,scan_labels = self.Color3DBoundingBox(bboxes,scan_color)
        #        self.colorizeObject(scan_color)
        #        self.PlotPcl(scan_pts)


        #self.colorize(self.scan_labels)
        #self.PlotPcl(self.scan_pts)
            
        #self.SaveBoundingBoxes(boundingboxes)
    def Save3DBoundingBox(self,bboxes,filename):

        label_path = os.path.join(self._bbox_path,"labels_2")

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

        labels = kitti.Object3d.loadParameters()

        labels.loadBox3D()
        kitti.write_label(labels,file_path)

        #for obj in bboxes.items():
        #    f.write("%s: " % obj[0])
        #    for bb in obj[1]:
        #        pts = bb['bb']['vertices']
        #        for l in range(0, pts.shape[0]):
        #            for c in range(0, pts.shape[1]):
        #                f.write("%lf " % pts[l,c])
        #    f.write("\n")
        #f.close()

    def Color3DBoundingBox(self,bboxes,colorframe = []):

        #scan_labels = self.scan_labels
        scan_pts = self.scan_pts
        scan_labels = colorframe
        bb_pts = np.array([])
        for classname,data in bboxes.items():
            for idx in range(0,len(data)):
                vertices = np.asarray(data[idx]['bb']['vertices'])
                #vertices = bb
                #vertices = np.array(list())
                if(len(bb_pts[:]) == 0):
                    bb_pts = vertices
                else:
                    bb_pts = np.concatenate((bb_pts,vertices))
    
        value = max(scan_labels)+1
        redcolor = np.ones(bb_pts.shape[0],dtype=int)*value
        
        scan_labels = np.concatenate((scan_labels,redcolor))

        sizepoints =  np.ones(self.scan_pts.shape[0],dtype=int)

        scan_pts = np.concatenate((scan_pts,bb_pts))
        
        sizebb = np.ones(bb_pts.shape[0],dtype=int)*4

        self.sizepoints = np.concatenate((sizepoints,sizebb))
        return(scan_pts,scan_labels)

    def Create3DBoundingBoxes(self,ScanPts,ScanLabels,MergeColorFlag=0):

        # Check if objects of interest exist in the labels
        # ...
        # Do it here
        # ...

        # A segment may contain more than one object from the same class
        object_segment_idx = self.SplitIntoObjectSegments(ScanLabels)
        boundingboxes,label_color_scan = self.ObjectClassClustering(ScanPts,object_segment_idx,MergeColorFlag)
        
        return(boundingboxes,label_color_scan)
       

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
    
    def ObjectClassClustering(self,ScanPts,SegmentIdx,MergeColorFlag = 0):
        
        # Create 
        objects = dict()
        label_color_scan = []
        for name,segment in SegmentIdx.items():
        
            if(MergeColorFlag):
                single_class_objects,segment_scan_color = self.SegmentClustering(ScanPts,
                                                                                 segment,
                                                                                 label_color_scan)
                label_color_scan = segment_scan_color
            else:
                single_class_objects,segment_scan_color = self.SegmentClustering(ScanPts,
                                                                                 segment,
                                                                                 [])
                label_color_scan.append(segment_scan_color)
            
            objects.update({name:single_class_objects})
            
        return(objects,label_color_scan)

    def SegmentClustering(self,ScanPts,SegmentIdx,ColorScan = []):
        # Split segment (with multiple objects of the same class) 
        # in different instances (point clouds)
        #
        # Input: Segment with same objects
        # Output: dict with all instances (objects) 


        if len(ColorScan)>0: 
            max_value = int(max(ColorScan))+1
            segment_scan_color = ColorScan
        else:
            max_value = 1
            segment_scan_color = np.zeros(ScanPts.shape[0],dtype=int)
        
        # Clustering
        segment_points = ScanPts[SegmentIdx]
        db  = DBSCAN(eps=0.3, min_samples=5).fit(segment_points)
        
        # go through all clusters
        num_instances    = int(max(db.labels_))
        candidates       = np.array(db.labels_)
        instance_boundle = []

        for inst_label in  range(0,num_instances):

            inst_index = np.where(candidates == inst_label)
            instance_index  = SegmentIdx[inst_index]
            instance_points = segment_points[inst_index]
            segment_scan_color[instance_index] = int(max_value + inst_label)

            bb = self.Campute3DBoundingBox(instance_points)

            instance = dict([('bb', bb),('idx',instance_index)])
            instance_boundle.append(instance)

        return(instance_boundle,segment_scan_color)

    

    def getKeysByValue(self,dictOfElements, valueToFind):

        listOfKeys = []
        listOfItems = dictOfElements.items()
        for item  in listOfItems:
            if item[1] == valueToFind:
                listOfKeys = item[0]
        return  listOfKeys


    def Campute3DBoundingBoxFitting(self,points):

        
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')                                                                            
        #plt.ion()
      
        
        #ax.scatter(points.T[0,:], points.T[1,:], points.T[2,:],s=10, c='g', marker='o') 
        #plt.show()

        t = np.mean(points,axis=0).reshape(3,1)
        pts = points.T - t
        #ax.scatter(pts[0,:], pts[1,:], pts[2,:],s=20, c='b', marker='o') 
        covariance = np.dot(pts,pts.T)
        (eigvals,eigvecs) = la.eig(covariance)
        #Rot = eigvecs * 
        #print(eigvals)
        maxval=np.argmax(np.absolute(eigvals[0:1]))

        xxidx = maxval
        if xxidx == 0:
             yyidx = 1
        else: 
             yyidx = 0

        Rot = np.array((eigvecs[:,xxidx],eigvecs[:,yyidx],eigvecs[:,2]))

        
        #print("max indice %d\n",maxval)
        #print(Rot)

        angle = self.rotationMatrixToEulerAngles(Rot)
        yaw = math.degrees((angle[2]))

        print(yaw)
        r = R.from_euler('z', -yaw, degrees=True)
        Rot = r.as_matrix()
        #print(Rot)
        #print(math.degrees(angle[0]))
        #print(math.degrees(angle[1]))
        #print(math.degrees(angle[2]))
        #plt(pts[1,:],pts[2,:],pts[3,:])

        #pts = np.dot(Rot,pts)
        
        #ax.scatter(pts[0,:], pts[1,:], pts[2,:],s=20, c='r', marker='x')

        # Bounding Box
        x = pts[0,:]
        y = pts[1,:]
        z = pts[2,:]

        #xminbound = float(min(x)) - 0.05
        #xmaxbound = float(max(x)) + 0.05
        #yminbound = float(min(y)) - 0.05
        #ymaxbound = float(max(y)) + 0.05
        #zminbound = float(min(z)) - 0.05
        #zmaxbound = float(max(z)) + 0.05

        xminbound = float(min(x))
        xmaxbound = float(max(x))
        yminbound = float(min(y))
        ymaxbound = float(max(y))
        zminbound = float(min(z))
        zmaxbound = float(max(z))

        p = list()
        p.append(np.array([xminbound, yminbound, zminbound]))
        p.append(np.array([xmaxbound, yminbound, zminbound]))
        p.append(np.array([xmaxbound, yminbound, zmaxbound]))
        p.append(np.array([xminbound, yminbound, zmaxbound]))
        p.append(np.array([xminbound, ymaxbound, zmaxbound]))
        p.append(np.array([xmaxbound, ymaxbound, zmaxbound]))
        p.append(np.array([xmaxbound, ymaxbound, zminbound]))
        p.append( np.array([xminbound, ymaxbound, zminbound]))

        bbox = np.array([])
        for vert in p:
            #corner = np.dot(Rot.T,vert.reshape(3,1)) + t
            corner = vert.reshape(3,1) + t
            if len(bbox) == 0:
                bbox = corner.T
            else:
            #ax.scatter(corner[0], corner[1], corner[2],s=30, c='g', marker='o')
                bbox = np.concatenate((bbox,corner.T))
        
        #ax.scatter(p0[0], p0[1], p0[2],s=30, c='g', marker='o')
        #ax.scatter(p1[0], p1[1], p1[2],s=30, c='g', marker='o')
        #ax.scatter(p2[0], p2[1], p2[2],s=30, c='g', marker='o')
        #ax.scatter(p3[0], p3[1], p3[2],s=30, c='g', marker='o')
        #ax.scatter(p4[0], p4[1], p4[2],s=30, c='g', marker='o')
        #ax.scatter(p5[0], p5[1], p5[2],s=30, c='g', marker='o')
        #ax.scatter(p6[0], p6[1], p6[2],s=30, c='g', marker='o')
        #ax.scatter(p7[0], p7[1], p7[2],s=30, c='g', marker='o')

        #ax.scatter(points[:,0], points[:,1], points[:,2],s=20, c='b', marker='x')
        #ax.set_xlabel("X")
        #ax.set_ylabel("Y")
        #plt.show()
        

        R_rect = ('R_rect',Rot)
        #input("Press [enter] to continue.")
        orig = ('origin', t)
        # bounding boxe vetices coordinates
        bbox_pts = ('vertices', bbox)
        #dim = ('dim:', {'h': height, 'w': width, 'l': length})

        return(dict([orig,R_rect,bbox_pts]))


    def Campute3DBoundingBox(self,points):

        # Transform to object referencial 
        
        # translation
        t = np.mean(points,axis=0).reshape(3,1) # compute object's mass center 
        t[2] = 0 # object frame is mass center in the ground surface (as described kitti paper)
        pts = points.T - t

        # rotation 
        # ....
        rz = R.from_euler('z', 0, degrees=False) 

        # get 3D bounding box bounderies 
        x = pts[0,:]
        y = pts[1,:]
        z = pts[2,:]

        
        xminbound = float(min(x))
        xmaxbound = float(max(x))
        yminbound = float(min(y))
        ymaxbound = float(max(y))
        zminbound = float(min(z))
        zmaxbound = float(max(z))

        height = np.abs(zmaxbound - zminbound) # height is z axis
        length = np.abs(xmaxbound - xminbound) # length
        width = np.abs(ymaxbound - yminbound)  # width
    
        #Rot = r.as_matrix()
        obj = {
            't': t,
            'rz':rz,
            'h': height,
            'w': width,
            'l': length
        }
        return(obj)

    def CreateTransf(self,R,T):
        row = np.zeros((1,3),dtype=int)

        a = np.concatenate((R,row))
        b = np.concatenate((T,np.array([1]).reshape(1,1)))
        c = np.concatenate((a,b),axis=1)

        return(c)

    def rotationMatrixToEulerAngles(self,R):
 
        #assert(isRotationMatrix(R))
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        
        singular = sy < 1e-6
    
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
    
        return np.array([x, y, z])

    def reset(self):
        """ Reset. """
        # last key press (it should have a mutex, but visualization is not
        # safety critical, so let's do things wrong)
        self.action = "no"  # no, next, back, quit are the possibilities

        # new canvas prepared for visualizing data
        self.canvas = SceneCanvas(keys='interactive', show=True)
        # interface (n next, b back, q quit, very simple)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)
        # grid
        self.grid = self.canvas.central_widget.add_grid()


        #if self.instances:
        print("Using instances in visualizer")
        self.inst_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.canvas.scene)
        self.grid.add_widget(self.inst_view, 0, 0)
        self.inst_vis = visuals.Markers()
        self.inst_view.camera = 'turntable'
        self.inst_view.add(self.inst_vis)
        visuals.XYZAxis(parent=self.inst_view.scene)
        # self.inst_view.camera.link(self.scan_view.camera)

    def resetTwo(self):
        """ Reset. """
        # last key press (it should have a mutex, but visualization is not
        # safety critical, so let's do things wrong)
        self.action = "no"  # no, next, back, quit are the possibilities

        # new canvas prepared for visualizing data
        self.canvas = SceneCanvas(keys='interactive', show=True)
        # interface (n next, b back, q quit, very simple)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)
        # grid
        self.grid = self.canvas.central_widget.add_grid()

        # laserscan part
        self.scan_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.canvas.scene)
        self.grid.add_widget(self.scan_view, 0, 0)
        self.scan_vis = visuals.Markers()
        self.scan_view.camera = 'turntable'
        self.scan_view.add(self.scan_vis)
        visuals.XYZAxis(parent=self.scan_view.scene)

        #if self.instances:
        print("Using instances in visualizer")
        self.inst_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.canvas.scene)
        self.grid.add_widget(self.inst_view, 0, 1)
        self.inst_vis = visuals.Markers()
        self.inst_view.camera = 'turntable'
        self.inst_view.add(self.inst_vis)
        visuals.XYZAxis(parent=self.inst_view.scene)
        # self.inst_view.camera.link(self.scan_view.camera)
    
    def draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()
        
    # interface
    def key_press(self, event):
        self.canvas.events.key_press.block()
        if event.key == 'N':
            self.offset += 1
            if self.offset >= self.total:
                self.offset = 0
            self.CreateAll3DBoundingBoxes()
        elif event.key == 'B':
            self.offset -= 1
            if self.offset < 0:
                self.offset = self.total - 1
            self.CreateAll3DBoundingBoxes()
        elif event.key == 'Q' or event.key == 'Escape':
            self.destroy()

    def PlotPcl(self,points):        
    
        self.inst_vis.set_data(points,
                             face_color=self.inst_label_color[..., ::-1],
                             edge_color=self.inst_label_color[..., ::-1],
                             size=self.sizepoints)

    def PlotPclTwo(self,points):        
    
        # Generate 3D bounding boxes of all Laser scans
        # plot scan
        power = 16
        # print()
        range_data = np.copy(self.scan.unproj_range)
        # print(range_data.max(), range_data.min())
        range_data = range_data**(1 / power)
        # print(range_data.max(), range_data.min())
        viridis_range = ((range_data - range_data.min()) /
                        (range_data.max() - range_data.min()) *
                        255).astype(np.uint8)
        viridis_map = self.get_mpl_colormap("viridis")
        viridis_colors = viridis_map[viridis_range]

        self.scan_vis.set_data(self.scan.points,
                            face_color=viridis_colors[..., ::-1],
                            edge_color=viridis_colors[..., ::-1],
                            size=1)


        self.inst_vis.set_data(points,
                             face_color=self.inst_label_color[..., ::-1],
                             edge_color=self.inst_label_color[..., ::-1],
                             size=self.sizepoints)

    def colorizeObject(self,ScanLabels):
        """ Colorize pointcloud with the color of each semantic label
        """
        #shapescan = self.scan.points.shape
        #sem_label = self.scan_labels['sem']
        #labels_of_interest_num = self._labels_of_interest_num
        #instlabel = self.scan_labels['inst']
        self.inst_label_color = self.inst_color_lut[ScanLabels]
        self.inst_label_color = self.inst_label_color.reshape((-1, 3))
        return

    def colorize(self,ScanLabels):
        """ Colorize pointcloud with the color of each semantic label
        """
        shapescan = self.scan.points.shape
        sem_label = ScanLabels
        labels_of_interest_num = self._labels_of_interest_num
        #instlabel = self.scan_labels['inst']
        new_scan_labels = np.zeros(shapescan[0],dtype=int)

        for labelnum in labels_of_interest_num:
            idx = np.where(sem_label == labelnum)[0]
            new_scan_labels[idx] = labelnum

        self.inst_label_color = self.inst_color_lut[new_scan_labels]
        self.inst_label_color = self.inst_label_color.reshape((-1, 3))

    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        vispy.app.quit()

    def run(self):
        vispy.app.run()