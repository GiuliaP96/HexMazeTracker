# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:52:20 2021


"""

'''
Title: Tracker

Description: A simple offline tracker for detection of rats 
             in the novel Hex-Maze experiment. Serves as a 
             replacement for the manual location scorer

Organistaion: Genzel Lab, Donders Institute    
              Radboud University, Nijmegen

Author(s): Atharva Kand-Giulia Porro
'''

from itertools import groupby
from datetime import date, timedelta, datetime
from pathlib import Path 
from collections import deque
from tools import mask   #, kalman_filter

import cv2
#import seaborn as sns
import matplotlib.pyplot as plt
import math
import time
import logging
import threading
#import argparse
#import os
import numpy as np
import csv

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
BG_SUB = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 150, detectShadows = False)

FONT = cv2.FONT_HERSHEY_TRIPLEX
RT_FPS = 25

MIN_RAT_SIZE = 5

font = cv2.FONT_HERSHEY_PLAIN #cv2.FONT_HERSHEY_TRIPLEX #
colors = np.random.uniform(0, 255, size=(100, 3))

#vid_width,vid_height = 1176, 712
 
#find the shortest distance between two points in space
def points_dist(p1, p2):
    dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return dist

#convert time in milli seconds to -> hh:mm:ss,uuu format
def convert_milli(time):
    sec = (time / 1000) % 60
    minute = (time / (1000*60)) % 60
    hr = (time / (1000*60*60)) % 24

    return f'{int(hr):02d}:{int(minute):02d}:{sec:.3f}'

def load_network(self, n):
    
        self.net = cv2.dnn.readNet('weights/yolov3_training_best.weights', 'tools/yolov3_training.cfg')
        self.classes = []
        with open("tools/classes.txt", "r") as f:
           self.classes = f.read().splitlines()
        print('Network loaded', self.net)
                   
def load_session(self,  n):       
        #experiment meta-data
        self.rat = input("Enter rat number: ")
        self.date = input("Enter date of trial: ")       
        self.goal = input("Enter GOAL node of session: ")  
        
def load_StartNodes(self, num_trials):   
        self.start_nodes = []
        for i in range(int(num_trials)): ##1, sel.num
          node = input('Enter START node of trial num {}: '.format(i+1))
          self.start_nodes.append(int(node))           
                        
def define_variables(self, vp, nl, file_id, n):  
       
        nsp = str(date.today()) + '_' + file_id
        self.save = 'logs/'  + '{}'.format(nsp + '.txt')  #os.path.join(gui.save_path, nsp)
        self.node_list = str(nl)
        self.cap = cv2.VideoCapture(str(vp))
      #  self.paused =False
        self.start=False ##start node
        self.frame = None
        self.record_detections = False
        self.frame_count= None
        self.disp_frame = None
        self.pos_centroid = None
     #   self.kf_coords = None    ##
        self.frame_rate = 0      ##
        self.trial_num = 0        
        self.count_human= 0
        self.count_rat=0
        self.count_head=0        
        self.end_session = False
        
        self.centroid_list = deque(maxlen = 500)         #change maxlen value to chnage how long the pink line is
        self.node_pos = []
        self.path =[] ##all points crossed for heatmap
        self.time_points= [] ##time point for velocity
        self.node_id = []   ##node num
        self.saved_nodes = [] 
        self.saved_velocities=[]
        self.summary_trial=[]
    #    self.KF_age = 0      
        self.hex_mask = mask.create_mask(self.node_list)
       # self.KF = kalman_filter.KF()
        ## video to be saved
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')    #change to MP4V
        #codec = cv2.VideoWriter_fourcc(*'XVID') .avi
        self.save_video = nsp + '.mp4' #'{}'.format('.mp4')
      #  out_name = file_id + '.mp4' ##output video name
        self.vid_fps =int(self.cap.get(cv2.CAP_PROP_FPS))
        self.out = cv2.VideoWriter('videos/{}'.format(self.save_video), self.codec, self.vid_fps, (1176,712))   #.
        #print('File video created')     
        ## save first frame of the video [heatmap]
        success, self.image = self.cap.read()
        if success:
           cv2.imwrite("first_frame.jpg", self.image)  # save frame as JPEG file

class Tracker:
    def __init__(self, vp, nl, file_id):
        '''Tracker class initialisations'''
        
        self.num_trials = input("Enter num total trials: ")        
        threads = list()
        cnn = threading.Thread(target=load_network,args=(self, 1))
        #, args=(1,)
        threads.append(cnn)
        session = threading.Thread(target=load_session,args=(self,1))     #     ,  
        threads.append(session) 
        var = threading.Thread(target=define_variables,args=(self,vp, nl, file_id, 1))   
        threads.append(var)        
      #  start_nodes =  threading.Thread(target=load_StartNodes,args=(self, 1))
       # threads.append(start_nodes)      
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()            
        load_StartNodes(self, self.num_trials)                         
        self.start_nodes_locations = self.find_location(self.node_list, self.start_nodes, self.goal)
        self.run_vid()
        
    #process and display video 
    def run_vid(self):
        '''
        Frame by Frame looping of video
        '''
        print('loading tracker...\n')
        Start = time.time()
        time.sleep(2.0)
        
        with open(self.save, 'a+') as file:
            file.write(f"Rat number: {self.rat} , Date: {self.date} \n")

        while True:         
            ret, self.frame = self.cap.read()

            self.frame_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count=  self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.converted_time = convert_milli(int(self.frame_time))
            
            #process and display frame
            if self.frame is not None:
                self.disp_frame = self.frame.copy()
                self.disp_frame = cv2.resize(self.disp_frame, (1176, 712))                 
                self.preprocessing(self.disp_frame)  #, Rat, tracker,Init,boxes
                self.annotate_frame(self.disp_frame)
                cv2.imshow('Tracker', self.disp_frame)
                self.out.write(self.disp_frame)
                
            if  self.end_session == True:
                print ('End in ', convert_milli(int(Start-time.time())))

            #log present centroid position if program is in 'save mode'
            if self.record_detections:
                if self.pos_centroid is not None:
                    converted_time = convert_milli(int(self.frame_time))
                    if self.saved_nodes:
                        logger.info(f'{converted_time} : The rat position is: {self.pos_centroid} @ {self.saved_nodes[-1]}')
                  
                    else:
                        logger.info(f'{converted_time} : The rat position is: {self.pos_centroid}')
                #if self.endtrial = True
        
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print ('End in ', int(Start-time.time()), Start, time.time())
                print('#Program ended by user')
                break
        
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

    def preprocessing(self, frame): #,Rat, tracker, Init, boxes
        '''
        pre-process frame - apply mask, bg subtractor and morphology operations
        

        Input: Frame (i.e image to be preprocessed)
        
        '''   
        frame  = np.array(frame)      
        self.t1 = time.time()
        ##start of each trial = self.start==True when rat near start node 
        if not self.record_detections:            
           if self.trial_num == int(self.num_trials): 
               print('End session, counted', self.num_trials,self.trial_num)
           else:
               ##calculate coordinate rectangle in start node  
             node =  self.start_nodes_locations[self.trial_num]
             x = int(node[0])
             y= int(node[1]) 
             w= 15 #30
             h=  13 # 20                    
             cv2.rectangle(self.disp_frame, (x-w,y+h), (x+w,y-h),(0, 0,255), 2)
            
             #apply mask on frame from mask.py                                 
             for i in range(0,3):
                 frame[:, :, i] = frame[:, :, i] * self.hex_mask        
            #background subtraction and morphology 
             backsub = BG_SUB.apply(frame)                               
           # cv2.rectangle(frame, ((x-w/2),(y-h/2)), ((x+w/2), (y+h/2)),(0, 250, 0), 2)
             black = np.zeros((backsub.shape[0], backsub.shape[1], 3), np.uint8)
             black_ROI = cv2.rectangle(black,(x,y), (x+w,y+h),(255, 255, 255), -1)   
             gray = cv2.cvtColor(black_ROI,cv2.COLOR_BGR2GRAY)
             ret, mask = cv2.threshold(gray,127,255, 0)
             masked_ROIs = cv2.bitwise_and(backsub,backsub,mask = mask)
             self.find_start(masked_ROIs, node)               
        ##look for rat if found in start position , self.record_detections True if CNN find itnot
        if self.record_detections:   #self.start and            
            self.CNN(frame)       
  
    def find_start(self, frame,node):
        '''
        Function to find start of each trial [rat close to start node]

        '''        
        contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)          #_, cont, _ in new opencv version             
        #find contours greater than the minimum area and 
        #caluclate means of the of all such contours         
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5:            #prev  = 5  5
                  cv2.drawContours(frame, contour, -1, (0,255,0), 3)
                  contour_moments = cv2.moments(contour)           
                  cx = int(contour_moments['m10'] / contour_moments['m00'])
                  cy = int(contour_moments['m01'] / contour_moments['m00'])                  
                  pos_start= (cx,cy)
                  if  5 >  points_dist(pos_start, node) > 3: 
                       print('pos start',pos_start, node, 'distance', points_dist(pos_start, node), 'trial num', self.trial_num)
                       self.saved_nodes = []
                       self.node_pos = []
                       self.centroid_list = []
                       self.trial_num += 1
                       logger.info('Recording Trial {}'.format(self.trial_num)) 
                       self.time_points=[]
                       self.summary_trial=[] 
                       self.saved_nodes = []
                       self.centroid_list.append(pos_start) 
                       self.record_detections = True  
        
    def CNN(self, frame):
        ##input to the CNN
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        self.net.setInput(blob)
        height, width, _ = frame.shape #frame width and width to draw boxes in correct position
        output_layers_names = self.net.getUnconnectedOutLayersNames()    
        layerOutputs = self.net.forward(output_layers_names)   ##get prediction from cnn layers
        
        boxes = []  #boxes coordinate as x,y,height, width
        confidences = [] 
        class_ids = [] #researcher, rat, head       
    
        for output in layerOutputs:
          for detection in output: ##dection in frame check with pretrained darknet
              scores = detection[5:]            
              class_id = np.argmax(scores)
              confidence = scores[class_id]
             
              if class_id == 0 or class_id == 1:                  
                if confidence > 0.8:
                 center_x = int(detection[0]*width)
                 center_y = int(detection[1]*height)
                 w = int(detection[2]*width)
                 h = int(detection[3]*height)
                 x = int(center_x - w/2)
                 y = int(center_y - h/2)
                 boxes.append([x, y, w, h])
                 confidences.append((float(confidence)))
                 class_ids.append(class_id)
      ##apply non-max suppression- eliminate double boxes (boxes, confidences, conf_threshold, nms_threshold)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.2) ##keep boxes with higher confidence
 
    ##go through the detections remeainingafter filtering out the one with confidence < 0.7
        if len(indexes)>0:   ##indices box= box[i], x=box[0],y=box[1],w=[box[2],h=box[3]]
          for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(self.classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]    ##different color for each detected object
            cv2.rectangle(self.disp_frame, (x,y), (x+w, y+h), color, 2) ##bounding box
            cv2.putText(self.disp_frame, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 1)
                       
            if label == 'researcher':
                self.count_human += 1   
                           
            if label == 'rat':
                center_rat =  (x,y) #(int((x + w)/2), int((y + h)/2))
                if center_rat is not None:
                    self.count_rat += 1 
                    self.pos_centroid = center_rat
                   # self.centroid_list.append(self.pos_centroid)                   
                    if len(self.centroid_list) > 2:                                            
                     if points_dist(self.pos_centroid, self.goal_location) < 20:                        
                         cv2.putText(self.disp_frame, "Goal location reached", (30,70), 0, 1, (0,250,0), 2) 
                         print('Rat End trial')
                         if self.trial_num == int(self.num_trials):   
                                  cv2.putText(self.disp_frame, "Session finished", (30,70), 0, 1, (0,250,0), 2)                                  
                                  print('session end with', self.trial_num, '=', self.num_trials)                                                                                                            
                                  self.calculate_velocity(self.time_points)
                                  self.save_to_file(self.save)  
                                  self.end_session = True              
                         self.count_rat =0   
                         self.create_heatmap()
                         self.start =False
                         self.record_detections = False #not self.record_detections                   
                                                             
            if label == 'head':
                 self.pos_centroid= (x,y) 
                 if self.pos_centroid is not None:
                  # self.total_detections += 1                                  
                   self.centroid_list.append(self.pos_centroid)                                                                     
                   self.count_head += 1 
                   point =[x,y]
                   self.path.append(point)
                   if len(self.centroid_list) > 2:
                     if points_dist(self.pos_centroid, self.centroid_list[-2]) > 2:
                      #  self.kf_coords = self.KF.estimate()
                        self.pos_centroid = self.centroid_list[-2]
                     ##Check if rat reached Goal location
                     if points_dist(self.pos_centroid, self.goal_location) < 20:                        
                         cv2.putText(self.disp_frame, "Goal location reached", (30,70), 0, 1, (0,250,0), 2) 
                         print('Head end trial')
                         ##Check if session is finished                                        
                         if self.trial_num == int(self.num_trials):   
                                  cv2.putText(self.disp_frame, "Session finished", (30,70), 0, 1, (0,250,0), 2)                                  
                                  print('sessionend', self.trial_num)                                                                                                            
                                  self.calculate_velocity(self.time_points)
                                  self.save_to_file(self.save)   
                                  self.create_heatmap()   
                                  self.end_session = True
                         self.count_head =0 
                         
                         self.record_detections = not self.record_detections 
                
                                                                       

    def calculate_velocity(self,time_points): #
    ##calculate rat speed between two consecutive nodes  
      bridges = { ('124', '201'):0.60,
           ('121', '302'):1.72,
           ('223', '404'):1.69,
           ('324', '401'):0.60,
           ('305', '220'):0.60}
      if len(time_points) > 3:
            lenght=0
            self.first_node= time_points[0][1]            
            format = '%H:%M:%S.%f' 
            # first_time=((time_points[i][0])/ 1000) % 60 
            
        ##iterate over list of touple with time points and nodes IDs
        ###grab start time and node name and next node         
            for i in range(0, len(time_points)):
              start_node= time_points[i][1]
              start_time= datetime.strptime((time_points[i][0]), format).time()
              j=i+1
              if j == len(time_points):
                self.last_node= time_points[i][1]                
              else:
                end_node= time_points[j][1]
                end_time=datetime.strptime((time_points[j][0]), format).time()
                difference = timedelta(hours= end_time.hour-start_time.hour, minutes= end_time.minute-start_time.minute, seconds=end_time.second-start_time.second, microseconds=end_time.microsecond-start_time.microsecond).total_seconds()
                if (start_node, end_node) in bridges:
                          lenght= bridges[(start_node, end_node)]
                          
                elif(end_node, start_node) in bridges:
                        lenght= bridges[(end_node, start_node)] 
                        
                else:
                          lenght=0.30    ##30cm within islands
                speed= round(lenght/difference, 3)
                self.summary_trial.append([(start_node,end_node),(time_points[i][0],time_points[j][0]),difference,lenght,speed])
                self.saved_velocities.append(speed)

            
    @staticmethod
    def annotate_node(frame, point, node, t):
        '''Annotate traversed nodes on to the frame

        Input: Frame (to be annotated), Point: x, y coords of node, Node: Node name, t 1=start,2=walked,3=goal
        '''

        if t==2:
           cv2.circle(frame, point, 20, color = (0, 69, 255), thickness = 1)
           cv2.putText(frame, str(node), (point[0]+2, point[1]+2), 
                    fontScale=0.5, fontFace=FONT, color = (0, 69, 255), thickness=1,
                    lineType=cv2.LINE_AA) 
        if t==3:
           cv2.circle(frame, point, 20, color = (0, 0, 255), thickness = 1)
           cv2.putText(frame, str(node), (point[0]+2, point[1]+2), 
                    fontScale=0.5, fontFace=FONT, color = (0, 0, 255), thickness=1,
                    lineType=cv2.LINE_AA)
           cv2.putText(frame, 'End', (point[0], point[1]-6), 
                    fontScale=0.5, fontFace=FONT, color = (0, 0, 255), thickness=1,
                    lineType=cv2.LINE_AA)


    def annotate_frame(self, frame):
        '''
        Annotates frame with frame information, path and nodes resgistered

        ''' 
        #dictionary of node names and corresponding coordinates
        nodes_dict = mask.create_node_dict(self.node_list)    
          ##annotate time, fps and goal node of the session          
        cv2.putText(frame, str(self.converted_time), (970,670), 
                        fontFace = FONT, fontScale = 0.75, color = (240,240,240), thickness = 1)         
        fps = 1./(time.time()-self.t1)
        cv2.putText(self.disp_frame, "FPS: {:.2f}".format(fps), (970,650), fontFace = FONT, fontScale = 0.75, color = (240,240,240), thickness = 1)              
        self.annotate_node(frame, point = self.goal_location, node = self.goal, t= 3)
       
        #if the centroid position of rat is within 20 pixels of any node
        #register that node to a list. 
        if self.pos_centroid is not None:
            for node_name in nodes_dict:
                if points_dist(self.pos_centroid, nodes_dict[node_name]) < 20:
                    if self.record_detections: #condition to go into 'save mode'
                        self.saved_nodes.append(node_name)                        
                        self.node_pos.append(nodes_dict[node_name])
                        
                        ###save timepoints for speed calculation - self.calculate_velocity(self.time_points)
                        if len(self.time_points) <= 0:  
                           self.time_points.append([self.converted_time,node_name])
                        if node_name != self.saved_nodes[(len(self.saved_nodes))-2]:
                               self.time_points.append([self.converted_time,node_name])

        #annotate all nodes the rat has traversed
        for i in range(0, len(self.saved_nodes)):
            self.annotate_node(frame, point = self.node_pos[i], node = self.saved_nodes[i], t= 2) ##t=2 walked node during the trial
        #frame annotations during recording
        if self.record_detections:
            
            #savepath  = self.vid_save_path + '{}'.format('.mp4')
            cv2.putText(frame,'Trial:' + str(self.trial_num), (60,60), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
            cv2.putText(frame,'Currently writing to file...', (60,80), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
            
            cv2.putText(frame, "Rat Count: " + str(self.count_rat), (30,130), fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
            cv2.putText(frame, "Rat head Count: " + str(self.count_head), (30,160), fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)

            #draw the path that the rat has traversed
            if len(self.centroid_list) >= 2:
                for i in range(1, len(self.centroid_list)):
                    cv2.line(frame, self.centroid_list[i], self.centroid_list[i - 1], 
                             color = (255, 0, 255), thickness = 1)
            
          #  if self.pos_centroid is not None:
           #     cv2.line(frame, (self.pos_centroid[0] - 5, self.pos_centroid[1]), (self.pos_centroid[0] + 5, self.pos_centroid[1]), 
            #    color = (0, 255, 0), thickness = 2)
             #   cv2.line(frame, (self.pos_centroid[0], self.pos_centroid[1] - 5), (self.pos_centroid[0], self.pos_centroid[1] + 5), 
              #  color = (0, 255, 0), thickness = 2)
                                  
    
    #save recorded nodes to file
    def save_to_file(self, fname):
        savelist = []
        with open(fname, 'a+') as file:
            for k, g in groupby(self.saved_nodes):
                savelist.append(k)         
            file.writelines('%s,' % items for items in savelist)
            file.write('\nSummary Trial {}\nStart-Next Nodes// Time points(s) //Seconds//Lenght(cm)// Velocity(m/s)\n'.format(self.trial_num))            
            for i in range(0, len(self.summary_trial)):
                   line=" ".join(map(str,self.summary_trial[i]))
                   file.write(line + '\n')
            file.write('\n')
        file.close()
        
          
    def find_location(self, node_list, start_nodes, goal): 
        nodes_dict = mask.create_node_dict(self.node_list) 
        start_nodes_locations =[]
       # node_dict= self.create_node_dict(node_list)        
        for node_name in nodes_dict:               
             if node_name == str(goal):  
                 self.goal_location= nodes_dict[node_name]                 
             for i in start_nodes:
                 if node_name == str(i):  
                    start_nodes_locations.append(nodes_dict[node_name])
       # print('Location found, goal',self.goal_location,'start', start_nodes_locations)            
        return start_nodes_locations     

if __name__ == "__main__":
    today  = date.today()
    #parser = argparse.ArgumentParser(description = 'Enter video paths')
    #parser.add_argument('-i', '--vid',  type = str, dest= vid_path , help = 'Enter video path')
    #args = parser.parse_args()
    
    enter = input('Enter unique file name: ')
    file_id = '' if not enter else enter

    print('#\nLite Tracker version: v2.00\n#\n')
    import utils.gui as gui

    #logger intitialisations
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)

    logfile_name = 'logs/log_{}_{}.log'.format(str(today), file_id)

    fh = logging.FileHandler(str(logfile_name))
    formatter = logging.Formatter('%(levelname)s : %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh) 

    node_list = Path('node_list_new.csv').resolve()
    
    #save_path= gui.savedir
    vid_path = gui.vpath
    logger.info('Video Imported: {}'.format(vid_path))
    print('creating log files...')
   # logger.info('File saved in: {}'.format(save_path))
    
    Tracker(vp = vid_path, nl = node_list, file_id = file_id)


    


    
            
        