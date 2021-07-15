      # -*- coding: utf-8 -*-
'''
Title: Tracker

Description: A simple offline tracker for detection of rats 
             in the novel Hex-Maze experiment. Serves as a 
             replacement for the manual location scorer

Organistaion: Genzel Lab, Donders Institute    
              Radboud University, Nijmegen

Author(s): Atharva Kand-Giulia Porro

Notes: If run outside Colab uncomment last lines of run_vid (e.g. cv2.destroyAllWindows() and key = cv2.waitKey(1) & 0xFF)
'''

from itertools import groupby
from datetime import date, timedelta, datetime
from pathlib import Path 
from collections import deque
from tools import mask
import cv2
#import matplotlib.pyplot as plt
import math
import time
import logging
import threading
import numpy as np

FONT = cv2.FONT_HERSHEY_TRIPLEX
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

class Tracker:
    def __init__(self, vp, nl, out):
        '''Tracker class initialisations'''                      
        ##set of threads to load network, input and variables
        threads = list()
        ##thread to load network
        cnn = threading.Thread(target=self.load_network,args=(1,))
        threads.append(cnn)
        #thread to load session infos, date, rat number, goal and start locations, variables and create video and .txt saving path
        session = threading.Thread(target=self.load_session,args=(vp, nl, 1, out))      
        threads.append(session)                  
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join() 
        print('\n -Network loaded- ', self.net)            
        #find location of goal node and all start nodes
        self.start_nodes_locations = self.find_location(nl, self.start_nodes, self.goal)
        print('\n  ________  SUMMARY SESSION  ________  ')
        print('\nPath video file:' , self.save_video)
        print('\nPath .log and .txt files:' , self.save)
        print('\nTotal trials current session:', self.num_trials, '\n\nGoal location node ', self.goal)   
        for i in range(0, len(self.start_nodes)):
          print('\nStart node trial {} '.format(i+1), self.start_nodes[i], 'location', self.start_nodes_locations[i])    
        
        #logger intitialisations
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)

        logfile_name = '{}/logs/log_{}_{}.log'.format(out, str(self.date), 'Rat'+self.rat)

        fh = logging.FileHandler(str(logfile_name))
        formatter = logging.Formatter('%(levelname)s : %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh) 
        self.logger.info('Video Imported: {}'.format(vp))
        print('\ncreating log files...')
        
        self.run_vid()
   
    def load_network(self, n):
       # load the model of yolov3-  weights files and cnn structure (.cfg config file)  
        self.net = cv2.dnn.readNet('weights/yolov3_training_best.weights', 'tools/yolov3_training.cfg')
        #set the backend target to a CUDA-enabled GPU
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  ##Colab or GPU only
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        ## load object classes - researcher, rat, head
        self.classes = []
        with open("tools/classes.txt", "r") as f:
           self.classes = f.read().splitlines()     
    
    def load_session(self, vp, nl, n, out):       
        #experiment meta-data
        self.rat = input("\n>> Enter rat number: ")
        self.date = input("\n>> Enter date of trial: ")
        self.num_trials = input("\n>> Enter num total trials: ")  
        self.goal = input("\n>> Enter session GOAL node (num): ") 
        self.trial_type = input("\n>> Enter first trial type [1]-Normal [2]-New GoaL Location [3]-Probe [4]-Special(Ephys): ") 
        ##session start goals
        self.start_nodes = []
        for i in range(int(self.num_trials)): ##1, sel.num
          node = input('\n> Enter START node(num) of trial {}: '.format(i+1))
          self.start_nodes.append(int(node))  
        self.node_list = str(nl)
        self.center_researcher = None
        self.cap = cv2.VideoCapture(str(vp))
        self.start = True ##check start node if researcher is present before trial start
        self.end_session = False ##check last goal location reached
        self.frame = None
        self.record_detections = False ##True to save nodes
        self.frame_count = None
        self.special_start = False ##Ephys training to start timer 
        self.minutes = None
        self.disp_frame = None
        self.pos_centroid = None #keep centroids rat 
        self.Rat = None ##keep centroid of rat 
        self.frame_rate = 0      ##
        self.trial_num = 0        
        self.count_rat=0
        self.count_head=0   
        self.start_time = 0 #timer
        self.probe = False ##True if trial type = 3 and first trial
        self.NGL = False
        #change maxlen value to chnage how long the path line is
        self.centroid_list = deque(maxlen = 500)       
        self.node_pos = []
        self.time_points= [] ##time point for velocity
        self.node_id = []   ##node num
        self.saved_nodes = [] 
        self.saved_velocities=[]
        self.summary_trial=[]    
        
        self.save = '{}/logs/{}_{}'.format(out, str(self.date), 'Rat'+self.rat +'.txt')  # str(date.today())
        ##set output video saved in folder video/'date_unique file name'.mp4
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')   
        #self.codec = cv2.VideoWriter_fourcc(*'XVID')    #to change format video in .avi
        self.save_video =  '{}/videos/{}_{}.mp4'.format(out, str(self.date), 'Rat' + self.rat)   #or .avi
        self.vid_fps =int(self.cap.get(cv2.CAP_PROP_FPS))
        self.out = cv2.VideoWriter('{}'.format(self.save_video), self.codec, self.vid_fps, (1176,712))         
        
          
    #process and display video 
    def run_vid(self):
        '''
        Frame by Frame looping of video
        '''
        print('\nStarting video.....\n')
        with open(self.save, 'a+') as file:
            file.write(f"Rat number: {self.rat} , Date: {self.date} \n")
  
        while True:         
           ret, self.frame = self.cap.read()
           if self.end_session == False:
             self.frame_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
             self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
             self.frame_count=  self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
             self.converted_time = convert_milli(int(self.frame_time))

             #process and display frame
             if self.frame is not None:
                self.disp_frame = self.frame.copy()
                self.disp_frame = cv2.resize(self.disp_frame, (1176, 712))                 
                self.CNN(self.disp_frame)  #, Rat, tracker,Init,boxes
                self.annotate_frame(self.disp_frame)
               # cv2.imshow('Tracker', self.disp_frame)
                self.out.write(self.disp_frame)   
                              
             #log present centroid position if program is in 'save mode'
             if self.record_detections:
                if self.pos_centroid is not None:
                    if self.saved_nodes:
                        self.logger.info(f'{self.converted_time} : The rat position is: {self.pos_centroid} @ {self.saved_nodes[-1]}')                  
                    else:
                        self.logger.info(f'{self.converted_time} : The rat position is: {self.pos_centroid}') #pos_centroid
 
           #close video output and print time tracking
           if self.end_session == True:
                print( '\n' , self.converted_time, '\n >>>> Session ended with ', self.trial_num ,' trials out of', self.num_trials)           
                if not ret:
                    self.cap.release()
                    self.out.release()
                    break               
           # key = cv2.waitKey(1) & 0xFF
           # if key == ord('q'):
           #    print('Session ended with ', self.trial_num ,' trials')
           #    print('#Program ended by user')
           #   break                 
      #  cv2.destroyAllWindows()  ##Uncomment if not in cv2 ver 4.5.2

    def find_start(self, center_rat):
        '''
        Function to find start of each trial [rat at least 40 pixels from center of start node]

        '''             
        ##calculate coordinate rectangle in start node 
        print( '\n' , self.converted_time, '\n >>> Waiting Start Next Trial: ', self.trial_num +1 , ' Start node:', self.start_nodes[self.trial_num])       
        print('Rat position', self.pos_centroid, 'Node', self.start_nodes_locations[self.trial_num])
        node =  self.start_nodes_locations[self.trial_num]
        x = int(node[0])
        y= int(node[1]) 
        w= 15 
        h=  13                  
        cv2.rectangle(self.disp_frame, (x-w,y+h), (x+w,y-h),(0,255,0), 2) 
        if points_dist(center_rat, node) < 25: 
         # if self.center_researcher is not None: 
             #if points_dist(node, self.center_researcher) > 40:
                       self.trial_num += 1
                       print('\n >>> New Trial Start: ', self.trial_num, '\nLocation start rat',center_rat,'node', node, 'distance at start', round(points_dist(center_rat, node)))                      
                       self.logger.info('Recording Trial {}'.format(self.trial_num))                         
                       ##Handle first trial Ephys, probe and NGL special trials types - start time to run the timer
                       if self.trial_num == 1 and int(self.trial_type) != 1:
                            self.start_time = (self.frame_time/ (1000*60)) % 60
                            if int(self.trial_type) == 3:
                                self.probe = True 
                            if int(self.trial_type) == 2 or int(self.trial_type) == 4 :
                                self.NGL = True
                       if self.special_start == True:
                           self.start_time = (self.frame_time/ (1000*60)) % 60                           
                           self.NGL = True
                           self.special_start = False                   
                       self.node_pos = []
                       self.centroid_list = []
                       self.time_points=[]
                       self.summary_trial=[] 
                       self.saved_nodes = []        
                       self.node_id = []   ##node num
                       self.saved_velocities=[]
                       self.pos_centroid = node
                       self.centroid_list.append(self.pos_centroid) 
                       self.check = False  
                       self.record_detections = True  
                       self.start = False #make sure proximitycheck set to false for next start node
        
    def CNN(self, frame):
        self.t1 = time.time()     
        ##input to the CNN - blob.shape: (1, 3, 416, 416) 
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        self.net.setInput(blob)
        #frame width and width to draw boxes in correct position
        height, width, _ = frame.shape           
        output_layers_names = self.net.getUnconnectedOutLayersNames()  #get names of layers that gives output prediction 
        layerOutputs = self.net.forward(output_layers_names)   ##get prediction from cnn layers
        
        boxes = []  #boxes coordinate as x,y,height, width
        confidences = [] #set to 0.7
        class_ids = [] #researcher, rat, head 
        centroids = []
        self.Rat=None
    
        for output in layerOutputs:
          for detection in output: ##dection in frame check with pretrained darknet
              scores = detection[5:]            
              class_id = np.argmax(scores)
              confidence = scores[class_id]
              # filter out weak detections by ensuring the predicted
              # probability is greater than a minimum threshold
              if confidence > 0.7:
                 center_x = int(detection[0]*width)
                 center_y = int(detection[1]*height)
                 w = int(detection[2]*width)
                 h = int(detection[3]*height)
                 x = int(center_x - w/2)
                 y = int(center_y - h/2)
                 centroids.append((center_x,center_y))
                 boxes.append([x, y, w, h])
                 confidences.append((float(confidence)))
                 class_ids.append(class_id)
         ##apply non-max suppression- eliminate double boxes 
         ##(boxes, confidences, conf_threshold, nms_threshold)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.70, 0.3) ##keep boxes with higher confidence
        ##go through the detections after filtering out the one with confidence < 0.7
        if len(indexes)>0:   ##indices box= box[i], x=box[0],y=box[1],w=[box[2],h=box[3]]
          for i in indexes.flatten(): ##Return a copy of the array collapsed into one dimension
            x, y, w, h = boxes[i]       
            label = str(self.classes[class_ids[i]]) 
            confidence = str(round(confidences[i],2))
            color = colors[i]    ##different color for each detected object
            cv2.rectangle(self.disp_frame, (x,y), (x+w, y+h), color, 2) ##bounding box and label object
            cv2.putText(self.disp_frame, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 1)                              

            ##Check rat-researcher  proximity only when the training trial is not running
            if label == 'researcher':
              #not check for start first trial - self.start=True
                if not self.record_detections and not self.start:
                    self.center_researcher = centroids[i]
                    print('\n Checking proximity...')
                    if self.center_researcher is not None and self.Rat is not None:
                            if points_dist(self.Rat , self.center_researcher) <= 600:                          
                                self.start = True 
                                print('\n\n >>> Proximity Checked > start new trial')   
 
            ##Get box centroid if label object is head - main object to detect, if None take centroid rat body                                                                                             
            if label == 'head':
                 self.Rat = centroids[i]                 
                 if self.Rat is not None:                   
                  ##Check researcher proximity before start new trial [avoid start if rat walked in start node soon after it reached goal location]
                   ##If start of trial wait until rat is placed in new start node
                   if self.start == True:
                       self.find_start(self.Rat)
                    ##condition to save nodes
                   if self.record_detections:
                       self.count_head += 1 
                       self.object_detection(rat = self.Rat, frame = frame)
  
         ##Get box centroid if label object is rat [body + tail] if nohead etected
            if label == 'rat':
                if self.Rat is None: #get centroid of rat body only if rat head is not detected
                    self.Rat = centroids[i] # center of bounding box (x,y) 
                    if self.Rat is not None:  
                        if self.start == True:
                          self.find_start(self.Rat)       
                        if self.record_detections:
                            self.count_rat += 1
                            self.object_detection(rat = self.Rat, frame = frame)
                                                                                                                          
    def object_detection(self, rat, frame): 
       if self.pos_centroid is not None:
          if points_dist(self.pos_centroid, rat) < 70:
             self.pos_centroid = rat        
          else:
             self.pos_centroid = self.pos_centroid
       else:
          self.pos_centroid = rat    
       self.centroid_list.append(self.pos_centroid)       
                    
       ##New Goal location trial: first trial 10 minutes long
       if self.NGL:
           self.minutes = self.timer(start = self.start_time)
           if int(self.minutes)  >= 10:
               cv2.putText(self.disp_frame, "End NGL trial",(60,100), fontFace = FONT,
                           fontScale = 0.75, color = (0,255,0), thickness = 1) 
               print('n\n\n >>> End New Goal Location Trial - timeout', self.trial_num, ' out of ', self.num_trials)
               self.NGL = False
               self.end_trial(frame)                          
               ##Check if session is finished
               if self.trial_num == int(self.num_trials):
                   print('\n >>>>>>  Session ends with', self.trial_num, ' trials')
                   self.end_session = True  
               if self.trial_type == '4':               
                   self.start_time = (self.frame_time/ (1000*60)) % 60
                                      
       ##Probe trial: look for goal locatin reached after first 2 minutes
       if self.probe:
           self.minutes = self.timer(start = self.start_time)
           if int(self.minutes) >= 2: 
               if points_dist(self.pos_centroid, self.goal_location) <= 25:  
                   cv2.putText(self.disp_frame, "End Probe Trial", (60,100), fontFace = FONT,
                               fontScale = 0.75, color = (0,255,0), thickness = 1) 
                   print('\n\n >>> End Probe trial', self.trial_num, ' out of ', self.num_trials, '\nCount rat', self.count_rat, ' head', self.count_head)                   
                   self.probe = False
                   self.end_trial(frame) 
                   ##Check if session is finished
                   if self.trial_num == int(self.num_trials):
                       print('\n >>>>>>  Session ends with', self.trial_num, ' trials')
                       self.end_session = True                            

        ##Normal training - Check if rat reached Goal location  
       if not self.probe and not self.NGL:                     
           if points_dist(self.pos_centroid, self.goal_location) <= 20:
               cv2.putText(self.disp_frame, "Goal location reached", (60,100), fontFace = FONT,
                                fontScale = 0.75, color = (0,255,0), thickness = 1)
               print('\n\n >>> Goal location reached. End of trial ', self.trial_num, ' out of ', self.num_trials, '\nCount rat', self.count_rat, ' head', self.count_head)
               self.end_trial(frame)
               ##Check if session is finished
               if self.trial_num == int(self.num_trials):
                   print('\n >>>>>>  Session ends with', self.trial_num, ' trials')
                   self.end_session = True 
               ##Ephys training - normal trials end after 15 minutes    
               if self.trial_type== '4':
                  self.minutes = self.timer(start = self.start_time)                     
                  if int(self.minutes)  >= 15:
                      cv2.putText(self.disp_frame, "End 15 minutes normal training",(60,100), fontFace = FONT,
                           fontScale = 0.75, color = (0,255,0), thickness = 1) 
                      print('n\n\n >>> End Normal training trial - 15 minutes passed', self.trial_num, ' out of ', self.num_trials)
                      self.special_start = True ##start a new timer for next NGL trial
                      
  
    
    def end_trial(self, frame): 
      #make sure last node is saved and written to file before self.record_detections = False
      self.pos_centroid = self.goal_location
      self.centroid_list.append(self.pos_centroid) 
      self.annotate_frame(frame) 
      ##if rat reached goal node calculate velocities and save to file
      self.count_rat = 0 
      self.count_head = 0
      self.calculate_velocity(self.time_points)
      self.save_to_file(self.save)
      self.start = False                             
      self.record_detections = False       
    
    #Timer for new goal location and probe trials
    def timer(self, start):
        end = (self.frame_time/ (1000*60)) % 60
        duration = end - start       
        if duration < 0:
            duration = duration + 60
        print('Timer:' , round(duration, 2),  'minutes')     
        return int(duration)      
     
    def calculate_velocity(self,time_points): #
    ##calculate rat speed between two consecutive nodes  
      bridges = { ('124', '201'):0.60,
           ('121', '302'):1.72,
           ('223', '404'):1.69,
           ('324', '401'):0.60,
           ('305', '220'):0.60}
      if len(time_points) > 2:
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
                try:
                   speed = round(float(lenght)/float(difference), 3)
                except ZeroDivisionError:
                   speed= 0
                finally:
                 self.summary_trial.append([(start_node,end_node),(time_points[i][0],time_points[j][0]),difference,lenght,speed])
                 self.saved_velocities.append(speed)

            
    @staticmethod
    def annotate_node(frame, point, node, t):
        '''Annotate traversed nodes on to the frame

        Input: Frame (to be annotated), Point: x, y coords of node, Node: Node name, t 1=start,2=walked,3=goal
        '''
        if t==1:
           cv2.circle(frame, point, 20, color = (0, 255, 0), thickness = 2)
           cv2.putText(frame, str(node), (point[0]- 16, point[1]), 
                    fontScale=0.5, fontFace=FONT, color = (0, 255, 0), thickness=1,
                    lineType=cv2.LINE_AA)
           cv2.putText(frame, 'Start', (point[0] - 16, point[1]-22), 
                    fontScale=0.5, fontFace=FONT, color = (0, 255, 0), thickness=1,
                    lineType=cv2.LINE_AA)
       
        if t==2:
           cv2.circle(frame, point, 20, color = (20, 110, 245), thickness = 1)
           cv2.putText(frame, str(node), (point[0] -16, point[1]), 
                    fontScale=0.5, fontFace=FONT, color = (0, 69, 255), thickness=1,
                    lineType=cv2.LINE_AA) 
        if t==3:
           cv2.circle(frame, point, 20, color = (0, 0, 250), thickness = 2)
           cv2.putText(frame, str(node), (point[0] -16, point[1]), 
                    fontScale=0.5, fontFace=FONT, color = (0, 0, 255), thickness=1,
                    lineType=cv2.LINE_AA)
           cv2.putText(frame, 'End', (point[0]- 16, point[1]-22), 
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
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (970,650), fontFace = FONT, fontScale = 0.75, color = (240,240,240), thickness = 1)          
        self.annotate_node(frame, point = self.goal_location, node = self.goal, t= 3)        
        
        ##if traker is waiting rat to be in start node position
        if self.start==True:
            cv2.putText(frame,'Next trial:' + str(self.trial_num+1), (60,60), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
            cv2.putText(frame,'Waiting start new trial...', (60,80), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)    
            
            self.annotate_node(frame, point = self.start_nodes_locations[self.trial_num], node = self.start_nodes[self.trial_num] , t= 1)


        #frame annotations during recording
        if self.record_detections:          
        #if the centroid position of rat is within 22 pixels of any node
        #register that node to a list. 
          if self.pos_centroid is not None:
            for node_name in nodes_dict:
                if points_dist(self.pos_centroid, nodes_dict[node_name]) <= 20:                    
                        self.saved_nodes.append(node_name)                        
                        self.node_pos.append(nodes_dict[node_name])
                        print('\nTrial', self.trial_num,  ' Node', node_name,'\nTime', self.converted_time,' FPS', round(fps, 3))
                        ###save timepoints for speed calculation - self.calculate_velocity(self.time_points)
                        if len(self.time_points) == 0:  
                           self.time_points.append([self.converted_time,node_name])
                        
                        if node_name != self.saved_nodes[(len(self.saved_nodes))-2]:
                               self.time_points.append([self.converted_time,node_name])
                                           
            #savepath  = self.vid_save_path + '{}'.format('.mp4')
            cv2.putText(frame,'Trial:' + str(self.trial_num), (60,60), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)
            cv2.putText(frame,'Currently writing to file...', (60,80), 
                        fontFace = FONT, fontScale = 0.75, color = (255,255,255), thickness = 1)           
            cv2.putText(frame, "Rat Count: " + str(self.count_rat), (40,130), 
                        fontFace = FONT, fontScale = 0.65, color = (255,255,255), thickness = 1)
            cv2.putText(frame, "Rat-head Count: " + str(self.count_head), (40,160), 
                        fontFace = FONT, fontScale = 0.65, color = (255,255,255), thickness = 1)
        
    
         #draw the path that the rat has traversed [centroid = head trace]
            if len(self.centroid_list) >= 2:
                for i in range(1, len(self.centroid_list)):
                    cv2.line(frame, self.centroid_list[i], self.centroid_list[i - 1], 
                             color = (255,98, 98), thickness = 1)     

            if self.pos_centroid is not None:
                cv2.line(frame, (self.pos_centroid[0] - 5, self.pos_centroid[1]), (self.pos_centroid[0] + 5, self.pos_centroid[1]), 
                color = (0, 255, 0), thickness = 2)
                cv2.line(frame, (self.pos_centroid[0], self.pos_centroid[1] - 5), (self.pos_centroid[0], self.pos_centroid[1] + 5), 
                color = (0, 255, 0), thickness = 2)
       
        #annotate all nodes the rat has traversed
        for i in range(0, len(self.saved_nodes)):
            self.annotate_node(frame, point = self.node_pos[i], node = self.saved_nodes[i], t= 2) ##t=2 walked node during the trial
 
                                      
    #save recorded nodes to file
    def save_to_file(self, fname):
        savelist = []
        with open(fname, 'a+') as file:
            for k, g in groupby(self.saved_nodes):
                savelist.append(k)         
            file.writelines('%s,' % items for items in savelist)
            print('\nNode crossed: {}'.format(items for items in savelist))  
            file.write('\nSummary Trial {}\nStart-Next Nodes// Time points(s) //Seconds//Lenght(cm)// Velocity(m/s)\n'.format(self.trial_num)) 
            print('\nSummary Trial {}\nStart-Next Nodes// Time points(s) //Seconds//Lenght(cm)// Velocity(m/s)\n'.format(self.trial_num)) 
            for i in range(0, len(self.summary_trial)):
                   line=" ".join(map(str,self.summary_trial[i]))
                   file.write(line + '\n')
                   print(line + '\n')
            file.write('\n')
        file.close()
                  
    def find_location(self, node_list, start_nodes, goal): 
        nodes_dict = mask.create_node_dict(self.node_list) 
        start_nodes_locations =[]
        for node_name in nodes_dict:               
             if node_name == str(goal):  
                 self.goal_location= nodes_dict[node_name]                          
        for node in start_nodes:
          for node_name in nodes_dict: 
               if node_name == str(node):  
                 start_nodes_locations.append(nodes_dict[node_name])
        return start_nodes_locations     

if __name__ == "__main__":
    today  = date.today()
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", dest='vid_path', help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", dest='output', help='full path for saving processed video output')
    args = parser.parse_args()
    if args.vid_path is None: #or args.output is None
        sys.exit("Please provide path to input and output video files! See --help")
    print('\nVideo path' , args.vid_path, 'Logs output', args.output) #, 'save to ', args.output

   # enter = input('\n>> Enter unique file name: ')
   # file_id = '' if not enter else enter
    node_list = Path('/content/TrackerColab/node_list_new.csv').resolve() 
    print('\n\nTracker version: v2.00\n\n')
    
    Tracker(vp = args.vid_path, nl = node_list, out = args.output)       
