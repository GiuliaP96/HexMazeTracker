# -*- coding: utf-8 -*-
'''
Title: Tracker

Description: A simple offline tracker for detection of rats 
             in the novel Hex-Maze experiment. Serves as a 
             replacement for the manual location scorer

Organistaion: Genzel Lab, Donders Institute    
              Radboud University, Nijmegen

Author(s): Atharva Kand-Giulia Porro

Note: If run outside Colab uncomment last lines of run_vid (e.g. cv2.destroyAllWindows() and key = cv2.waitKey(1) & 0xFF)
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
                   
def load_session(self,  n):       
        #experiment meta-data
        self.rat = input("Enter rat number: ")
        self.date = input("Enter date of trial: ")       
        self.goal = input("Enter GOAL node of session: ")  
        
def load_StartNodes(self, num_trials):   
        ##session start goals
        self.start_nodes = []
        for i in range(int(num_trials)): ##1, sel.num
          node = input('Enter START node of trial num {}: '.format(i+1))
          self.start_nodes.append(int(node))           
                        
def define_variables(self, vp, nl, file_id, n, out):          
    
        self.save = '{}/logs/{}_{}'.format(out, str(date.today()), file_id +'.txt')
        self.node_list = str(nl)
        self.cap = cv2.VideoCapture(str(vp))

        self.start=True ##check start node
        self.end_session = False ##check last goal location reached
        self.frame = None
        self.record_detections = False
        self.frame_count= None
        self.disp_frame = None
        self.pos_centroid = None
        self.frame_rate = 0      ##
        self.trial_num = 0        
        self.count_human= 0
        self.count_rat=0
        self.count_head=0        
               
        self.centroid_list = deque(maxlen = 500)         #change maxlen value to chnage how long the pink line is
        self.node_pos = []
        self.path =[] ##all points crossed for heatmap
        self.time_points= [] ##time point for velocity
        self.node_id = []   ##node num
        self.saved_nodes = [] 
        self.saved_velocities=[]
        self.summary_trial=[]    

        ##set output video saved in folder video/'date_unique file name'.mp4
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')    #if errors change to MP4V
        #codec = cv2.VideoWriter_fourcc(*'XVID')        #change format video .avi
        self.save_video =  '{}/videos/{}_{}.mp4'.format(out, str(date.today()), file_id)                 # nsp + '.mp4'
        self.vid_fps =int(self.cap.get(cv2.CAP_PROP_FPS))
        self.out = cv2.VideoWriter('{}'.format(self.save_video), self.codec, self.vid_fps, (1176,712))   #.    
        ## save first frame of the video [heatmap]
       # success, self.image = self.cap.read()
        #if success:
         #  cv2.imwrite("first_frame.jpg", self.image)  # save frame as JPEG file

class Tracker:
    def __init__(self, vp, nl, file_id, out):
        '''Tracker class initialisations'''        
        self.num_trials = input("Enter num total trials: ")        
        ##set of threads to load network, input and variables
        threads = list()
        ##thread to load network
        cnn = threading.Thread(target=load_network,args=(self, 1))
        threads.append(cnn)
        #thread to load session infos, date, rat number and goal location
        session = threading.Thread(target=load_session,args=(self,1))      
        threads.append(session) 
        ##thread to load all variables and create video and .txt saving path
        var = threading.Thread(target=define_variables,args=(self,vp, nl, file_id, 1, out))   
        threads.append(var)           
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join() 
        ##ask start node for each trial    
        load_StartNodes(self, self.num_trials) 
        print('Network loaded', self.net)            
        #find location of goal node and all start nodes
        self.start_nodes_locations = self.find_location(self.node_list, self.start_nodes, self.goal)
        print('\nNumber of trials current session', self.num_trials, '\nGoal location node ', self.goal)   
        for i in range(0, len(self.start_nodes)):
          print('\nStart node trial {} '.format(i+1), self.start_nodes[i], 'location', self.start_nodes_locations[i])    
        self.Start = time.time()
        self.run_vid()
        
    #process and display video 
    def run_vid(self):
        '''
        Frame by Frame looping of video
        '''
        print('Starting video...\n')
        with open(self.save, 'a+') as file:
            file.write(f"Rat number: {self.rat} , Date: {self.date} \n")
  
        while True:         
            ret, self.frame = self.cap.read()

            self.frame_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count=  self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.converted_time = convert_milli(int(self.frame_time))
   
            #close video output and print time tracking
            if self.end_session:
                print ('End Session in ', round(time.time() - self.Start, 2))
                print('Session ended with ', self.trial_num ,' trials')           
                self.cap.release()
                self.out.release()
                break               
            
            #process and display frame
            if self.frame is not None:
                self.disp_frame = self.frame.copy()
                self.disp_frame = cv2.resize(self.disp_frame, (1176, 712))                 
                self.CNN(self.disp_frame)  #, Rat, tracker,Init,boxes
                self.annotate_frame(self.disp_frame)
           #     cv2.imshow('Tracker', self.disp_frame)
                self.out.write(self.disp_frame)
                              
            #log present centroid position if program is in 'save mode'
            if self.record_detections:
                if self.pos_centroid is not None:
                    if self.saved_nodes:
                        logger.info(f'{self.converted_time} : The rat position is: {self.pos_centroid} @ {self.saved_nodes[-1]}')                  
                    else:
                        logger.info(f'{self.converted_time} : The rat position is: {self.pos_centroid}') #pos_centroid

           # key = cv2.waitKey(1) & 0xFF
            #if key == ord('q'):
             #   print ('End Session in ', convert_milli(round(time.time() - self.Start, 2)))
              #  print('Session ended with ', self.trial_num ,' trials')
               # print('#Program ended by user')
                #break        
        self.cap.release()
        self.out.release()
       # cv2.destroyAllWindows()  ##Uncomment if not in cv2 ver 4.5.2


    def find_start(self, center_rat):
        '''
        Function to find start of each trial [rat close to start node]

        '''             
        ##calculate coordinate rectangle in start node 
        print( '\n' , self.converted_time, ' Next Trial', self.trial_num +1 , ' Start node', self.start_nodes[self.trial_num])       
        print('Rat position', self.pos_centroid, 'Node', self.start_nodes_locations[self.trial_num])
        node =  self.start_nodes_locations[self.trial_num]
        x = int(node[0])
        y= int(node[1]) 
        w= 15 
        h=  13                  
        cv2.rectangle(self.disp_frame, (x-w,y+h), (x+w,y-h),(0,255,0), 2) 
        if points_dist(center_rat, node) < 60: 
                       self.trial_num += 1
                       print('\nTrial ', self.trial_num, '\nTracking start from ',center_rat, node, 'distance', round(points_dist(center_rat, node)))                      
                       logger.info('Recording Trial {}'.format(self.trial_num))                        
                       self.node_pos = []
                       self.path = []
                       self.centroid_list = []
                       self.time_points=[]
                       self.summary_trial=[] 
                       self.saved_nodes = []        
                       self.node_id = []   ##node num
                       self.saved_velocities=[]
                       self.centroid_list.append(center_rat) 
                       self.record_detections = True  
                       self.start= False
        
    def CNN(self, frame):
        self.t1 = time.time()
        ##input to the CNN - blob.shape: (1, 3, 416, 416)
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        self.net.setInput(blob)
        #frame width and width to draw boxes in correct position
        height, width, _ = frame.shape
        output_layers_names = self.net.getUnconnectedOutLayersNames()    
        layerOutputs = self.net.forward(output_layers_names)   ##get prediction from cnn layers
        
        boxes = []  #boxes coordinate as x,y,height, width
        confidences = [] #set to 0.7
        class_ids = [] #researcher, rat, head       
    
        for output in layerOutputs:
          for detection in output: ##dection in frame check with pretrained darknet
              scores = detection[5:]            
              class_id = np.argmax(scores)
              confidence = scores[class_id]
              if confidence > 0.7:
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
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.70, 0.2) ##keep boxes with higher confidence
    ##go through the detections remeainingafter filtering out the one with confidence < 0.7
        if len(indexes)>0:   ##indices box= box[i], x=box[0],y=box[1],w=[box[2],h=box[3]]
          for i in indexes.flatten():
            x, y, w, h = boxes[i]       
            label = str(self.classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]    ##different color for each detected object
            cv2.rectangle(self.disp_frame, (x,y), (x+w, y+h), color, 2) ##bounding box
            cv2.putText(self.disp_frame, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 1)                       
            #if label == 'researcher':
             #   self.count_human += 1                              
            if label == 'rat':
                center_rat =(x,y) # (int((x + w)/2), int((y + h)/2))
                if center_rat is not None:
                    self.count_rat += 1                   
                    if self.start == True:
                          self.find_start(center_rat)                    
                    if self.record_detections:             
                      self.path.append(center_rat)       
                       ##Check if rat reached Goal location                    
                      if points_dist(center_rat, self.goal_location) <= 22:                        
                           cv2.putText(self.disp_frame, "Goal location reached", (30,70), 0, 1, (0,250,0), 2) 
                           print('\nRat end trial ', self.trial_num, ' out of ', self.num_trials, '\nCount rat', self.count_rat, ' head', self.count_head)
                           self.count_rat=0    
                           self.calculate_velocity(self.time_points)
                           self.save_to_file(self.save)
                           self.start = True
                           self.record_detections = False 
                           ##Check if session is finished      
                           if self.trial_num == int(self.num_trials):   
                                  print('\nEnd session with ', self.trial_num, 'trials out of ', self.num_trials, '\nCount rat', self.count_rat, ' head', self.count_head)                                                                                                           
                                  self.end_session = True  
                         # self.create_heatmap()
                                             
                                                             
            if label == 'head':
              if x is not None:
                 center_head= (x,y)#int((x + w)/2), int((y + h)/2)
                 if center_head is not None:
                   self.pos_centroid = center_head
                   self.count_head += 1 
                   if self.start == True: 
                          self.find_start(self.pos_centroid)                    
                   if self.record_detections:
                     self.centroid_list.append(self.pos_centroid)                    
                     ##Check if rat reached Goal location
                     if points_dist(self.pos_centroid, self.goal_location) <= 22:                        
                         cv2.putText(self.disp_frame, "\nGoal location reached", (30,70), 0, 1, (0,250,0), 2) 
                         print('Head end trial ', self.trial_num, ' out of ', self.num_trials, '\nCount rat', self.count_rat, ' head', self.count_head)
                         self.calculate_velocity(self.time_points)
                         self.save_to_file(self.save)                         
                         self.count_head =0                      
                         ##Check if session is finished                                        
                         if self.trial_num == int(self.num_trials):   
                                  cv2.putText(self.disp_frame, "\nEnd Session", (30,70), 0, 1, (0,250,0), 2)                                  
                                  print('Session ends with', self.trial_num)                                                                                                              
                              #    self.create_heatmap()                                  
                                  self.end_session = True                       
                         else:                             
                             self.start= True  
                             self.record_detections = False  


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
           cv2.circle(frame, point, 20, color = (0, 255, 0), thickness = 1)
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

        #annotate all nodes the rat has traversed
        for i in range(0, len(self.saved_nodes)):
            self.annotate_node(frame, point = self.node_pos[i], node = self.saved_nodes[i], t= 2) ##t=2 walked node during the trial
        #frame annotations during recording
        if self.record_detections:          
        #if the centroid position of rat is within 22 pixels of any node
        #register that node to a list. 
          if self.pos_centroid is not None:
            for node_name in nodes_dict:
                if points_dist(self.pos_centroid, nodes_dict[node_name]) <= 22:                    
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
 
                                      
    #save recorded nodes to file
    def save_to_file(self, fname):
        savelist = []
        with open(fname, 'a+') as file:
            for k, g in groupby(self.saved_nodes):
                savelist.append(k)         
            file.writelines('%s,' % items for items in savelist)
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

    enter = input('Enter unique file name: ')
    file_id = '' if not enter else enter

    print('#\nLite Tracker version: v2.00\n#\n')
    #logger intitialisations
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)

    logfile_name = '{}/log_{}_{}.log'.format(args.output, str(today), file_id)

    fh = logging.FileHandler(str(logfile_name))
    formatter = logging.Formatter('%(levelname)s : %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh) 

    node_list = Path('/content/TrackerColab/node_list_new.csv').resolve()
    
    logger.info('Video Imported: {}'.format(args.vid_path))
    print('creating log files...')
    
    Tracker(vp = args.vid_path, nl = node_list, file_id = file_id, out = args.output) 


    


    
            
        
