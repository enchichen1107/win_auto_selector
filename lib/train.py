import multiprocessing
import cv2 as cv
import numpy as np
import sys
import os
import os.path
import random
import ctypes
import shutil
import pyautogui
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import sqlite3
from distutils.dir_util import copy_tree
from keras import models
from keras.models import Model
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import clone_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import pandas as pd
import shutil


modelName = "init"
fontpath = "./assets/NotoSansTC-Medium.otf"
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh





class TrainWorker(object):
    
    
    def __init__(self):
        self._is_alive = multiprocessing.Value(ctypes.c_bool, True)

        # init screen size
        s_size = pyautogui.size()        
        self.Width = s_size[0]
        self.Height = s_size[1]
        self.width = self.Width
        self.height =  self.Height
        if self.Width>1280:
            self.width = 1280
        if self.Height>800:
            self.height = 800

        # init used folders
        self.dirFace_train_0 = "./models/data/train/0/"
        self.dirFace_train_1 = "./models/data/train/1/"
        self.dirFace_test_0 = "./models/data/test/0/"
        self.dirFace_test_1 = "./models/data/test/1/"
        self.new_data = [self.dirFace_train_0, self.dirFace_train_1, self.dirFace_test_0, self.dirFace_test_1]
        self.new_srcs = ["./models/data/train/", "./models/data/test/"]
        self.new_train_srcs = [self.dirFace_train_0, self.dirFace_train_1]
        
        self.dirFace_buffer0 = "./models/buffer/data/0/"
        self.dirFace_buffer1 = "./models/buffer/data/1/"
        Path(self.dirFace_buffer0).mkdir(parents=True, exist_ok=True)
        Path(self.dirFace_buffer1).mkdir(parents=True, exist_ok=True)
        self.buffer_srcs = [self.dirFace_buffer0, self.dirFace_buffer1]
        
        self.dirFace_retrieve_train_0 = "./models/retrieve/train/0/"
        self.dirFace_retrieve_train_1 = "./models/retrieve/train/1/"
        self.dirFace_retrieve_test_0 = "./models/retrieve/test/0/"
        self.dirFace_retrieve_test_1 = "./models/retrieve/test/1/"
        self.dests = [self.dirFace_retrieve_train_0, self.dirFace_retrieve_test_0, self.dirFace_retrieve_train_1, self.dirFace_retrieve_test_1]
        self.dirs = ["./models/retrieve/train/", "./models/retrieve/test/"]
        for folder in self.dests:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        
        
        self.buffer_len = len([name for name in os.listdir(self.dirFace_buffer0) if os.path.isfile(os.path.join(self.dirFace_buffer0, name))])
        
        # init numbers for retrieving
        self.target_num = 0  
        self.ncm_num = 0
        if self.buffer_len > 0:
            if self.buffer_len < 90:
                self.target_num = 36  
                self.ncm_num = 36
            else:
                self.target_num = 45  
                self.ncm_num = 45
                       
        # init numbers for collecting
        self.CTRL_TOTAL = 90
        self.EXP_TOTAL = 90
        self.prompt = True
        self.response1 = False
        self.response2 = False
        self.pressed1 = False
        self.pressed2 = False
        self.show_control = False
        self.show_instru = False
        self.mouseX = 0
        self.mouseY = 0
        self.rest = 0
        
        # init numbers for training
        self.try_lr = 0.003
        self.EPOCH = 20
        self.tEPOCH = 35
        self.ncm_fc1 = 16
        self.train_batch = 8
        self.test_batch = 8     
       
        # init chosen facial part
        conn = sqlite3.connect('./models/key_book.db')
        c = conn.cursor()
        c.execute("SELECT facePart FROM facials")
        record = c.fetchall()
        self.facePart = record[0][0]

        # init tilted positions (for cases who can't sit upright)
        listOfTables = c.execute(
            """SELECT * FROM sqlite_master WHERE type='table' 
            AND name='positions'; """).fetchall()
 
        if listOfTables == []:

            c.execute("""CREATE TABLE IF NOT EXISTS positions (
            pos INT
            )""")
            conn.commit()

            c.execute("INSERT INTO positions VALUES (:pos)",
                {
                    'pos': 0
                })
            self.settled = 0
            conn.commit()
            conn.close()

        else:

            c.execute("SELECT pos FROM positions")
            record = c.fetchall()
            self.settled = record[0][0]
            conn.commit()
            conn.close()


        # init positions for mediapipe and size for cropped image
        self.pos = []
        self.pos2 = []
        self.sz = []

        if self.facePart=="brow":
            self.pos = [54, 345]
            self.pos2 = [284, 116]
            self.sz = [96,48]
        elif self.facePart=="nose":
            self.pos = [119,426]
            self.pos2 = [348, 206]
            self.sz = [60,26]
        else:
            self.pos = [207,430]
            self.pos2 = [427, 210]
            self.sz = [78,28]
            
        # 100 for category 0 and 100 for category 1
        self.buffer_limit = 100               
            
        
    def draw_circle_red(self, event, x, y, flags, param):
        ''' function for handle opencv mouse event '''
        # handle prompt
        if self.prompt and event == cv.EVENT_LBUTTONDOWN and not self.response1:
            self.response1 = True
            self.response2 = False
        elif self.prompt and event == cv.EVENT_LBUTTONDOWN and self.response1:
            self.response1 = False
            self.response2 = True
        # handle drawing collect data phase's circle
        elif event == cv.EVENT_LBUTTONDOWN and not self.pressed1 and x>self.mouseX-30 and x<self.mouseX+30 and y>self.mouseY-30 and y<self.mouseY+30:
            self.pressed1 = True
            self.show_control = True
        elif event == cv.EVENT_LBUTTONDOWN and self.pressed1 and x>self.mouseX-30 and x<self.mouseX+30 and y>self.mouseY-30 and y<self.mouseY+30:
            self.pressed1 = False
            self.show_instru = False
            self.pressed2 = True    
        
        
    def get_is_alive(self):
        with self._is_alive.get_lock():
            return self._is_alive.value
        

    def set_is_alive(self, value):
        with self._is_alive.get_lock():
            self._is_alive.value = value
    is_alive = property(get_is_alive, set_is_alive)
           
        
    def start(self):
        ''' entry function for training '''
        self.domain_increment()
        self.collect()
        self.retrieve()
        self.model_training()
        self.update_buffer()
        
        
    def stop(self): 
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        cap.release()
        cv.destroyAllWindows()
                
                
    def domain_increment(self):
        ''' get current domain count and update it '''
        conn = sqlite3.connect('./models/key_book.db')
        c = conn.cursor()
        listOfTables = c.execute(
            """SELECT * FROM sqlite_master WHERE type='table' 
            AND name='domains'; """).fetchall()
 
        if listOfTables == []:

            c.execute("""CREATE TABLE IF NOT EXISTS domains (
            id INT,
            counts INT
            )""")
            conn.commit()

            # init domain count
            c.execute("INSERT INTO domains VALUES (:id, :counts)",
                {
                    'id': 1,
                    'counts': 1
                })
            conn.commit()
            conn.close()

        else:

            # get current domain count
            c.execute("SELECT counts FROM domains WHERE id = 1")
            record = c.fetchone()
            new_counts = record[0]+1
            conn.commit()
            
            # update domain count
            c.execute("UPDATE domains SET counts = ? WHERE id = ?", (new_counts,1))
            conn.commit()
            conn.close()
            

    def collect(self):
        
        ''' function for collecting data '''
        
        # remove old domain data      
        for folder in self.new_data:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)

        
        cv.namedWindow("subtle facial", cv.WINDOW_NORMAL)
        cv.resizeWindow('subtle facial', self.width, self.height)
        cv.moveWindow('subtle facial',int((self.Width-self.width)/2), int((self.Height-self.height)/2))
        cv.setMouseCallback('subtle facial', self.draw_circle_red)
        
        # get current domain count
        conn = sqlite3.connect('./models/key_book.db')
        c = conn.cursor()
        c.execute("SELECT counts FROM domains WHERE id = 1")
        record = c.fetchone()
        name_idx = record[0]
        conn.commit()
        conn.close()
        
        '''
        DATA COLLECTION PROMPT
        '''
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.flip(frame, 1)
            font = ImageFont.truetype(fontpath, 35)      
            imgPil = Image.fromarray(frame)                
            draw = ImageDraw.Draw(imgPil)                
            draw.text((10, 10), "接下來會進行30回合收資料\n每回合都需點擊畫面中的圓點\n藍點請保持一般表情、紅點請做微表情\n準備好請點擊畫面", fill=(219, 12, 242), font=font)
            frame = np.array(imgPil)      
            cv.imshow("subtle facial", frame)
            if self.response1:
                break
            key = cv.waitKey(1)
            if key == ord('q'):
                sys.exit(0)

        '''
        DATA PREPARE PROMPT
        '''

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # detect faces
                frame = cv.flip(frame, 1)
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                img_h, img_w = frame.shape[:2]
                results = face_mesh.process(rgb_frame)
                if not results.multi_face_landmarks:
                    continue
                else:
                    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                    
                    # logic for tilted head positions
                    if self.settled == 0:
                        if (mesh_points[self.pos[1]][0]-mesh_points[self.pos[0]][0])*(mesh_points[self.pos[1]][1]-mesh_points[self.pos[0]][1])<(mesh_points[self.pos2[0]][0]-mesh_points[self.pos2[1]][0])*(mesh_points[self.pos2[1]][1]-mesh_points[self.pos2[0]][1]):
                            # tilt left
                            self.settled = 2
                        else:
                            # tilt right
                            self.settled = 1
                        conn = sqlite3.connect('./models/key_book.db')
                        c = conn.cursor()
                        c.execute("UPDATE positions SET pos = ?", (self.settled,))
                        print(self.settled)
                        conn.commit()
                        conn.close()
                                
                    
                    # draw rectangles and circles
                    if self.settled == 1:
                        cv.rectangle(frame, tuple((mesh_points[self.pos[0]][0]-5,mesh_points[self.pos[0]][1]-5)),tuple((mesh_points[self.pos[1]][0]+5,mesh_points[self.pos[1]][1]+5)),(0,255,0),3)
                        
                        cv.circle(frame, tuple((mesh_points[self.pos[0]][0]-5,mesh_points[self.pos[0]][1]-5)), radius=5, color=(0, 0, 255), thickness=-1)
                        cv.circle(frame, tuple((mesh_points[self.pos[1]][0]+5,mesh_points[self.pos[1]][1]+5)), radius=5, color=(0, 0, 255), thickness=-1)
                    elif self.settled == 2:
                        cv.rectangle(frame, tuple((mesh_points[self.pos2[1]][0]-5,mesh_points[self.pos2[0]][1]-5)),tuple((mesh_points[self.pos2[0]][0]+5,mesh_points[self.pos2[1]][1]+5)),(0,255,0),3)
                        
                        cv.circle(frame, tuple((mesh_points[self.pos2[0]][0]-5,mesh_points[self.pos2[0]][1]-5)), radius=5, color=(0, 0, 255), thickness=-1)
                        cv.circle(frame, tuple((mesh_points[self.pos2[1]][0]+5,mesh_points[self.pos2[1]][1]+5)), radius=5, color=(0, 0, 255), thickness=-1)

                    font = ImageFont.truetype(fontpath, 35)      
                    imgPil = Image.fromarray(frame)                
                    draw = ImageDraw.Draw(imgPil)                
                    draw.text((10, 10), "請確認欲使用的微表情是否能被系統正確框出\n若無法 請將光源調亮\n準備好請點擊畫面", fill=(219, 12, 242), font=font)
                    frame = np.array(imgPil)      
                    cv.imshow("subtle facial", frame)
                    if self.response2:
                        break
                    key = cv.waitKey(1)
                    if key == ord('q'):
                        sys.exit(0)
            
                
        '''
        DATA COLLECTION
        '''

        self.prompt = False

        # calculate positions for targets////////////////////////////
        mid_pt = []
        for y in range(int(self.height/4),self.height,int(self.height/4)):
            for x in range(int(self.width/4),self.width,int(self.width/4)):
                mid_pt.append([x,y])

        pt_pos = random.randint(0,8) 
        prev_pos = pt_pos
        pt_cnt = np.zeros(9)

        control = 0
        ctrl_cnt = 0
        exp_cnt = 0
        show_green = 0
        round = 0

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # detect faces
                frame = cv.flip(frame, 1)
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                img_h, img_w = frame.shape[:2]
                results = face_mesh.process(rgb_frame)
                if not results.multi_face_landmarks:
                    continue
                else:
                    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                
                    
                    # crop image
                    if self.settled == 1:
                        cropped_img = frame[mesh_points[self.pos[0]][1]:mesh_points[self.pos[1]][1],mesh_points[self.pos[0]][0]:mesh_points[self.pos[1]][0]].copy()
                    elif self.settled == 2:
                        cropped_img = frame[mesh_points[self.pos2[0]][1]:mesh_points[self.pos2[1]][1],mesh_points[self.pos2[1]][0]:mesh_points[self.pos2[0]][0]].copy()
                    try:
                        cropped_img = cv.resize(cropped_img,(self.sz[0],self.sz[1]))
                    except Exception as e:
                        continue

                    
                    # logic for alternatively showing instructions and random circles
                    self.mouseX = mid_pt[pt_pos][0]
                    self.mouseY = mid_pt[pt_pos][1]
                    if self.pressed1:
                        if self.show_control:
                            control+=1
                            show_green+=1
                        if show_green>10:
                            cv.circle(frame, (mid_pt[pt_pos][0],mid_pt[pt_pos][1]), 30, (0,0,255), -1)
                    elif self.pressed2 and self.rest<30:
                        self.rest+=1
                    elif self.pressed2 and self.rest==30:
                        self.rest = 0
                        show_green = 0
                        self.pressed2 = False
                        pt_pos = random.randint(0,8)
                        while pt_cnt[pt_pos]==(self.EXP_TOTAL/(3*9)+1) or pt_pos==prev_pos:
                            pt_pos = random.randint(0,8)
                        pt_cnt[pt_pos]+=1
                        prev_pos = pt_pos
                    else:
                        font = ImageFont.truetype(fontpath, 35)      
                        imgPil = Image.fromarray(frame)                
                        draw = ImageDraw.Draw(imgPil)                
                        draw.text((mid_pt[pt_pos][0]+40,mid_pt[pt_pos][1]-30), "請保持一般表情\n點擊圓點", fill=(255, 0, 0), font=font)
                        frame = np.array(imgPil)
                        cv.circle(frame, (mid_pt[pt_pos][0],mid_pt[pt_pos][1]), 30, (255,0,0), -1)



                    key = cv.waitKey(1) & 0xFF
                    if key == ord('q'):
                        sys.exit(0)

                    if self.rest==1:
                        round+=1

                    # collect subtle facial expression img
                    if self.rest==1 or self.rest==5 or self.rest==9:
                        exp_cnt+=1               
                        # saving faces according to detected coordinates 
                        FaceFileName = ""
                        if exp_cnt<=(self.EXP_TOTAL*0.8):
                            FaceFileName = "./models/data/train/1/exp_" + str(name_idx)+"-"+ str(exp_cnt) + ".jpg" # folder path and random name image
                        else:
                            FaceFileName = "./models/data/test/1/exp_" + str(name_idx)+"-"+ str(exp_cnt) + ".jpg" # folder path and random name image

                        cv.imwrite(FaceFileName, cropped_img)


                    if self.rest>1 and self.rest<10:
                        font = ImageFont.truetype(fontpath, 35)      
                        imgPil = Image.fromarray(frame)                
                        draw = ImageDraw.Draw(imgPil)                
                        draw.text((mid_pt[pt_pos][0]+40,mid_pt[pt_pos][1]-30), "正在收微表情", fill=(0, 0, 255), font=font)
                        frame = np.array(imgPil)

                    # collect control img
                    if control==1 or control==5 or control==9:
                        ctrl_cnt+=1
                        FaceFileName = ""
                        if ctrl_cnt<=(self.CTRL_TOTAL*0.8):
                            FaceFileName = "./models/data/train/0/ctrl_" + str(name_idx)+"-"+ str(ctrl_cnt) + ".jpg" # folder path and random name image
                        else:
                            FaceFileName = "./models/data/test/0/ctrl_" + str(name_idx)+"-"+ str(ctrl_cnt) + ".jpg" # folder path and random name image

                        cv.imwrite(FaceFileName, cropped_img)

                    elif control>10:
                        self.show_control = False
                        self.show_instru = True
                        control = 0
                        
                    if ctrl_cnt == self.CTRL_TOTAL and exp_cnt == self.EXP_TOTAL:
                        break


                    if self.show_control:
                        font = ImageFont.truetype(fontpath, 35)      
                        imgPil = Image.fromarray(frame)                
                        draw = ImageDraw.Draw(imgPil)                
                        draw.text((mid_pt[pt_pos][0]+40,mid_pt[pt_pos][1]-30), "正在收一般表情", fill=(255, 0, 0), font=font)
                        frame = np.array(imgPil)
                                                
                    

                    if self.show_instru:
                        font = ImageFont.truetype(fontpath, 35)      
                        imgPil = Image.fromarray(frame)                
                        draw = ImageDraw.Draw(imgPil)                
                        draw.text((mid_pt[pt_pos][0]+40,mid_pt[pt_pos][1]-30), "請作微表情\n點擊圓點", fill=(0, 0, 255), font=font)
                        frame = np.array(imgPil)
                    

                    # draw rectangle and circles
                    if self.settled == 1:
                        cv.rectangle(frame, tuple((mesh_points[self.pos[0]][0]-5,mesh_points[self.pos[0]][1]-5)),tuple((mesh_points[self.pos[1]][0]+5,mesh_points[self.pos[1]][1]+5)),(0,255,0),3)
                        
                        cv.circle(frame, tuple((mesh_points[self.pos[0]][0]-5,mesh_points[self.pos[0]][1]-5)), radius=5, color=(0, 0, 255), thickness=-1)
                        cv.circle(frame, tuple((mesh_points[self.pos[1]][0]+5,mesh_points[self.pos[1]][1]+5)), radius=5, color=(0, 0, 255), thickness=-1)
                    elif self.settled == 2:
                        cv.rectangle(frame, tuple((mesh_points[self.pos2[1]][0]-5,mesh_points[self.pos2[0]][1]-5)),tuple((mesh_points[self.pos2[0]][0]+5,mesh_points[self.pos2[1]][1]+5)),(0,255,0),3)
                        
                        cv.circle(frame, tuple((mesh_points[self.pos2[0]][0]-5,mesh_points[self.pos2[0]][1]-5)), radius=5, color=(0, 0, 255), thickness=-1)
                        cv.circle(frame, tuple((mesh_points[self.pos2[1]][0]+5,mesh_points[self.pos2[1]][1]+5)), radius=5, color=(0, 0, 255), thickness=-1)

                    cv.putText(frame, "Round "+str(round)+" / 30", (100,100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)

                    

                # Video Window
                cv.imshow('subtle facial',frame)
        
        '''
        ANALYSIS PROMPT
        '''

        self.prompt = True
        print("end collection")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            font = ImageFont.truetype(fontpath, 35)      
            imgPil = Image.fromarray(frame)                
            draw = ImageDraw.Draw(imgPil)                
            draw.text((10, 10), "結束收資料流程，接下來將進行分析\n約需等待三分鐘，請點擊畫面開始分析", fill=(219, 12, 242), font=font)
            frame = np.array(imgPil)
            cv.imshow("subtle facial", frame)
            if self.response1:
                break
            key = cv.waitKey(1)
            if key == ord('q'):
                sys.exit(0)
                

    def retrieve(self):
        ''' function for retrieving data '''
        
        shutil.copytree(self.dirFace_train_0, self.dirFace_retrieve_train_0)
        shutil.copytree(self.dirFace_train_1, self.dirFace_retrieve_train_1)
        shutil.copytree(self.dirFace_test_0, self.dirFace_retrieve_test_0) 
        shutil.copytree(self.dirFace_test_1, self.dirFace_retrieve_test_1)
                
        ncm_fc1 = 16
        
        if self.buffer_len!=0:
            picked = []
            # mir //////////////////////////////////////////////////

            # calculate pre loss
            model = load_model('./models/'+modelName+'.h5')
            pre_loss_0 = []
            name_arr_0 = []
            pre_loss_1 = []
            name_arr_1 = []
            folder_dir = self.buffer_srcs[0]
            for images in os.listdir(folder_dir):
                if images.endswith(".jpg"):
                    img = cv.imread(folder_dir+images) 
                    img = np.expand_dims(img, axis=0)
                    img = img.astype(np.float32) / 255.0
                    single_loss_value, _ = model.evaluate(img, np.array([1]))
                    pre_loss_0.append(single_loss_value)
                    name_arr_0.append(folder_dir+images)
            folder_dir = self.buffer_srcs[1]
            for images in os.listdir(folder_dir):
                if (images.endswith(".jpg")):
                    img = cv.imread(folder_dir+images) 
                    img = np.expand_dims(img, axis=0)
                    img = img.astype(np.float32) / 255.0
                    single_loss_value, _ = model.evaluate(img, np.array([0]))
                    pre_loss_1.append(single_loss_value)
                    name_arr_1.append(folder_dir+images)

            new_model = clone_model(model)
            new_model.compile(optimizer=Adam(learning_rate=self.try_lr),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            new_model.set_weights(model.get_weights())
            train_datagen = ImageDataGenerator(rescale=1./255)

            test_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_directory(self.new_srcs[0], target_size=(self.sz[0],self.sz[1]), 
                                                                batch_size=self.train_batch, class_mode='binary', shuffle=True)
            test_generator = test_datagen.flow_from_directory(self.new_srcs[1], target_size=(self.sz[0],self.sz[1]), 
                                                                batch_size=self.test_batch, class_mode='binary', shuffle=True)
            new_model.fit(train_generator, epochs=self.EPOCH, steps_per_epoch=train_generator.samples/self.train_batch, 
                                validation_data=test_generator, validation_steps=test_generator.samples/self.test_batch)


            # calculate post loss after update model with new domain data
            post_loss_0 = []
            post_loss_1 = []
            folder_dir = self.buffer_srcs[0]
            for images in os.listdir(folder_dir):
                if images.endswith(".jpg"):
                    img = cv.imread(folder_dir+images) 
                    img = np.expand_dims(img, axis=0)
                    img = img.astype(np.float32) / 255.0
                    single_loss_value, _ = new_model.evaluate(img, np.array([1]))
                    post_loss_0.append(single_loss_value)
            folder_dir = self.buffer_srcs[1]
            for images in os.listdir(folder_dir):
                if images.endswith(".jpg"):
                    img = cv.imread(folder_dir+images) 
                    img = np.expand_dims(img, axis=0)
                    img = img.astype(np.float32) / 255.0
                    single_loss_value, _ = new_model.evaluate(img, np.array([0]))
                    post_loss_1.append(single_loss_value)


            # retrieve data with most interfered
            pd.options.display.float_format = '{:.30f}'.format
            scores1 = np.array(post_loss_1)-np.array(pre_loss_1)
            scores1 = np.array(scores1)
            name_arr_1 = np.array(name_arr_1)
            name_arr_1 = name_arr_1.ravel()
            df = pd.DataFrame()
            df['scores1']=pd.Series(scores1)
            df['name_arr_1']=pd.Series(name_arr_1)
            df = df.sort_values(by="scores1",ascending=False)
            df1 = df[:self.target_num]

            pd.options.display.float_format = '{:.30f}'.format
            scores0 = np.array(post_loss_0)-np.array(pre_loss_0)
            scores0 = np.array(scores0)
            name_arr_0 = np.array(name_arr_0)
            name_arr_0 = name_arr_0.ravel()
            df = pd.DataFrame()
            df['scores0']=pd.Series(scores0)
            df['name_arr_0']=pd.Series(name_arr_0)
            df = df.sort_values(by="scores0",ascending=False)
            df2 = df[:self.target_num]


            # 1:4 test, train
            cnt = 0
            for img in df2['name_arr_0']:
                cnt+=1
                if cnt<=self.target_num/5:
                    shutil.copy(img, self.dests[1])
                else:
                    shutil.copy(img, self.dests[0])
                picked.append(img)
            cnt = 0
            for img in df1['name_arr_1']:
                cnt+=1
                if cnt<=self.target_num/5:
                    shutil.copy(img, self.dests[3])
                else:
                    shutil.copy(img, self.dests[2])
                picked.append(img)


            # ncm //////////////////////////////////////////////////          
            
            model = load_model('./models/'+modelName+'.h5')
            feature_network = Model(model.input, model.get_layer('fc1').output)

            # prepare feature from layer fc1
            all_feat_0 = np.empty([1,ncm_fc1])
            name_arr_0 = []
            all_feat_1 = np.empty([1,ncm_fc1])
            name_arr_1 = []
            folder_dir = self.buffer_srcs[0]
            for images in os.listdir(folder_dir):
                if images.endswith(".jpg"):
                    img = cv.imread(folder_dir+images) 
                    img = np.expand_dims(img, axis=0)
                    img = img.astype(np.float32) / 255.0
                    feature = feature_network(img,training=False)
                    all_feat_0 = np.vstack((all_feat_0,feature))
                    name_arr_0.append(folder_dir+images)
            folder_dir = self.buffer_srcs[1]
            for images in os.listdir(folder_dir):
                if images.endswith(".jpg"):
                    img = cv.imread(folder_dir+images) 
                    img = np.expand_dims(img, axis=0)
                    img = img.astype(np.float32) / 255.0
                    feature = feature_network(img,training=False)
                    all_feat_1 = np.vstack((all_feat_1,feature))
                    name_arr_1.append(folder_dir+images)  
            name_arr_0 = np.array(name_arr_0)
            name_arr_0 = np.reshape(name_arr_0,(-1,1))
            name_arr_1 = np.array(name_arr_1)
            name_arr_1 = np.reshape(name_arr_1,(-1,1))
            all_feat_0 = np.delete(all_feat_0, (0), axis=0)
            all_feat_1 = np.delete(all_feat_1, (0), axis=0)

            # calculate mean for features
            ctrl_mean_0 = np.mean(all_feat_0, axis=0)
            ctrl_mean_1 = np.mean(all_feat_1, axis=0)

            # retrieve data with nearest feature distance with mean
            all_dist_0 = []
            for row in all_feat_0:
                dist = np.linalg.norm(row - ctrl_mean_0)
                all_dist_0.append(dist)
            all_dist_0 = np.array(all_dist_0)
            all_dist_0 = np.reshape(all_dist_0,(-1,1))
            df0 = np.hstack((all_dist_0,name_arr_0))
            g0 = df0[df0[:, 0].argsort()]

            all_dist_1 = []
            for row in all_feat_1:
                dist = np.linalg.norm(row - ctrl_mean_1)
                all_dist_1.append(dist)
            all_dist_1 = np.array(all_dist_1)
            all_dist_1 = np.reshape(all_dist_1,(-1,1))
            df1 = np.hstack((all_dist_1,name_arr_1))
            g1 = df1[df1[:, 0].argsort()]

            cnt = 0
            j = 0
            while cnt < self.ncm_num:
                img = g0[j,1]
                while img in picked:
                    j+=1
                    img = g0[j,1]
                cnt+=1
                if cnt<=self.ncm_num/5:
                    shutil.copy(img, self.dests[1])
                else:
                    shutil.copy(img, self.dests[0])
                picked.append(img)
            cnt = 0
            j = 0
            while cnt < self.ncm_num:
                img = g1[j,1]
                while img in picked:
                    j+=1
                    img = g1[j,1]
                cnt+=1
                if cnt<=self.ncm_num/5:
                    shutil.copy(img, self.dests[3])
                else:
                    shutil.copy(img, self.dests[2])
                picked.append(img)
        
        
        
        
    def model_training(self):
        
        ''' function for training model '''

        filepath = './models/'+modelName+'.h5'
        
        if not os.path.isfile(filepath):

            model = models.Sequential()

            model.add(layers.Conv2D(8, (3, 3), activation='relu', name='layer_conv1', padding='same', input_shape=(self.sz[1],self.sz[0],3)))
            model.add(layers.MaxPooling2D((2, 2), name='layer_maxpool1'))

            model.add(layers.Conv2D(8, (3, 3), activation='relu', name='layer_conv2', padding='same'))
            model.add(layers.MaxPooling2D((2, 2), name='layer_maxpool2'))

            model.add(layers.Flatten(name='flatten1'))
            model.add(layers.Dense(16, activation='relu', name='fc1'))
            model.add(layers.Dropout((0.1), name='dropout'))
            model.add(layers.Dense(8, activation='relu', name='fc2'))
            model.add(layers.Dense(1, activation='sigmoid', name='fc3'))

            model.summary()

            model.compile(optimizer=Adam(learning_rate=self.try_lr),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        

            checkpoint = ModelCheckpoint(filepath=filepath, 
                                         monitor='val_loss',
                                         verbose=1, 
                                         save_best_only=True,
                                         mode='min')
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta = 0.001, verbose = 1)
            callbacks = [checkpoint]

            train_datagen = ImageDataGenerator(rescale=1./255)

            test_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_directory(self.dirs[0], target_size=(self.sz[1],self.sz[0]), 
                                                                batch_size=self.train_batch, class_mode='binary', shuffle=True)
            test_generator = test_datagen.flow_from_directory(self.dirs[1], target_size=(self.sz[1],self.sz[0]), 
                                                                batch_size=self.test_batch, class_mode='binary', shuffle=True)
            model.fit(train_generator, epochs=self.tEPOCH, steps_per_epoch=train_generator.samples/self.train_batch, 
                                validation_data=test_generator, validation_steps=test_generator.samples/self.test_batch, callbacks=callbacks)

            scores = model.evaluate(test_generator)
            print("%s%s: %.2f%%" % ("evaluate ",model.metrics_names[1], scores[1]*100))

            model.save(filepath)

        
        else:

            model = load_model(filepath)  
            checkpoint = ModelCheckpoint(filepath=filepath, 
                                         monitor='val_loss',
                                         verbose=1, 
                                         save_best_only=True,
                                         mode='min')

            callbacks = [checkpoint]

            train_datagen = ImageDataGenerator(rescale=1./255)

            test_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_directory(self.dirs[0], target_size=(self.sz[1],self.sz[0]), 
                                                                batch_size=self.train_batch, class_mode='binary', shuffle=True)
            test_generator = test_datagen.flow_from_directory(self.dirs[1], target_size=(self.sz[1],self.sz[0]), 
                                                                batch_size=self.test_batch, class_mode='binary', shuffle=True)
            model.fit(train_generator, epochs=self.tEPOCH, steps_per_epoch=train_generator.samples/self.train_batch, 
                                validation_data=test_generator, validation_steps=test_generator.samples/self.test_batch,           callbacks=callbacks)

            scores = model.evaluate(test_generator)
            print("%s%s: %.2f%%" % ("evaluate ",model.metrics_names[1], scores[1]*100))

            model.save(filepath)
            
            
            
           
    def update_buffer(self):
                
        ''' function for updating buffer '''
        
        conn = sqlite3.connect('./models/key_book.db')
        c = conn.cursor()
        c.execute("SELECT counts FROM domains WHERE id = 1")
        record = c.fetchone()
        domain_cnt = record[0]
        conn.commit()
        conn.close()
        
        random_picked_cnt = self.buffer_limit/domain_cnt

        if self.buffer_len==0:
            for i in range(len(self.buffer_srcs)):
                src = self.new_train_srcs[i]
                dest = self.buffer_srcs[i]
                copy_tree(src, dest)

        elif self.buffer_len<self.buffer_limit:
            picked_file = []
            target_num = self.buffer_limit-self.buffer_len
            # first random choose from src to add up to buffer limit
            for i in range(len(self.buffer_srcs)):
                src = self.new_train_srcs[i]
                dest = self.buffer_srcs[i]
                cnt = 0        
                while cnt<target_num:
                    f = random.choice(os.listdir(src))
                    while f in picked_file:
                        f = random.choice(os.listdir(src))
                    picked_file.append(f)
                    cnt+=1
                    shutil.copy(os.path.join(src, f), dest)
            # for the remaining random exchange buffer data with new domain data
            target_num2 = random_picked_cnt-target_num
            for i in range(len(self.buffer_srcs)):
                src = self.new_train_srcs[i]
                dest = self.buffer_srcs[i]
                cnt = 0
                while cnt<target_num2:
                    del_file = random.choice(os.listdir(dest))
                    os.remove(os.path.join(dest, del_file))
                    f = random.choice(os.listdir(src))
                    while f in picked_file:
                        f = random.choice(os.listdir(src))
                    picked_file.append(f)
                    cnt+=1
                    shutil.copy(os.path.join(src, f), dest)
                    
        else:
            # random exchange buffer data with new domain data
            target_num = random_picked_cnt
            for i in range(len(self.buffer_srcs)):
                src = self.new_train_srcs[i]
                dest = self.buffer_srcs[i]
                cnt = 0
                picked_file = []
                while cnt<target_num:
                    del_file = random.choice(os.listdir(dest))
                    os.remove(os.path.join(dest, del_file))
                    f = random.choice(os.listdir(src))
                    while f in picked_file:
                        f = random.choice(os.listdir(src))
                    picked_file.append(f)
                    cnt+=1
                    shutil.copy(os.path.join(src, f), dest)
                    
        '''
        ANALYSIS PROMPT
        '''

        self.prompt = True
        print("end buffer update")

        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            font = ImageFont.truetype(fontpath, 35)      
            imgPil = Image.fromarray(frame)                
            draw = ImageDraw.Draw(imgPil)                
            draw.text((10, 10), "結束分析\n請點擊畫面離開", fill=(219, 12, 242), font=font)
            frame = np.array(imgPil)
            cv.imshow("subtle facial", frame)
            if self.response2:
                break
            key = cv.waitKey(1)
            if key == ord('q'):
                sys.exit(0)
        
        


                                             
class TrainProcess(object):
    def __init__(self):
        self.worker  = None
        self.process = None

    def __delete__(self):
        self.stop()

    def start(self):
        self.stop()
        self.worker  = TrainWorker()
        self.process = multiprocessing.Process(target=self.worker.start)
        self.process.start()
        self.stop()

    def stop(self):
        if self.is_alive:
            self.worker.is_alive = False
            self.worker.stop()
            self.process.join()
            self.worker  = None
            self.process = None

    def get_is_alive(self):
        return bool(self.worker and self.process)
    is_alive = property(get_is_alive)







