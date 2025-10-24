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
from db.handler import DBhandler
from config import *


class TrainWorker(object):
    
    
    def __init__(self):
        self.model_name = MODEL_NAME
        self.font_path = FONT_PATH
        self.db = DBhandler()
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
        # Data folders
        self.dirFace_train_0 = DIR_FACE_TRAIN_0
        self.dirFace_train_1 = DIR_FACE_TRAIN_1
        self.dirFace_test_0 = DIR_FACE_TEST_0
        self.dirFace_test_1 = DIR_FACE_TEST_1
        self.new_data = NEW_DATA
        self.new_srcs = NEW_SRCS
        self.new_train_srcs = NEW_TRAIN_SRCS

        # Buffer folders
        self.dirFace_buffer0 = DIR_FACE_BUFFER_0
        self.dirFace_buffer1 = DIR_FACE_BUFFER_1
        Path(self.dirFace_buffer0).mkdir(parents=True, exist_ok=True)
        Path(self.dirFace_buffer1).mkdir(parents=True, exist_ok=True)
        self.buffer_srcs = BUFFER_SRCS

        # Retrieve folders
        self.dirFace_retrieve_train_0 = DIR_FACE_RETRIEVE_TRAIN_0
        self.dirFace_retrieve_train_1 = DIR_FACE_RETRIEVE_TRAIN_1
        self.dirFace_retrieve_test_0 = DIR_FACE_RETRIEVE_TEST_0
        self.dirFace_retrieve_test_1 = DIR_FACE_RETRIEVE_TEST_1
        self.dests = DESTS
        self.dirs = DIRS

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
        record = self.db.get_face_part()
        self.facePart = record[0][0]

        # init tilted positions (for cases who can't sit upright)
        listOfTables = self.db.get_positions_tables()
 
        if listOfTables == []:

            self.db.create_positions_table()
            self.db.insert_face_pos(0)
            self.settled = 0

        else:

            record = self.db.get_face_pos()
            self.settled = record[0][0]


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
        import mediapipe as mp 
        self.face_mesh = mp.solutions.face_mesh
        self.face_detection = mp.solutions.face_detection
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
        self.db.domain_increment()


    def collect(self):
        """Main function for collecting facial expression data."""

        # 1️. Prepare data folders
        self._reset_new_data_folders()

        # 2️. Setup OpenCV window
        self._setup_cv_window()

        # 3️. Get current domain index
        name_idx = self.db.get_domain_count()

        # 4️. Show data collection instructions
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self._show_data_collection_prompt(cap)

        # 5️. Prepare face detection and determine head position
        self._prepare_face_mesh(cap)

        # 6️. Collect subtle facial expression and control images
        self._collect_faces(cap, name_idx)

        # 7️. Show collect end prompt
        self._show_collect_end_prompt()

            

    def retrieve(self):
        """Retrieve images based on interference and NCM selection."""
        # --- 1. Copy base directories ---
        self._copy_base_dirs()

        if self.buffer_len == 0:
            return

        ncm_fc1 = 16
        picked = []

        # --- 2. Evaluate pre-loss on existing model ---
        model = load_model(f'./models/{self.model_name}.h5')
        pre_loss_0, names_0 = self._evaluate_folder(model, self.buffer_srcs[0], label=1)
        pre_loss_1, names_1 = self._evaluate_folder(model, self.buffer_srcs[1], label=0)

        # --- 3. Train cloned model with new data ---
        new_model = self._train_new_model(model)

        # --- 4. Evaluate post-loss ---
        post_loss_0, _ = self._evaluate_folder(new_model, self.buffer_srcs[0], label=1)
        post_loss_1, _ = self._evaluate_folder(new_model, self.buffer_srcs[1], label=0)

        # --- 5. Pick most interfered images ---
        df0 = self._get_top_interfered(pre_loss_0, post_loss_0, names_0)
        df1 = self._get_top_interfered(pre_loss_1, post_loss_1, names_1)

        picked.extend(self._copy_selected(df0['name'], cls=0))
        picked.extend(self._copy_selected(df1['name'], cls=1))

        # --- 6. Nearest Class Mean (NCM) selection ---
        self._perform_ncm_selection(model, picked, ncm_fc1)



    def model_training(self):
        """Training model"""
        filepath = './models/'+self.model_name+'.h5'
        
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
            callbacks = [checkpoint]

            datagen = ImageDataGenerator(rescale=1./255)

            train_generator = datagen.flow_from_directory(self.dirs[0], target_size=(self.sz[1],self.sz[0]), 
                                                                batch_size=self.train_batch, class_mode='binary', shuffle=True)
            test_generator = datagen.flow_from_directory(self.dirs[1], target_size=(self.sz[1],self.sz[0]), 
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

            datagen = ImageDataGenerator(rescale=1./255)

            train_generator = datagen.flow_from_directory(self.dirs[0], target_size=(self.sz[1],self.sz[0]), 
                                                                batch_size=self.train_batch, class_mode='binary', shuffle=True)
            test_generator = datagen.flow_from_directory(self.dirs[1], target_size=(self.sz[1],self.sz[0]), 
                                                                batch_size=self.test_batch, class_mode='binary', shuffle=True)
            model.fit(train_generator, epochs=self.tEPOCH, steps_per_epoch=train_generator.samples/self.train_batch, 
                                validation_data=test_generator, validation_steps=test_generator.samples/self.test_batch, callbacks=callbacks)

            scores = model.evaluate(test_generator)
            print("%s%s: %.2f%%" % ("evaluate ",model.metrics_names[1], scores[1]*100))

            model.save(filepath)



    def update_buffer(self):
        """Update training buffer with new domain samples."""

        # --- 1️. Retrieve domain count ---
        domain_cnt = self.db.get_domain_count()
        random_picked_cnt = self.buffer_limit / domain_cnt

        # --- 2️. Handle different buffer states ---
        if self.buffer_len == 0:
            self._copy_all_new_to_buffer()

        elif self.buffer_len < self.buffer_limit:
            target_num = self.buffer_limit - self.buffer_len
            random_picked_cnt_remaining = random_picked_cnt - target_num
            self._add_new_data_to_buffer(target_num)
            self._replace_buffer_with_new_data(random_picked_cnt_remaining)

        else:
            # Buffer full: just random replacement
            self._replace_buffer_with_new_data(random_picked_cnt)

        # --- 3️. Show analysis prompt ---
        self._show_analysis_prompt()







    # ============================================================
    #                    HELPER FUNCTIONS
    # ============================================================

    def _copy_base_dirs(self):
        """Copy base directories before training/retrieval."""
        shutil.copytree(self.dirFace_train_0, self.dirFace_retrieve_train_0)
        shutil.copytree(self.dirFace_train_1, self.dirFace_retrieve_train_1)
        shutil.copytree(self.dirFace_test_0, self.dirFace_retrieve_test_0)
        shutil.copytree(self.dirFace_test_1, self.dirFace_retrieve_test_1)


    def _evaluate_folder(self, model, folder_dir, label):
        """Evaluate all images in a folder and return losses and file names."""
        losses, names = [], []
        for img_name in os.listdir(folder_dir):
            if img_name.endswith(".jpg"):
                img = cv.imread(os.path.join(folder_dir, img_name))
                img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
                loss, _ = model.evaluate(img, np.array([label]), verbose=0)
                losses.append(loss)
                names.append(os.path.join(folder_dir, img_name))
        return losses, names


    def _train_new_model(self, base_model):
        """Clone and train model on new domain data."""
        new_model = clone_model(base_model)
        new_model.compile(optimizer=Adam(learning_rate=self.try_lr),
                          loss='binary_crossentropy', metrics=['accuracy'])
        new_model.set_weights(base_model.get_weights())

        datagen = ImageDataGenerator(rescale=1. / 255)
        train_gen = datagen.flow_from_directory(self.new_srcs[0],
                                                target_size=self.sz,
                                                batch_size=self.train_batch,
                                                class_mode='binary', shuffle=True)
        test_gen = datagen.flow_from_directory(self.new_srcs[1],
                                               target_size=self.sz,
                                               batch_size=self.test_batch,
                                               class_mode='binary', shuffle=True)

        new_model.fit(train_gen, epochs=self.EPOCH,
                      steps_per_epoch=train_gen.samples / self.train_batch,
                      validation_data=test_gen,
                      validation_steps=test_gen.samples / self.test_batch)
        return new_model


    def _get_top_interfered(self, pre_loss, post_loss, names):
        """Compute top interfered images based on pre- and post-loss difference."""
        scores = np.array(post_loss) - np.array(pre_loss)
        df = pd.DataFrame({'score': scores, 'name': names})
        df = df.sort_values(by='score', ascending=False)
        return df[:self.target_num]


    def _copy_selected(self, df_col, cls):
        """Copy images into dest folders, 1:4 test:train ratio."""
        picked = []
        for i, img in enumerate(df_col):
            is_test = i <= self.target_num / 5
            dst_idx = cls * 2 + (1 if is_test else 0)
            shutil.copy(img, self.dests[dst_idx])
            picked.append(img)
        return picked


    def _perform_ncm_selection(self, model, picked, ncm_fc1):
        """Select nearest samples by NCM (Nearest Class Mean)."""
        feature_network = Model(model.input, model.get_layer('fc1').output)

        # Extract features for both domains
        feats_0, names_0 = self._extract_features(feature_network, self.buffer_srcs[0], ncm_fc1)
        feats_1, names_1 = self._extract_features(feature_network, self.buffer_srcs[1], ncm_fc1)

        # Compute means
        mean_0, mean_1 = np.mean(feats_0, axis=0), np.mean(feats_1, axis=0)

        # Retrieve nearest samples
        self._select_nearest_samples(feats_0, names_0, mean_0, picked, cls=0)
        self._select_nearest_samples(feats_1, names_1, mean_1, picked, cls=1)


    def _extract_features(self, feature_network, folder_dir, ncm_fc1):
        """Get fc1 features for all images in folder."""
        feats, names = np.empty([0, ncm_fc1]), []
        for img_name in os.listdir(folder_dir):
            if img_name.endswith(".jpg"):
                img = cv.imread(os.path.join(folder_dir, img_name))
                img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
                feat = feature_network(img, training=False).numpy()
                feats = np.vstack((feats, feat))
                names.append(os.path.join(folder_dir, img_name))
        return feats, names


    def _select_nearest_samples(self, feats, names, mean_feat, picked, cls):
        """Pick nearest NCM samples not already picked."""
        dists = np.linalg.norm(feats - mean_feat, axis=1)
        sorted_idx = np.argsort(dists)
        cnt, j = 0, 0
        while cnt < self.ncm_num and j < len(sorted_idx):
            img = names[sorted_idx[j]]
            if img not in picked:
                is_test = cnt <= self.ncm_num / 5
                dst_idx = cls * 2 + (1 if is_test else 0)
                shutil.copy(img, self.dests[dst_idx])
                picked.append(img)
                cnt += 1
            j += 1


    def _copy_all_new_to_buffer(self):
        """Copy all new domain data into buffer initially."""
        for i, src in enumerate(self.new_train_srcs):
            dest = self.buffer_srcs[i]
            copy_tree(src, dest)


    def _add_new_data_to_buffer(self, target_num):
        """Add random new data to fill up buffer limit."""
        picked_files = []
        for i, src in enumerate(self.new_train_srcs):
            dest = self.buffer_srcs[i]
            self._random_copy(src, dest, target_num, picked_files)


    def _replace_buffer_with_new_data(self, target_num):
        """Replace some buffer data randomly with new domain data."""
        for i, src in enumerate(self.new_train_srcs):
            dest = self.buffer_srcs[i]
            picked_files = []
            self._random_replace(src, dest, target_num, picked_files)


    def _random_copy(self, src, dest, count, picked_files):
        """Randomly copy 'count' files from src → dest."""
        cnt = 0
        while cnt < count:
            f = random.choice(os.listdir(src))
            while f in picked_files:
                f = random.choice(os.listdir(src))
            shutil.copy(os.path.join(src, f), dest)
            picked_files.append(f)
            cnt += 1


    def _random_replace(self, src, dest, count, picked_files):
        """Randomly delete and replace 'count' files in dest with files from src."""
        cnt = 0
        while cnt < count:
            del_file = random.choice(os.listdir(dest))
            os.remove(os.path.join(dest, del_file))

            f = random.choice(os.listdir(src))
            while f in picked_files:
                f = random.choice(os.listdir(src))
            shutil.copy(os.path.join(src, f), dest)
            picked_files.append(f)
            cnt += 1


    def _show_analysis_prompt(self):
        """Display the analysis end prompt."""
        self.prompt = True
        print("end buffer update")

        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        font = ImageFont.truetype(self.font_path, 35)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
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


    def _reset_new_data_folders(self):
        """Remove old domain data and recreate directories."""
        for folder in self.new_data:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)


    def _setup_cv_window(self):
        """Initialize OpenCV window and mouse callback."""
        cv.namedWindow("subtle facial", cv.WINDOW_NORMAL)
        cv.resizeWindow('subtle facial', self.width, self.height)
        cv.moveWindow('subtle facial',
                    int((self.Width - self.width)/2),
                    int((self.Height - self.height)/2))
        cv.setMouseCallback('subtle facial', self.draw_circle_red)


    def _show_data_collection_prompt(self, cap):
        """Show instructions before starting data collection."""
        font = ImageFont.truetype(self.font_path, 35)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            imgPil = Image.fromarray(frame)
            draw = ImageDraw.Draw(imgPil)
            draw.text((10, 10),
                    "接下來會進行30回合收資料\n每回合都需點擊畫面中的圓點\n藍點請保持一般表情、紅點請做微表情\n準備好請點擊畫面",
                    fill=(219, 12, 242), font=font)
            frame = np.array(imgPil)
            cv.imshow("subtle facial", frame)

            if self.response1:
                break
            if cv.waitKey(1) & 0xFF == ord('q'):
                sys.exit(0)


    def _detect_face(self, face_mesh, frame):
        """Detect face landmarks using MediaPipe and return mesh points."""
        # Flip and convert to RGB
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        # Process frame with face mesh
        results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return frame, None

        # Convert landmarks to numpy array (pixel coordinates)
        mesh_points = np.array([
            np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
            for p in results.multi_face_landmarks[0].landmark
        ])

        return frame, mesh_points
    

    def _determine_head_tilt(self, mesh_points):
        """Determine head tilt and update DB if not settled yet."""
        if self.settled != 0:
            return

        # Compare the two positions to determine tilt
        left_area = (mesh_points[self.pos2[0]][0]-mesh_points[self.pos2[1]][0]) * (mesh_points[self.pos2[1]][1]-mesh_points[self.pos2[0]][1])
        right_area = (mesh_points[self.pos[1]][0]-mesh_points[self.pos[0]][0]) * (mesh_points[self.pos[1]][1]-mesh_points[self.pos[0]][1])

        if right_area < left_area:
            self.settled = 2  # tilt left
        else:
            self.settled = 1  # tilt right

        # Save to DB
        self.db.update_face_pos(self.settled)


    def _draw_face_guides(self, frame, mesh_points):
        """Draw rectangles and circles around the relevant facial points."""
        if self.settled == 1:
            # Draw rectangle
            cv.rectangle(
                frame,
                (mesh_points[self.pos[0]][0]-5, mesh_points[self.pos[0]][1]-5),
                (mesh_points[self.pos[1]][0]+5, mesh_points[self.pos[1]][1]+5),
                (0, 255, 0), 3
            )
            # Draw circles
            cv.circle(frame, (mesh_points[self.pos[0]][0]-5, mesh_points[self.pos[0]][1]-5), 5, (0,0,255), -1)
            cv.circle(frame, (mesh_points[self.pos[1]][0]+5, mesh_points[self.pos[1]][1]+5), 5, (0,0,255), -1)

        elif self.settled == 2:
            # Draw rectangle
            cv.rectangle(
                frame,
                (mesh_points[self.pos2[1]][0]-5, mesh_points[self.pos2[0]][1]-5),
                (mesh_points[self.pos2[0]][0]+5, mesh_points[self.pos2[1]][1]+5),
                (0, 255, 0), 3
            )
            # Draw circles
            cv.circle(frame, (mesh_points[self.pos2[0]][0]-5, mesh_points[self.pos2[0]][1]-5), 5, (0,0,255), -1)
            cv.circle(frame, (mesh_points[self.pos2[1]][0]+5, mesh_points[self.pos2[1]][1]+5), 5, (0,0,255), -1)
    

    def _prepare_face_mesh(self, cap):
        """Detect face and determine head tilt position."""
        with self.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame, mesh_points = self._detect_face(face_mesh, frame)
                if mesh_points is None:
                    continue

                self._determine_head_tilt(mesh_points)
                self._draw_face_guides(frame, mesh_points)

                font = ImageFont.truetype(self.font_path, 35)      
                imgPil = Image.fromarray(frame)                
                draw = ImageDraw.Draw(imgPil)                
                draw.text((10, 10), "請確認欲使用的微表情是否能被系統正確框出\n若無法 請將光源調亮\n準備好請點擊畫面", fill=(219, 12, 242), font=font)
                frame = np.array(imgPil)      
                cv.imshow("subtle facial", frame)
                
                if self.response2:
                    break
                if cv.waitKey(1) & 0xFF == ord('q'):
                    sys.exit(0)


    def _calculate_target_positions(self):
        """Calculate 9 target positions on the screen for data collection."""
        positions = []
        for y in range(int(self.height / 4), self.height, int(self.height / 4)):
            for x in range(int(self.width / 4), self.width, int(self.width / 4)):
                positions.append([x, y])
        return positions
    

    def _init_target_tracking(self):
        """Initialize target positions and counters for data collection."""
        pt_pos = random.randint(0, 8)
        prev_pos = pt_pos
        pt_cnt = np.zeros(9)
        return pt_pos, prev_pos, pt_cnt
    

    def _crop_face(self, frame, mesh_points):
        """Crop face region based on head tilt position."""
        try:
            if self.settled == 1:
                cropped_img = frame[mesh_points[self.pos[0]][1]:mesh_points[self.pos[1]][1],
                                    mesh_points[self.pos[0]][0]:mesh_points[self.pos[1]][0]].copy()
            elif self.settled == 2:
                cropped_img = frame[mesh_points[self.pos2[0]][1]:mesh_points[self.pos2[1]][1],
                                    mesh_points[self.pos2[1]][0]:mesh_points[self.pos2[0]][0]].copy()
            else:
                return None
            cropped_img = cv.resize(cropped_img, (self.sz[0], self.sz[1]))
            return cropped_img
        except Exception:
            return None
        

    def _process_targets_and_collect(self, frame, cropped_img, mid_pt, pt_pos, prev_pos, pt_cnt,
                                 control, show_green, exp_cnt, ctrl_cnt, round_counter, name_idx):
        """Process target display, user interaction, and collect images."""

        # Set mouse position for the current target
        self.mouseX = mid_pt[pt_pos][0]
        self.mouseY = mid_pt[pt_pos][1]

        # Control capture (red circle)
        if self.pressed1:
            if self.show_control:
                control += 1
                show_green += 1
            if show_green > 10:
                cv.circle(frame, (self.mouseX, self.mouseY), 30, (0, 0, 255), -1)

        # Subtle expression timing
        elif self.pressed2 and self.rest < 30:
            self.rest += 1
        elif self.pressed2 and self.rest >= 30:
            self.rest = 0
            show_green = 0
            self.pressed2 = False
            # Pick a new target position
            pt_pos = random.randint(0, 8)
            while pt_cnt[pt_pos] >= (self.EXP_TOTAL / (3 * 9) + 1) or pt_pos == prev_pos:
                pt_pos = random.randint(0, 8)
            pt_cnt[pt_pos] += 1
            prev_pos = pt_pos

        # Instruction display (blue circle)
        else:
            font = ImageFont.truetype(self.font_path, 35)
            imgPil = Image.fromarray(frame)
            draw = ImageDraw.Draw(imgPil)
            draw.text((mid_pt[pt_pos][0]+40,mid_pt[pt_pos][1]-30), "請保持一般表情\n點擊圓點", fill=(255, 0, 0), font=font)
            frame = np.array(imgPil)
            cv.circle(frame, (self.mouseX, self.mouseY), 30, (255, 0, 0), -1)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            sys.exit(0)

        if self.rest==1:
            round_counter+=1

        # Collect subtle expression images
        if self.rest in [1, 5, 9]:
            exp_cnt += 1
            folder = "./models/data/train/1/" if exp_cnt <= (self.EXP_TOTAL * 0.8) else "./models/data/test/1/"
            cv.imwrite(f"{folder}exp_{name_idx}-{exp_cnt}.jpg", cropped_img)

        if self.rest>1 and self.rest<10:
            font = ImageFont.truetype(self.font_path, 35)      
            imgPil = Image.fromarray(frame)                
            draw = ImageDraw.Draw(imgPil)                
            draw.text((mid_pt[pt_pos][0]+40,mid_pt[pt_pos][1]-30), "正在收微表情", fill=(0, 0, 255), font=font)
            frame = np.array(imgPil)

        # Collect control images
        if control in [1, 5, 9]:
            ctrl_cnt += 1
            folder = "./models/data/train/0/" if ctrl_cnt <= (self.CTRL_TOTAL * 0.8) else "./models/data/test/0/"
            cv.imwrite(f"{folder}ctrl_{name_idx}-{ctrl_cnt}.jpg", cropped_img)

        # Reset control display
        if control > 10:
            self.show_control = False
            self.show_instru = True
            control = 0

        if ctrl_cnt == self.CTRL_TOTAL and exp_cnt == self.EXP_TOTAL:
            return frame, pt_pos, prev_pos, pt_cnt, control, show_green, exp_cnt, ctrl_cnt, round_counter

        if self.show_control:
            font = ImageFont.truetype(self.font_path, 35)      
            imgPil = Image.fromarray(frame)                
            draw = ImageDraw.Draw(imgPil)                
            draw.text((mid_pt[pt_pos][0]+40,mid_pt[pt_pos][1]-30), "正在收一般表情", fill=(255, 0, 0), font=font)
            frame = np.array(imgPil)                              

        if self.show_instru:
            font = ImageFont.truetype(self.font_path, 35)      
            imgPil = Image.fromarray(frame)                
            draw = ImageDraw.Draw(imgPil)                
            draw.text((mid_pt[pt_pos][0]+40,mid_pt[pt_pos][1]-30), "請作微表情\n點擊圓點", fill=(0, 0, 255), font=font)
            frame = np.array(imgPil)

        return frame, pt_pos, prev_pos, pt_cnt, control, show_green, exp_cnt, ctrl_cnt, round_counter


    def _collect_faces(self, cap, name_idx):
        """Collect subtle expression and control images."""
        self.prompt = False

        mid_pt = self._calculate_target_positions()
        pt_pos, prev_pos, pt_cnt = self._init_target_tracking()
        control, ctrl_cnt, exp_cnt = 0, 0, 0
        show_green, round_counter = 0, 0

        with self.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame, mesh_points = self._detect_face(face_mesh, frame)
                if mesh_points is None:
                    continue

                cropped_img = self._crop_face(frame, mesh_points)
                if cropped_img is None:
                    continue

                # Process target display, user interaction, and image collection
                frame, pt_pos, prev_pos, pt_cnt, control, show_green, exp_cnt, ctrl_cnt, round_counter = \
                    self._process_targets_and_collect(
                        frame, cropped_img, mid_pt, pt_pos, prev_pos, pt_cnt,
                        control, show_green, exp_cnt, ctrl_cnt, round_counter, name_idx
                    )

                # Draw face guides
                self._draw_face_guides(frame, mesh_points)

                # Show round counter
                cv.putText(frame, f"Round {round_counter} / 30", (50, 50),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)

                # Display frame
                cv.imshow('subtle facial', frame)

                # Exit when collection completed
                if ctrl_cnt == self.CTRL_TOTAL and exp_cnt == self.EXP_TOTAL:
                    break

                # Exit on 'q'
                if cv.waitKey(1) & 0xFF == ord('q'):
                    sys.exit(0)


    def _show_collect_end_prompt(self):
        """Display final analysis prompt after collection."""
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        font = ImageFont.truetype(self.font_path, 35)
        self.prompt = True
        print("end collection")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            imgPil = Image.fromarray(frame)
            draw = ImageDraw.Draw(imgPil)
            draw.text((10, 10),
                    "結束收資料流程，接下來將進行分析\n約需等待三分鐘，請點擊畫面開始分析",
                    fill=(219, 12, 242), font=font)
            frame = np.array(imgPil)
            cv.imshow("subtle facial", frame)

            if self.response1:
                break
            if cv.waitKey(1) & 0xFF == ord('q'):
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







