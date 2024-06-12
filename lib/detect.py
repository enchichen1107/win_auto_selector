import multiprocessing
import cv2 as cv
import numpy as np
import ctypes
import time
from tensorflow.keras.models import load_model
import mediapipe as mp
import sqlite3



modelName = "init"
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
fontpath = "./assets/NotoSansTC-Medium.otf"





class Worker(object):
    
    
    def __init__(self, queue):
        self._is_alive = multiprocessing.Value(ctypes.c_bool, True)
        self.queue     = queue
        self.width = 300
        self.height = 300      
        
    def get_is_alive(self):
        with self._is_alive.get_lock():
            return self._is_alive.value
        
    def set_is_alive(self, value):
        with self._is_alive.get_lock():
            self._is_alive.value = value

    is_alive = property(get_is_alive, set_is_alive)    
        
    def start(self):
        self.run()       
        
    def stop(self): 
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        cap.release()
        cv.destroyAllWindows()


    def run(self):

        # init opencv settings
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        cv.namedWindow("subtle facial", cv.WINDOW_NORMAL)
        cv.resizeWindow('subtle facial', self.width, self.height)
        cv.moveWindow('subtle facial',100,50)
                  
        
        self.model = load_model('./models/'+modelName+'.h5')
   
        # get facial part
        conn = sqlite3.connect('./models/key_book.db')
        c = conn.cursor()
        c.execute("SELECT facePart FROM facials")
        record = c.fetchall()
        self.facePart = record[0][0]

        # get head tilted position
        c.execute("SELECT pos FROM positions")
        record = c.fetchall()
        conn.commit()
        self.settled = record[0][0]
        conn.close()

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
        
        # start real time capture and detect
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:

            prediction = 0
            success_cnt = 0
            while True:
                prediction = 0
                ret, frame = cap.read()
                frame = cv.flip(frame, 1)
                if not ret:
                    break
                if not self.is_alive:
                    break
    
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                img_h, img_w = frame.shape[:2]
                results = face_mesh.process(rgb_frame)
                if not results.multi_face_landmarks:
                    continue
                else:
                        
                    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])


                    # crop img
                    if self.settled == 1:
                        cropped_img = frame[mesh_points[self.pos[0]][1]:mesh_points[self.pos[1]][1],mesh_points[self.pos[0]][0]:mesh_points[self.pos[1]][0]].copy()
                    elif self.settled == 2:
                        cropped_img = frame[mesh_points[self.pos2[0]][1]:mesh_points[self.pos2[1]][1],mesh_points[self.pos2[1]][0]:mesh_points[self.pos2[0]][0]].copy()
                    try:
                        cropped_img = cv.resize(cropped_img,(self.sz[0],self.sz[1]))
                    except Exception as e:
                        continue

                    # model predict
                    img_array = np.expand_dims(cropped_img, axis=0)
                    img = img_array.astype(np.float32) / 255.0
                    prediction = self.model.predict(img,verbose=0)
                    prediction = prediction[0][0]


                    if prediction>0.55:
                        if success_cnt==5: # logic for blocking false alarm
                            self.queue.put(1)
                            cv.putText(frame, "DETECTED", (100,100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv.LINE_AA)
                            print("gotcha")
                            print(prediction)
                            success_cnt = 0
                        else:
                            success_cnt+=1


                key = cv.waitKey(1)
                if key == ord('q'):
                    break

                cv.imshow("subtle facial", frame)




                                             
class Process(object):

    def __init__(self, queue):
        self.worker  = None
        self.process = None
        self.queue   = queue

    def __delete__(self):
        self.stop()

    def start(self):
        self.stop()
        self.worker  = Worker(self.queue)
        self.process = multiprocessing.Process(target=self.worker.start)
        self.process.start()

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







