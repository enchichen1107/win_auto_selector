from tkinter import *
import pyautogui
import queue
from lib.train import TrainProcess
from PIL import ImageTk
import multiprocessing
from pynput import mouse
from db.handler import DBhandler
from gui.panels import Panel
from gui.actions import Action
from config import MODEL_NAME

clicked = False

def on_click(x, y, button, pressed):
    ''' mouse left clicked listener '''
    if pressed: 
        pass
    elif not pressed:
        global clicked
        clicked = True
            # Stop listener
        return False



class MainWindow(object):   
    
    def __init__(self):
        self.process = None
        self.queue = multiprocessing.Queue(1)
        self.root = Tk()
        self.root.title('Auto Selector')
        self.root.geometry("200x200")
        self.root.attributes('-alpha',0.8)
        self.vkblogo = ImageTk.PhotoImage(file="./assets/auto_selector.ico")
        self.root.iconphoto(True, self.vkblogo)
        self.continue_subtle = 0
        self.model_name = MODEL_NAME
        
        x, y = pyautogui.position()
        self.root.geometry('+{}+{}'.format(x+10,y-100))
        s_size = pyautogui.size()
        self.width = s_size[0]
        self.height = s_size[1]
        
        # Database
        self.db = DBhandler()
        self.db.create_keys_table()
        
        # End process on exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_deleted)
        self.root.after_idle(self.process_queue)
             
        
        
    def __delete__(self):
        self.stop()
        self.process = None


        
    def btn_callback(self, key_text):
        '''  button callback for hot key panel '''

        global clicked
        clicked = False
        keys = key_text.split(',')

        with mouse.Listener(
        on_click=on_click) as listener:
            listener.join()

        # apply hot keys until left click is detected
        while clicked==False:
            pass


        if len(keys)==1:
            pyautogui.click(button='right')
        elif len(keys)==2:
            pyautogui.hotkey(keys[0],keys[1], interval=0.05)
        elif len(keys)==3:
            pyautogui.hotkey(keys[0],keys[1],keys[2], interval=0.05)
        elif len(keys)==4:
            pyautogui.hotkey(keys[0],keys[1],keys[2],keys[3], interval=0.05)
        elif len(keys)==5:
            pyautogui.hotkey(keys[0],keys[1],keys[2],keys[3],keys[4], interval=0.05)

        clicked = False
                
        

    def show(self):
        '''  show root window '''
        Panel.show_main_menu(self, self.db)


        
    def show_subtle_continue(self):
        self.continue_subtle = 1
        self.show_subtle_panel()
        

    
    def show_subtle_panel(self):
        '''  show detect subtle and hot key window '''
        Panel.show_subtle_hot_key(self, self.db)


    
    def subtle_backto(self):
        self.stop()



    def subtle_hide(self):
        if hasattr(self, 'subtle'):
            self.subtle.withdraw()



    def restore_init(self):
        '''  restore all settings '''
        Action.restore_all(self, self.db)

        # self.db.restore_all()

        # if os.path.isfile('./models/'+self.model_name+'.h5'):
        #     os.remove('./models/'+modelName+'.h5')
        # if os.path.exists('./models/retrieve'):
        #         shutil.rmtree('./models/retrieve')
        # if os.path.exists('./models/data'):
        #         shutil.rmtree('./models/data')
        # if os.path.exists('./models/buffer'):
        #         shutil.rmtree('./models/buffer')
        
        # self.restore_backto()





    def restore_backto(self):
        self.restore.destroy()
        delattr(self, 'restore')
        self.root.deiconify()
        self.show()



    def show_restore(self):
        '''  confirm window for restore operation '''
        Panel.show_restore(self)
        
        
        
    def update(self):
        '''  update hot keys '''
        self.db.update_key(self.delete_box.get(), self.descrip_editor.get(), self.hotkeys_editor.get())
        self.editor.withdraw()
        self.root.deiconify()
        self.show()
        
        

    def edit(self):
        ''' show edit panel '''
        Panel.show_edit(self, self.db)
        

    
    def backto(self):
        self.editor.destroy()
        delattr(self, "editor")
        self.root.deiconify()
        self.show()     
               
    
    
    def create(self):
        '''  create a record '''
        Panel.show_create(self)

        

    def delete(self):
        '''  delete a record '''

        self.db.delete_key(self.delete_box.get())
        self.edit()
                             
        
        
    def create_window_delete(self):
        ''' delete create panel '''

        if hasattr(self,'creator'):
            self.creator.destroy()
            delattr(self, "creator")
        if hasattr(self,'dialogue_text'):
            delattr(self, "dialogue_text")
        self.editor.deiconify()
        self.edit()
        



    def submit(self):
        ''' submit created hot key '''
        Action.submit_create_key(self)



    def record_hotkey(self):
        """Record a hotkey sequence"""
        Action.record_hotkey(self)



        
    def submit_face(self):
        '''  submit chosen facial parts '''

        self.db.create_facials_table()
        self.db.insert_face(self.faceVar.get())

        for widget in self.root.winfo_children():
            widget.destroy()
        self.show()
       


    def mainloop(self, *args):
        self.show()
        return self.root.mainloop(*args)
        

        
        
    def start_detect(self):
        '''  start to detect subtle facial expression '''
        Action.detect_subtle_show_hotkey(self)



    def remind_backto(self):
        self.stop()
        self.remind.destroy()
        self.root.deiconify()



    def start_train(self):
        self.stop()
        self.train_process = TrainProcess()
        self.train_process.start()



    def stop(self):
        self.continue_subtle = 0
        if not self.process:
            pass
        else:
            self.process.stop()
        if hasattr(self,'subtle')==True:
            self.subtle.destroy()
            delattr(self, "subtle")
        self.root.deiconify()  



    def process_queue(self):
        # process queue asynchronously
        try:
            while True:
                data = self.queue.get_nowait()
                if data:
                    print(data)
                    self.subtle.deiconify()
                    self.show_subtle_panel()
                    break
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue) # 100 ms



    def on_window_deleted(self):
        self.stop()
        self.root.destroy()
        
        



