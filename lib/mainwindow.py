from tkinter import *
import pyautogui
import time
import sqlite3
import os.path
import queue
from   lib.detect import Process
from lib.train import TrainProcess
import shutil
from PIL import ImageTk
import multiprocessing
from pynput import mouse
from pynput import keyboard
from db.handler import DBhandler

modelName = "init"
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

        for widget in self.root.winfo_children():
            widget.destroy()

        # if hasn't chosen facial part, then choose it first
        listOfTables = self.db.get_face_tables()
 
        if listOfTables == []:
        
            welcome = Label(self.root, text="請擇一微表情部位")
            welcome.grid(row=0, column=0)
            
            self.faceVar = StringVar()
            radiobutton1 = Radiobutton(self.root, text='眉毛',
                                variable=self.faceVar, value='brow')
            radiobutton1.grid(row=1, column=0, pady=2)
            radiobutton2 = Radiobutton(self.root, text='鼻子',
                                variable=self.faceVar, value='nose')
            radiobutton2.grid(row=2, column=0, pady=2)
            radiobutton3 = Radiobutton(self.root, text='嘴唇',
                                variable=self.faceVar, value='lips')
            radiobutton3.grid(row=3, column=0, pady=2)
            check_button = Button(self.root, text='確認送出', command=self.submit_face)
            check_button.grid(row=4, column=0, pady=2)
            
            self.root.grid_columnconfigure((0), weight=1)
            self.root.grid_rowconfigure((0,1,2,3,4), weight=1)
            
        
        # Create Button
        else:
        
            self.start_btn = Button(self.root, text="使用微表情", command=self.start)
            self.start_btn.grid(row=0, column=0, pady=5, ipadx=10)
            
            self.train_btn = Button(self.root, text="訓練模型", command=self.start_train)
            self.train_btn.grid(row=1, column=0, pady=5, ipadx=10)

            self.continue_btn = Button(self.root, text="連續用快捷鍵", command=self.show_subtle_continue)
            self.continue_btn.grid(row=2, column=0, pady=5, ipadx=10)

            self.create_btn = Button(self.root, text="編輯快捷鍵", command=self.edit)
            self.create_btn.grid(row=3, column=0, pady=5, ipadx=10)
            
            self.restore_btn = Button(self.root, text="恢復初始狀態", command=self.show_restore)
            self.restore_btn.grid(row=4, column=0, pady=5, ipadx=10)
            
            self.root.grid_columnconfigure((0), weight=1)
            self.root.grid_rowconfigure((0,1,2,3,4), weight=1)


        
        
        
    def show_subtle_continue(self):
        self.continue_subtle = 1
        self.show_subtle_panel()
        


    
    
    def show_subtle_panel(self):
        
        self.root.withdraw()
        
        if hasattr(self,'subtle')==False:
            self.subtle = Toplevel(self.root)  
            self.subtle.title('Short cut panel')
            self.subtle.geometry("150x250")
        
        self.subtle.attributes("-topmost", True)
        
        x, y = pyautogui.position()
        self.subtle.geometry('+{}+{}'.format(x+10,y-100))
        self.subtle.attributes('-alpha',0.8)
        self.subtle.deiconify()
        
        # get all keys
        records = self.db.get_all_keys()
        cnt = len(records)

        # Create btns
        self.buttons = []

        for idx,record in enumerate(records):

            self.buttons.append(Button(self.subtle, text=str(record[0]),
                                      command=lambda record=record:self.btn_callback(record[1])))
            self.buttons[idx].grid(row=idx, column=0, pady=3)
        
        # Create backto Button
        if self.continue_subtle == 0:

            self.hide_btn = Button(self.subtle, fg="#1152f7", text="隱藏面板", command=self.subtle_hide)
            self.hide_btn.grid(row=cnt, column=0, pady=3, ipadx=10)

            self.delete_btn = Button(self.subtle, fg="#1152f7", text="停止偵測", command=self.subtle_backto)
            self.delete_btn.grid(row=cnt+1, column=0, pady=3, ipadx=10)


            self.subtle.grid_columnconfigure((0), weight=1)
            for i in range(0,cnt+2):
                self.subtle.grid_rowconfigure(i, weight=1)
        else:
            self.subtle.grid_columnconfigure((0), weight=1)
            for i in range(0,cnt):
                self.subtle.grid_rowconfigure(i, weight=1)
        
        self.subtle.protocol("WM_DELETE_WINDOW", self.subtle_backto)




    
    def subtle_backto(self):
        self.stop()




    def subtle_hide(self):
        if hasattr(self, 'subtle'):
            self.subtle.withdraw()



    def restore_init(self):
        '''  restore all settings '''
        self.db.restore_all()

        if os.path.isfile('./models/'+modelName+'.h5'):
            os.remove('./models/'+modelName+'.h5')
        if os.path.exists('./models/retrieve'):
                shutil.rmtree('./models/retrieve')
        if os.path.exists('./models/data'):
                shutil.rmtree('./models/data')
        if os.path.exists('./models/buffer'):
                shutil.rmtree('./models/buffer')
        
        self.restore_backto()





    def restore_backto(self):
        self.restore.destroy()
        delattr(self, 'restore')
        self.root.deiconify()
        self.show()





    def show_restore(self):
        '''  confirm window for restore operation '''

        self.root.withdraw()
        
        if hasattr(self,'restore')==False :
            self.restore = Toplevel(self.root) 
            self.restore.title('Restore panel')
            self.restore.geometry("150x150")
        
        self.restore.geometry('+{}+{}'.format(int(self.width/2-100),int(self.height/2-100)))
        self.restore.attributes('-alpha',0.8)
        self.restore.deiconify()

        self.confirm_btn = Button(self.restore, text="確定恢復初始設定", command=self.restore_init)
        self.confirm_btn.grid(row=0, column=0, pady=2, ipadx=10)

        self.back_btn = Button(self.restore, text="回主選單", command=self.restore_backto)
        self.back_btn.grid(row=1, column=0, pady=2, ipadx=10)

        self.restore.grid_columnconfigure((0), weight=1)
        self.restore.grid_rowconfigure((0,1), weight=1)
     
        self.restore.protocol("WM_DELETE_WINDOW", self.restore_backto)
        
        
        
        
        
    def update(self):
        '''  update hot keys '''
        self.db.update_key(self.delete_box.get(), self.descrip_editor.get(), self.hotkeys_editor.get())

        self.editor.withdraw()
        self.root.deiconify()
        self.show()
        
        
        

    def edit(self):
        ''' show edit panel '''
        
        self.root.withdraw()
        
        
        if hasattr(self,'editor')==False:
            self.editor = Toplevel(self.root)
            self.editor.title('Update A Setting')
            self.editor.geometry("270x350")
        
        for widget in self.editor.winfo_children():
            widget.destroy()
            

        self.editor.geometry('+{}+{}'.format(int(self.width/2-100),int(self.height/2-100)))
        self.editor.attributes('-alpha',0.8)
        self.editor.deiconify()
        
        records = self.db.get_all_keys()
        cnt = len(records)
        
        self.descrips = []
        self.options = []
        for idx,record in enumerate(records):
            self.descrips.append(Label(self.editor, text=str(record[2]) + " " + str(record[0]) + ": " + str(record[1])))
            self.descrips[idx].grid(row=idx+1, column=0, columnspan=2, pady=1)
            self.options.append(record[2])
        
        # Create Edit Button
        self.create_btn = Button(self.editor, text="新增快捷鍵", command=self.create)
        self.create_btn.grid(row=cnt+1, column=0, columnspan=2, pady=5, ipadx=70)

        # Create Text Box Labels
        self.delete_box_label = Label(self.editor, text="選擇 快捷鍵編號")
        self.delete_box_label.grid(row=cnt+2, column=0, pady=5)

        # Create Text Boxes
        self.delete_box = StringVar(self.editor)
        if len(self.options)>0:
            self.delete_box.set(self.options[0]) # default value
            self.menu = OptionMenu(self.editor, self.delete_box, *self.options)
        else:
            self.delete_box.set("尚未設定快捷鍵")
            self.menu = OptionMenu(self.editor, self.delete_box, "請先設定快捷鍵")    
        self.menu.grid(row=cnt+2, column=1, pady=5)

        # Create A Delete Button
        self.delete_btn = Button(self.editor, text="刪除快捷鍵", command=self.delete)
        self.delete_btn.grid(row=cnt+3, column=0, columnspan=2, pady=5, ipadx=70)
        
        # Create A backto Button
        self.back_btn = Button(self.editor, text="回主選單", command=self.backto)
        self.back_btn.grid(row=cnt+4, column=0, columnspan=2, pady=5, ipadx=70)
        
        self.editor.grid_columnconfigure((0,1), weight=1)
        for i in range(0,cnt+5):
            self.editor.grid_rowconfigure(i, weight=1)

        self.editor.protocol("WM_DELETE_WINDOW", self.backto)


    
    def backto(self):
        self.editor.destroy()
        delattr(self, "editor")
        self.root.deiconify()
        self.show()     
        
        
    
    
    def create(self):
        '''  create a record '''

        self.editor.withdraw()

        if hasattr(self,'creator')==True:           
            self.creator.destroy()
            delattr(self, "creator")

        self.creator = Toplevel(self.editor)
        self.creator.title('Create A Setting')
        self.creator.geometry("300x250")  

        for widget in self.creator.winfo_children():
            widget.destroy()
        
        self.creator.geometry('+{}+{}'.format(int(self.width/2-100),int(self.height/2-100)))
        self.creator.attributes('-alpha',0.8)
        self.creator.grid_columnconfigure(0, weight=1)
        self.creator.grid_columnconfigure(1, weight=1)
        for i in range(0, 4): 
            self.creator.grid_rowconfigure(i, weight=1)
        self.creator.protocol("WM_DELETE_WINDOW", self.create_window_delete)
        self.creator.deiconify()


        # Create Text Boxes
        self.descrip = Entry(self.creator, width=30)
        self.descrip.grid(row=0, column=1, padx=10, pady=(10, 0))

        self.hotkeys= Entry(self.creator, width=30)
        self.hotkeys.grid(row=1, column=1, padx=10, pady=(10, 0))


        # Create Text Box Labels
        self.descrip_label = Label(self.creator, text="功能名稱")
        self.descrip_label.grid(row=0, column=0, padx=10, pady=(20, 0))
        self.hotkeys_label = Label(self.creator, text="快捷鍵組合")
        self.hotkeys_label.grid(row=1, column=0, padx=10)


        # Record Hotkey Button
        self.record_btn = Button(self.creator, text="偵測快捷鍵", command=self.record_hotkey)
        self.record_btn.grid(row=2, column=0, columnspan=2, pady=5, ipadx=70)

        # Submit Button
        self.submit_btn = Button(self.creator, text="送出", command=self.submit)
        self.submit_btn.grid(row=3, column=0, columnspan=2, pady=5, ipadx=70)

        
        if hasattr(self,'dialogue_text'):
            self.dialogue = Label(self.creator, text=self.dialogue_text)
            self.dialogue.grid(row=4, column=0, columnspan=2,  pady=5)


        

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
        ''' submit created record '''

        description = self.descrip.get()
        hotkey = self.hotkeys.get()
        
        if hasattr(self,'dialogue'):
            self.dialogue.destroy()
        
        # logic for validate input
        if description=="" or hotkey=="":                    
            self.dialogue_text = "輸入不得為空白"
        elif ' ' in hotkey:
            self.dialogue_text = "快捷鍵組合不得含空白' '\n若需空白鍵請填space"
        elif ',' not in hotkey:
            self.dialogue_text = "快捷鍵組合請以半形逗號,做按鍵分割"
        else:
            keys = hotkey.split(',')
            success = all(k in pyautogui.KEYBOARD_KEYS for k in keys)

            if success:   
                self.db.insert_key(description, hotkey)     
                self.dialogue_text = "成功加入快捷鍵"
            else:
                self.dialogue_text = "按鍵輸入錯誤\n請查照按鍵名稱對照表"
                               

        # Clear The Text Boxes
        self.descrip.delete(0, END)
        self.hotkeys.delete(0, END)
        self.create()


    def record_hotkey(self):
        """Record a hotkey sequence using pynput"""

        self.hotkeys.delete(0, END)
        if hasattr(self, 'dialogue'):
            self.dialogue.destroy()

        self.dialogue = Label(self.creator, text="請按下快捷鍵組合（按 Enter 結束）…")
        self.dialogue.grid(row=4, column=0, columnspan=2, pady=5, ipadx=70)
        self.creator.update()

        detected_keys = []

        stop_keys = {"enter", "return", "esc", "escape"}

        normalize_map = {
            "ctrl_l": "ctrl", "ctrl_r": "ctrl",
            "shift_l": "shift", "shift_r": "shift",
            "alt_l": "alt", "alt_r": "alt",
            "cmd": "win", "cmd_l": "win", "cmd_r": "win",
            "option_l": "option", "option_r": "option",
            "windows": "win", "super": "win",
        }

        def normalize_key(k: str) -> str:
            k = k.lower()
            return normalize_map.get(k, k)

        def normalize_key(k: str) -> str:
            k = k.lower()
            k = normalize_map.get(k, k)
            if k not in pyautogui.KEYBOARD_KEYS:
                return None
            return k

        def on_press(key):
            try:
                k = key.char.lower() if key.char else key.name
            except AttributeError:
                k = str(key).replace("Key.", "")
            k = normalize_key(k)

            if k is None or k in stop_keys:
                return

            detected_keys.append(k)
            print(f"Pressed: {k}")


        def on_release(key):
            try:
                k = key.char.lower() if key.char else key.name
            except AttributeError:
                k = str(key).replace("Key.", "")
            k = normalize_key(k)

            if k in stop_keys:
                return False 

        self.creator.focus_set()

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        listener.join()

        if hasattr(self, "dialogue"):
            self.dialogue.destroy()

        hotkey_str = ",".join(detected_keys)
        self.hotkeys.insert(0, hotkey_str)

        self.dialogue = Label(self.creator, text=f"已偵測到: {hotkey_str or '(無)'}")
        self.dialogue.grid(row=4, column=0, columnspan=2, pady=5)




        
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
        

        
        
    def start(self):
        '''  start to detect subtle facial expression '''

        # remind create model first
        if not os.path.isfile('./models/'+modelName+'.h5'):
            self.root.withdraw()
        
            self.remind = Toplevel(self.root)
    
            self.remind.title('Reminder')
            self.remind.geometry("100x150")

            x, y = pyautogui.position()
            self.remind.geometry('+{}+{}'.format(x+10,y-100))
            self.remind.attributes('-alpha',0.8)
            
            dialogue1 = Label(self.remind, text="尚未建立模型")
            dialogue1.grid(row=0, column=0, pady=2)
            dialogue2 = Label(self.remind, text="請先回主選單")
            dialogue2.grid(row=1, column=0, pady=2)
            dialogue3 = Label(self.remind, text="點選模型訓練")
            dialogue3.grid(row=2, column=0, pady=2)
            btn = Button(self.remind, text="回主選單", command=self.remind_backto)
            btn.grid(row=3, column=0, pady=2, ipadx=10)
            self.remind.grid_columnconfigure((0), weight=1)
            self.remind.grid_rowconfigure((0,1,2,3), weight=1)

            self.remind.protocol("WM_DELETE_WINDOW", self.remind_backto)
       
        # get worker for detect subtle facial expression
        else:
            self.stop()
            self.root.withdraw()
            self.subtle = Toplevel(self.root)
            self.subtle.withdraw()
            self.process = Process(self.queue)
            self.process.start()
            




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
        # when gui is idle
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
        
        



