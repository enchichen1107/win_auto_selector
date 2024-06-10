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

modelName = "init"
clicked = False

def on_click(x, y, button, pressed):
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
        conn = sqlite3.connect('./models/key_book.db')
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS keys (
            descrip text,
            hotkeys text
        )""")
        conn.commit()
        conn.close()
        
        # End process on exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_deleted)
        self.root.after_idle(self.process_queue)
        
        
        
        
        
    def __delete__(self):
        self.stop()
        self.process = None


       
    


        
    def btn_callback(self, key_text):
        global clicked
        clicked = False
        keys = key_text.split(',')

        with mouse.Listener(
        on_click=on_click) as listener:
            listener.join()

        while clicked==False:
            pass


        if len(keys)==1:
            # time.sleep(0.5)
            pyautogui.click(button='right')
        elif len(keys)==2:
            # time.sleep(0.5)
            # pyautogui.click(button='left')
            pyautogui.hotkey(keys[0],keys[1], interval=0.05)
        elif len(keys)==3:
            # time.sleep(0.5)
            # pyautogui.click(button='left')
            pyautogui.hotkey(keys[0],keys[1],keys[2], interval=0.05)
        elif len(keys)==4:
            # time.sleep(0.5)
            # pyautogui.click(button='left')
            pyautogui.hotkey(keys[0],keys[1],keys[2],keys[3], interval=0.05)
        elif len(keys)==5:
            # time.sleep(0.5)
            # pyautogui.click(button='left')
            pyautogui.hotkey(keys[0],keys[1],keys[2],keys[3],keys[4], interval=0.05)
        # if self.continue_subtle==0:
        #     self.subtle.attributes("-topmost", False)

        clicked = False
            
            
        
        
        

    def show(self):

        for widget in self.root.winfo_children():
            widget.destroy()
        
        conn = sqlite3.connect('./models/key_book.db')
        c = conn.cursor()

        
        listOfTables = c.execute(
            """SELECT * FROM sqlite_master WHERE type='table' 
            AND name='facials'; """).fetchall()
 
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

            # Create Edit Button
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
        
        
        conn = sqlite3.connect('./models/key_book.db')
        c = conn.cursor()
        c.execute("SELECT *, oid FROM keys")
        records = c.fetchall()
        cnt = len(records)

        # Create btns
        self.buttons = []

        for idx,record in enumerate(records):

            self.buttons.append(Button(self.subtle, text=str(record[0]),
                                      command=lambda record=record:self.btn_callback(record[1])))
            self.buttons[idx].grid(row=idx, column=0, pady=3)

        conn.commit()
        conn.close()
        
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

        conn = sqlite3.connect('./models/key_book.db')
        c = conn.cursor()      
        c.execute("DROP TABLE IF EXISTS facials;")
        conn.commit()
        c.execute("DROP TABLE IF EXISTS domains;")
        conn.commit()
        conn.close()

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

        conn = sqlite3.connect('./models/key_book.db')
        c = conn.cursor()

        record_id = self.delete_box.get()

        c.execute("""UPDATE keys SET
            descrip = :descrip,
            hotkeys = :hotkeys

            WHERE oid = :oid""",
            {
            'descrip': self.descrip_editor.get(),
            'hotkeys': self.hotkeys_editor.get(),
            'oid': record_id
            })


        conn.commit()
        conn.close()

        self.editor.withdraw()
        self.root.deiconify()
        self.show()
        
        
        

    def edit(self):
        
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
        

        conn = sqlite3.connect('./models/key_book.db')
        c = conn.cursor()
        c.execute("SELECT *, oid FROM keys")
        records = c.fetchall()
        cnt = len(records)
        
        self.descrips = []
        self.options = []
        for idx,record in enumerate(records):
            self.descrips.append(Label(self.editor, text=str(record[2]) + " " + str(record[0]) + ": " + str(record[1])))
            self.descrips[idx].grid(row=idx+1, column=0, columnspan=2, pady=1)
            self.options.append(record[2])
            
        conn.commit()
        conn.close()
        
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
        # self.editor.withdraw()
        self.root.deiconify()
        self.show()     
        
        
    
    
    # 
    # Create Create function to create a record
    def create(self):

        self.editor.withdraw()

        if hasattr(self,'creator')==True:           
            self.creator.destroy()
            delattr(self, "creator")

        self.creator = Toplevel(self.editor)
        self.creator.title('Create A Setting')
        self.creator.geometry("300x300")

        for widget in self.creator.winfo_children():
            widget.destroy()
        
        self.creator.geometry('+{}+{}'.format(int(self.width/2-100),int(self.height/2-100)))
        self.creator.attributes('-alpha',0.8)
        self.creator.protocol("WM_DELETE_WINDOW", self.create_window_delete)
        self.creator.deiconify()


        # Create Text Boxes
        self.descrip = Entry(self.creator, width=20)
        self.descrip.grid(row=0, column=1, padx=10, pady=(10, 0))

        self.hotkeys= Entry(self.creator, width=20)
        self.hotkeys.grid(row=1, column=1)


        # Create Text Box Labels
        self.descrip_label = Label(self.creator, text="功能名稱")
        self.descrip_label.grid(row=0, column=0, pady=(20, 0))
        self.hotkeys_label = Label(self.creator, text="快捷鍵組合")
        self.hotkeys_label.grid(row=1, column=0)


        # Create Submit Button
        self.submit_btn = Button(self.creator, text="送出", command=self.submit)
        self.submit_btn.grid(row=2, column=0, columnspan=2, pady=10, ipadx=70)
        
        if hasattr(self,'dialogue_text'):
            self.dialogue = Label(self.creator, text=self.dialogue_text)
            self.dialogue.grid(row=3, column=0, columnspan=2,  pady=2)


        
    # Create Function to Delete A Record
    def delete(self):

        conn = sqlite3.connect('./models/key_book.db')
        c = conn.cursor()
        c.execute("DELETE from keys WHERE oid = " + self.delete_box.get())

        conn.commit()
        conn.close()

        self.edit()
        
               
        
        
        
    def create_window_delete(self):
   
        if hasattr(self,'creator'):
            self.creator.destroy()
            delattr(self, "creator")
        if hasattr(self,'dialogue_text'):
            delattr(self, "dialogue_text")
        self.editor.deiconify()
        self.edit()
        




    # Create Submit Function For database
    def submit(self):

        description = self.descrip.get()
        hotkey = self.hotkeys.get()
        
        if hasattr(self,'dialogue'):
            self.dialogue.destroy()
        
        if description=="" or hotkey=="":                    
            self.dialogue_text = "輸入不得為空白"
        elif ' ' in hotkey:
            self.dialogue_text = "快捷鍵組合不得含空白' '\n若需空白鍵請填space"
        elif ',' not in hotkey:
            self.dialogue_text = "快捷鍵組合請以半形逗號,做按鍵分割"
        else:
            keys = hotkey.split(',')
            success = True
            for k in keys:
                if k not in pyautogui.KEYBOARD_KEYS:
                    success = False
                    break
            if success:        
                conn = sqlite3.connect('./models/key_book.db')
                c = conn.cursor()
                c.execute("INSERT INTO keys VALUES (:descrip, :hotkeys)",
                        {
                            'descrip': description,
                            'hotkeys': hotkey
                        })

                conn.commit()
                conn.close()
                self.dialogue_text = "成功加入快捷鍵"
            else:
                self.dialogue_text = "按鍵輸入錯誤\n請查照按鍵名稱對照表"
                               

        # Clear The Text Boxes
        self.descrip.delete(0, END)
        self.hotkeys.delete(0, END)
        self.create()




        
    def submit_face(self):

        conn = sqlite3.connect('./models/key_book.db')
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS facials (
            facePart text
            )""")
        conn.commit()


        c.execute("INSERT INTO facials VALUES (:facePart)",
                {
                    'facePart': self.faceVar.get()
                })
        conn.commit()


        conn.close()

        for widget in self.root.winfo_children():
            widget.destroy()
        self.show()

        



    def mainloop(self, *args):
        self.show()
        return self.root.mainloop(*args)
        

        
        
    def start(self):

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
        # self.show()   






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
        
        



