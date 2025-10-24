from tkinter import Button, Label, Toplevel, END
import pyautogui
import os
import shutil
from pynput import keyboard
from lib.detect import Process


class Action:

    @staticmethod
    def restore_all(main_win, db_handler):
        """Restore all settings"""
        db_handler.restore_all()

        if os.path.isfile('./models/'+main_win.model_name+'.h5'):
            os.remove('./models/'+main_win.model_name+'.h5')
        if os.path.exists('./models/retrieve'):
                shutil.rmtree('./models/retrieve')
        if os.path.exists('./models/data'):
                shutil.rmtree('./models/data')
        if os.path.exists('./models/buffer'):
                shutil.rmtree('./models/buffer')
        
        main_win.restore_backto()



    @staticmethod
    def submit_create_key(main_win):
        """Submit the created hot key"""
        description = main_win.descrip.get()
        hotkey = main_win.hotkeys.get()
        
        if hasattr(main_win,'dialogue'):
            main_win.dialogue.destroy()
        
        # logic for validate input
        if description=="" or hotkey=="":                    
            main_win.dialogue_text = "輸入不得為空白"
        elif ' ' in hotkey:
            main_win.dialogue_text = "快捷鍵組合不得含空白' '\n若需空白鍵請填space"
        elif ',' not in hotkey:
            main_win.dialogue_text = "快捷鍵組合請以半形逗號,做按鍵分割"
        else:
            keys = hotkey.split(',')
            success = all(k in pyautogui.KEYBOARD_KEYS for k in keys)

            if success:   
                main_win.db.insert_key(description, hotkey)     
                main_win.dialogue_text = "成功加入快捷鍵"
            else:
                main_win.dialogue_text = "按鍵輸入錯誤\n請查照按鍵名稱對照表"
                               

        # Clear The Text Boxes
        main_win.descrip.delete(0, END)
        main_win.hotkeys.delete(0, END)
        main_win.create()



    @staticmethod
    def record_hotkey(main_win):
        """Detect hot key typed by user and record it"""
        main_win.hotkeys.delete(0, END)
        if hasattr(main_win, 'dialogue'):
            main_win.dialogue.destroy()

        main_win.dialogue = Label(main_win.creator, text="請按下快捷鍵組合（按 Enter 結束）")
        main_win.dialogue.grid(row=4, column=0, columnspan=2, pady=5, ipadx=70)
        main_win.creator.update()

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

        main_win.creator.focus_set()

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        listener.join()

        if hasattr(main_win, "dialogue"):
            main_win.dialogue.destroy()

        hotkey_str = ",".join(detected_keys)
        main_win.hotkeys.insert(0, hotkey_str)

        main_win.dialogue = Label(main_win.creator, text=f"已偵測到: {hotkey_str or '(無)'}")
        main_win.dialogue.grid(row=4, column=0, columnspan=2, pady=5)



    @staticmethod
    def detect_subtle_show_hotkey(main_win):
        # remind create model first
        if not os.path.isfile('./models/'+main_win.model_name+'.h5'):
            main_win.root.withdraw()
        
            main_win.remind = Toplevel(main_win.root)
    
            main_win.remind.title('Reminder')
            main_win.remind.geometry("100x150")

            x, y = pyautogui.position()
            main_win.remind.geometry('+{}+{}'.format(x+10,y-100))
            main_win.remind.attributes('-alpha',0.8)
            
            dialogue1 = Label(main_win.remind, text="尚未建立模型")
            dialogue1.grid(row=0, column=0, pady=2)
            dialogue2 = Label(main_win.remind, text="請先回主選單")
            dialogue2.grid(row=1, column=0, pady=2)
            dialogue3 = Label(main_win.remind, text="點選模型訓練")
            dialogue3.grid(row=2, column=0, pady=2)
            btn = Button(main_win.remind, text="回主選單", command=main_win.remind_backto)
            btn.grid(row=3, column=0, pady=2, ipadx=10)
            main_win.remind.grid_columnconfigure((0), weight=1)
            main_win.remind.grid_rowconfigure((0,1,2,3), weight=1)

            main_win.remind.protocol("WM_DELETE_WINDOW", main_win.remind_backto)
       
        # get worker for detect subtle facial expression
        else:
            main_win.stop()
            main_win.root.withdraw()
            main_win.subtle = Toplevel(main_win.root)
            main_win.subtle.withdraw()
            main_win.process = Process(main_win.queue)
            main_win.process.start()