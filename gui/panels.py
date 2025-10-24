from tkinter import Button, Label, Toplevel, StringVar, OptionMenu, Radiobutton, Entry
import pyautogui


class Panel:

    @staticmethod
    def show_main_menu(main_win, db_handler):
        """Show main menu with buttons"""
        for widget in main_win.root.winfo_children():
            widget.destroy()

        # if hasn't chosen facial part, then choose it first
        listOfTables = db_handler.get_face_tables()

        if not listOfTables:
            Panel.show_facial_selection(main_win)
        else:
            # Create buttons
            main_win.start_btn = Button(main_win.root, text="使用微表情", command=main_win.start_detect)
            main_win.start_btn.grid(row=0, column=0, pady=5, ipadx=10)
            
            main_win.train_btn = Button(main_win.root, text="訓練模型", command=main_win.start_train)
            main_win.train_btn.grid(row=1, column=0, pady=5, ipadx=10)

            main_win.continue_btn = Button(main_win.root, text="連續用快捷鍵", command=main_win.show_subtle_continue)
            main_win.continue_btn.grid(row=2, column=0, pady=5, ipadx=10)

            main_win.create_btn = Button(main_win.root, text="編輯快捷鍵", command=main_win.edit)
            main_win.create_btn.grid(row=3, column=0, pady=5, ipadx=10)
            
            main_win.restore_btn = Button(main_win.root, text="恢復初始狀態", command=main_win.show_restore)
            main_win.restore_btn.grid(row=4, column=0, pady=5, ipadx=10)
            
            main_win.root.grid_columnconfigure((0), weight=1)
            main_win.root.grid_rowconfigure((0,1,2,3,4), weight=1)



    @staticmethod
    def show_facial_selection(main_win):
        """Show facial part selection if not set"""
        welcome = Label(main_win.root, text="請擇一微表情部位")
        welcome.grid(row=0, column=0)
        
        main_win.faceVar = StringVar()
        radiobutton1 = Radiobutton(main_win.root, text='眉毛',
                            variable=main_win.faceVar, value='brow')
        radiobutton1.grid(row=1, column=0, pady=2)
        radiobutton2 = Radiobutton(main_win.root, text='鼻子',
                            variable=main_win.faceVar, value='nose')
        radiobutton2.grid(row=2, column=0, pady=2)
        radiobutton3 = Radiobutton(main_win.root, text='嘴唇',
                            variable=main_win.faceVar, value='lips')
        radiobutton3.grid(row=3, column=0, pady=2)
        check_button = Button(main_win.root, text='確認送出', command=main_win.submit_face)
        check_button.grid(row=4, column=0, pady=2)
        
        main_win.root.grid_columnconfigure((0), weight=1)
        main_win.root.grid_rowconfigure((0,1,2,3,4), weight=1)



    @staticmethod
    def show_subtle_hot_key(main_win, db_handler):
        """Show subtle/hotkey buttons panel"""
        main_win.root.withdraw()
        
        if hasattr(main_win,'subtle')==False:
            main_win.subtle = Toplevel(main_win.root)  
            main_win.subtle.title('Short cut panel')
            main_win.subtle.geometry("150x250")
        
        main_win.subtle.attributes("-topmost", True)
        
        x, y = pyautogui.position()
        main_win.subtle.geometry('+{}+{}'.format(x+10,y-100))
        main_win.subtle.attributes('-alpha',0.8)
        main_win.subtle.deiconify()
        
        # get all keys
        records = db_handler.get_all_keys()
        cnt = len(records)

        # Create btns
        main_win.buttons = []

        for idx,record in enumerate(records):

            main_win.buttons.append(Button(main_win.subtle, text=str(record[0]),
                                      command=lambda record=record:main_win.btn_callback(record[1])))
            main_win.buttons[idx].grid(row=idx, column=0, pady=3)
        
        # Create backto Button
        if main_win.continue_subtle == 0:

            main_win.hide_btn = Button(main_win.subtle, fg="#1152f7", text="隱藏面板", command=main_win.subtle_hide)
            main_win.hide_btn.grid(row=cnt, column=0, pady=3, ipadx=10)

            main_win.delete_btn = Button(main_win.subtle, fg="#1152f7", text="停止偵測", command=main_win.subtle_backto)
            main_win.delete_btn.grid(row=cnt+1, column=0, pady=3, ipadx=10)


            main_win.subtle.grid_columnconfigure((0), weight=1)
            for i in range(0,cnt+2):
                main_win.subtle.grid_rowconfigure(i, weight=1)
        else:
            main_win.subtle.grid_columnconfigure((0), weight=1)
            for i in range(0,cnt):
                main_win.subtle.grid_rowconfigure(i, weight=1)
        
        main_win.subtle.protocol("WM_DELETE_WINDOW", main_win.subtle_backto)



    @staticmethod
    def show_restore(main_win):
        """Show confirm window for destroy"""
        main_win.root.withdraw()
        
        if hasattr(main_win,'restore')==False :
            main_win.restore = Toplevel(main_win.root) 
            main_win.restore.title('Restore panel')
            main_win.restore.geometry("150x150")
        
        main_win.restore.geometry('+{}+{}'.format(int(main_win.width/2-100),int(main_win.height/2-100)))
        main_win.restore.attributes('-alpha',0.8)
        main_win.restore.deiconify()

        main_win.confirm_btn = Button(main_win.restore, text="確定恢復初始設定", command=main_win.restore_init)
        main_win.confirm_btn.grid(row=0, column=0, pady=2, ipadx=10)

        main_win.back_btn = Button(main_win.restore, text="回主選單", command=main_win.restore_backto)
        main_win.back_btn.grid(row=1, column=0, pady=2, ipadx=10)

        main_win.restore.grid_columnconfigure((0), weight=1)
        main_win.restore.grid_rowconfigure((0,1), weight=1)
     
        main_win.restore.protocol("WM_DELETE_WINDOW", main_win.restore_backto)



    @staticmethod
    def show_edit(main_win, db_handler):
        """Show hot key edit window"""
        main_win.root.withdraw()
        
        if hasattr(main_win,'editor')==False:
            main_win.editor = Toplevel(main_win.root)
            main_win.editor.title('Update A Setting')
            main_win.editor.geometry("270x350")
        
        for widget in main_win.editor.winfo_children():
            widget.destroy()
            
        main_win.editor.geometry('+{}+{}'.format(int(main_win.width/2-100),int(main_win.height/2-100)))
        main_win.editor.attributes('-alpha',0.8)
        main_win.editor.deiconify()
        
        records = db_handler.get_all_keys()
        cnt = len(records)
        
        main_win.descrips = []
        main_win.options = []
        for idx,record in enumerate(records):
            main_win.descrips.append(Label(main_win.editor, text=str(record[2]) + " " + str(record[0]) + ": " + str(record[1])))
            main_win.descrips[idx].grid(row=idx+1, column=0, columnspan=2, pady=1)
            main_win.options.append(record[2])
        
        # Create Edit Button
        main_win.create_btn = Button(main_win.editor, text="新增快捷鍵", command=main_win.create)
        main_win.create_btn.grid(row=cnt+1, column=0, columnspan=2, pady=5, ipadx=70)

        # Create Text Box Labels
        main_win.delete_box_label = Label(main_win.editor, text="選擇 快捷鍵編號")
        main_win.delete_box_label.grid(row=cnt+2, column=0, pady=5)

        # Create Text Boxes
        main_win.delete_box = StringVar(main_win.editor)
        if len(main_win.options)>0:
            main_win.delete_box.set(main_win.options[0]) # default value
            main_win.menu = OptionMenu(main_win.editor, main_win.delete_box, *main_win.options)
        else:
            main_win.delete_box.set("尚未設定快捷鍵")
            main_win.menu = OptionMenu(main_win.editor, main_win.delete_box, "請先設定快捷鍵")    
        main_win.menu.grid(row=cnt+2, column=1, pady=5)

        # Create A Delete Button
        main_win.delete_btn = Button(main_win.editor, text="刪除快捷鍵", command=main_win.delete)
        main_win.delete_btn.grid(row=cnt+3, column=0, columnspan=2, pady=5, ipadx=70)
        
        # Create A backto Button
        main_win.back_btn = Button(main_win.editor, text="回主選單", command=main_win.backto)
        main_win.back_btn.grid(row=cnt+4, column=0, columnspan=2, pady=5, ipadx=70)
        
        main_win.editor.grid_columnconfigure((0,1), weight=1)
        for i in range(0,cnt+5):
            main_win.editor.grid_rowconfigure(i, weight=1)

        main_win.editor.protocol("WM_DELETE_WINDOW", main_win.backto)



    @staticmethod
    def show_create(main_win):
        """Show hot key create window"""
        main_win.editor.withdraw()

        if hasattr(main_win,'creator')==True:           
            main_win.creator.destroy()
            delattr(main_win, "creator")

        main_win.creator = Toplevel(main_win.editor)
        main_win.creator.title('Create A Setting')
        main_win.creator.geometry("300x250")  

        for widget in main_win.creator.winfo_children():
            widget.destroy()
        
        main_win.creator.geometry('+{}+{}'.format(int(main_win.width/2-100),int(main_win.height/2-100)))
        main_win.creator.attributes('-alpha',0.8)
        main_win.creator.grid_columnconfigure(0, weight=1)
        main_win.creator.grid_columnconfigure(1, weight=1)
        for i in range(0, 4): 
            main_win.creator.grid_rowconfigure(i, weight=1)
        main_win.creator.protocol("WM_DELETE_WINDOW", main_win.create_window_delete)
        main_win.creator.deiconify()


        # Create Text Boxes
        main_win.descrip = Entry(main_win.creator, width=30)
        main_win.descrip.grid(row=0, column=1, padx=10, pady=(10, 0))

        main_win.hotkeys= Entry(main_win.creator, width=30)
        main_win.hotkeys.grid(row=1, column=1, padx=10, pady=(10, 0))


        # Create Text Box Labels
        main_win.descrip_label = Label(main_win.creator, text="功能名稱")
        main_win.descrip_label.grid(row=0, column=0, padx=10, pady=(20, 0))
        main_win.hotkeys_label = Label(main_win.creator, text="快捷鍵組合")
        main_win.hotkeys_label.grid(row=1, column=0, padx=10)


        # Record Hotkey Button
        main_win.record_btn = Button(main_win.creator, text="偵測快捷鍵", command=main_win.record_hotkey)
        main_win.record_btn.grid(row=2, column=0, columnspan=2, pady=5, ipadx=70)

        # Submit Button
        main_win.submit_btn = Button(main_win.creator, text="送出", command=main_win.submit)
        main_win.submit_btn.grid(row=3, column=0, columnspan=2, pady=5, ipadx=70)

        
        if hasattr(main_win,'dialogue_text'):
            main_win.dialogue = Label(main_win.creator, text=main_win.dialogue_text)
            main_win.dialogue.grid(row=4, column=0, columnspan=2,  pady=5)


        