@echo on
call "C:\mini\Scripts\activate.bat" "C:\mini"
call conda activate mp
cd /d "C:\win_auto_selector"
python .
call conda deactivate