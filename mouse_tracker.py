import pyautogui
import pandas as pd
import time


mouse_positions = []

interval = 0.2  # seconds

screen_size_x, screen_size_y = pyautogui.size()

# Track mouse positions
start_time = time.time()
while True:
    x, y = pyautogui.position()
    if x>=screen_size_x-5 and x<=screen_size_x and y>=screen_size_y-5 and y<=screen_size_y:
        break
    mouse_positions.append((x, y, time.time() - start_time))
    time.sleep(interval)

# Save to csv file
df = pd.DataFrame(mouse_positions, columns=['X', 'Y', 'Time'])
df.to_csv('mouse_positions.csv', index=False)
