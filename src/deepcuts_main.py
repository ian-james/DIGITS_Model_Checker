import deeplabcut

import os
from pathlib import Path

path_config_file = os.path.join(os.getcwd(),'openfield-Pranav-2018-10-30/config.yaml')
deeplabcut.load_demo_data(path_config_file)

deeplabcut.train_network(path_config_file, shuffle=1, displayiters=10, saveiters=100)
deeplabcut.evaluate_network(path_config_file,plotting=False)

# Creating video path:
import os
videofile_path = os.path.join(os.getcwd(),'openfield-Pranav-2018-10-30/videos/m3v1mp4.mp4')


print("Start analyzing the video!")
#our demo video on a CPU with take ~30 min to analze! GPU is much faster!
deeplabcut.analyze_videos(path_config_file,[videofile_path])
