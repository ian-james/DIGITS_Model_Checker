import cv2
import logging
from enum import Enum

class DisplayCap_Info:     
    
    # Create a VideoCap_Info object with specific information
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.window_names = []
        self.mode = ""

    def __str__(self):
        return "DisplayCap_Info: " + str(self.args) + " " + str(self.kwargs) + " " + str(self.windows) + " " + str(self.mode)
    
    # Compare two VideoCap_Info objects.
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, DisplayCap_Info):
            return False
        return self.args == o.args and self.kwargs == o.kwargs and self.windows == o.windows and self.mode == o.mode
    
    # Support Context Manager 'with' statement
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        for window_name in self.window_names:
            cv2.destroyWindow(window_name)
        return False
    
    def __del__(self):
        for window_name in self.window_names:
            cv2.destroyWindow(window_name)

    # Add a new window to the display
    def add_new_window(self, name, cv2_window_mode = cv2.WINDOW_NORMAL,x=0,y=0):
        cv2.namedWindow(name, cv2_window_mode)
        cv2.moveWindow(name,x,y)
        self.windows_names.append(name)
        return self
    
    # Add multiple new windows to the display
    def add_multiple_new_windows(self, names, cv2_window_modes, positions):
        for i in range(len(names)):
            self.add_new_window(names[i], cv2_window_modes[i], positions[i][0], positions[i][1])
        return self
    
    # Show an image in a window
    def show_image(self, name, image):
        cv2.imshow(name, image)
        return self

    # Show multiple images in windows
    def show_multiple_images(self, names, images):
        for i in range(len(names)):
            self.show_image(names[i], images[i])
        return self
    
    # Wait for a key press
    def wait_key(self, delay = 0):
        return cv2.waitKey(delay)
    
    # Close a window
    def close_window(self, name):
        cv2.destroyWindow(name)
        return self
    
    # Close all windows
    def close_all_windows(self):
        cv2.destroyAllWindows()
        return self
    
    # Get a windows size
    def get_window_size(self, name):
        return cv2.getWindowImageRect(name)
    
    # Set a windows size
    def set_window_size(self, name, width, height):
        cv2.resizeWindow(name, width, height)
        return self
    
    # Set all windows into an equal grid format
    def set_windows_grid(self, rows, cols):
        cv2.setAllWindows(rows, cols)
        return self
    
    