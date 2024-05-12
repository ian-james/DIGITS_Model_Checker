import cv2
import logging
from enum import Enum, auto

def is_string_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


# Find the camera index.
def find_camera(cams_test = 10):    
    results = []
    for i in range(0, cams_test):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                results.append(i)
        except:
            pass
    print(f"Found {len(results)} cameras: {results}")
    return results
class VideoMode(Enum):
    CAMERA = 0
    VIDEO = 1

class WindowName(Enum):
    ORIGINAL = auto()
    LANDMARKS = auto()
    ENHANCED = auto()
    STATUS = auto()
    
class VideoCap_Info:     
    
    # Create a VideoCap_Info object with specific information
    def __init__(self,cap, fps_rate, width, height, mode):
        self.cap = cap
        self.fps_rate = fps_rate
        self.width = width
        self.height = height
        self.mode = mode
        self.frame_id = 0  

    # Create a VideoCap_Info object with no information.
    @classmethod
    def with_no_info(video_cap):
        return video_cap(cap = None, fps_rate = 0, width = 0, height = 0, mode = VideoMode.CAMERA)
          
    def __str__(self):
        return "VideoCap_Info: fps_rate=" + str(self.fps_rate) + " width=" + str(self.width) + " height=" + str(self.height) + " mode=" + str(self.mode)    
    
    # Compare two VideoCap_Info objects.
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, VideoCap_Info):
            return False
        return self.fps_rate == o.fps_rate and self.width == o.width and self.height == o.height and self.mode == o.mode
    
    # Support Context Manager 'with' statement
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if(self.cap is not None):
            self.cap.release()
        return False
    
    def __del__(self):
        if(self.cap is not None):
            self.cap.release()    
    
    
    def is_video(self):
        return self.mode == VideoMode.VIDEO
    
    def is_camera(self):
        return self.mode == VideoMode.CAMERA
    
    def is_opened(self):
        return self.cap.isOpened()
    
    # Setup a video or camera capture object
    def setup_capture(self, filename):
        if(is_string_int(filename)):
            camera_index = int(filename)
            self.setup_camera_capture(camera_index)
        else:
            self.setup_video_capture(filename)
                                     
    def setup_details(self,filename):      
        self.cap = cv2.VideoCapture(filename)
        if(not self.cap.isOpened()):
            raise ("FAILED TO LOAD VIDEO filename= '", filename, "'")

        self.fps_rate = self.cap.get(cv2.CAP_PROP_FPS)   
        # Get the input video size and frame rate
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"FPS= {self.fps_rate}")
        logging.info(f"Width = {self.width}")
        logging.info(f"Height= {self.height}")
    
    def setup_camera_capture(self, camera_index):
        self.mode = VideoMode.CAMERA
        if(camera_index < 0):
            cameras = find_camera()        
            if(len(cameras) == 0):
                raise ("FAILED TO FIND CAMERA")
            camera_index = cameras[0]
        
        self.setup_details(camera_index)
        logging.info(f"Using camera index= {camera_index}")

    def setup_video_capture(self, filename):    
        self.mode = VideoMode.VIDEO 
        self.setup_details(filename)
        logging.info(f"Using video file= {filename}") 
  
    
    def get_frame(self):                
        success, image = self.cap.read()
        if success:
            self.frame_id += 1
        return success, image