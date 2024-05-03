
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# BaseOptions allows an asset_path to the model(task)

# Vision is the space for all Landmarkers,options, and Results
# HandLandmarkerOptions is the option for the HandLandmarker
# HandLandmarker cotains handedness, normalized position, and world position

# Options for the HandLandmarker
# running_model
# num_hands: int = 2
# min_hand_detection_confidence: float = 0.5
# min_hand_presence_confidence: float = 0.5
# min_tracking_confidence: float = 0.5


def get_hand_model(num_hands = 1, min_detection_confdience= 0.5, min_presence_confidence =0.5, min_tracking_confidence=0.5):    
    base_options = python.BaseOptions(model_asset_path='./models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
        running_mode=  vision.RunningMode.VIDEO, #vision.RunningMode.IMAGE, #vision.RunningMode.VIDEO,vision.RunningMode.LIVE_STREAM
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confdience,
        min_hand_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    vision.HandLandmarkerOptions(base_options=base_options)               
    detector = vision.HandLandmarker.create_from_options(options)
    return detector

def get_video_hand_model():    
    base_options = python.BaseOptions(model_asset_path='./models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.8,
        min_hand_presence_confidence=0.8,
        min_tracking_confidence=0.8
    )
    vision.HandLandmarkerOptions(base_options=base_options,)               
    detector = vision.HandLandmarker.create_from_options(options)
    return detector

