# Open a CV2 window and display the image
# Have a main function that is modular and the image is static

import cv2

import logging
from log import set_log_level
from datetime import datetime, timezone
from mediapipe_helpers import get_hand_model, FrameProcessor, draw_landmarks_on_image, hand_landmarks_to_dataframe

from VideoCap import VideoCap_Info, WindowName

from main import process_frame_from_cap
from fps_timer import FPS


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.drawing_styles import _HAND_LANDMARK_STYLE, _RED
from mediapipe.framework.formats import landmark_pb2

import pandas as pd

def run_loop(cap, detector, show_visual=True):
        # Setup Display
    if(show_visual):
        for window_name in WindowName:
            cv2.namedWindow(window_name.name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name.name, 600,400)

        cv2.moveWindow(WindowName.ORIGINAL.name,0,0)
        cv2.moveWindow(WindowName.LANDMARKS.name,0,400)
        cv2.moveWindow(WindowName.ENHANCED.name,600,0)
        cv2.moveWindow(WindowName.STATUS.name,600,400)

    fps = FPS()
    fps.start()

    df = pd.DataFrame()

    while cap.is_opened():

        success, image = cap.get_frame()
        fps.update()

        if not success:
            if(cap.is_video()):
                logging.info("Finished the video.")
                break
            else:
                logging.info("Ignoring empty camera frame.")
                continue

        # Change BGR for processing
        if(show_visual):
            #print("ORIGINAL")   
            cv2.imshow(WindowName.ORIGINAL.name, image)
        bimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if(show_visual):
            #print("ENahnce")
            cv2.imshow(WindowName.ENHANCED.name, bimage)

        # Convert to Image format

        converted_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=bimage)
        detection_results = detector.detect(converted_frame)
        enhanced_frame = bimage

        if( detection_results and detection_results.hand_landmarks != []): 
            # STEP 5: Process the classification result. In this case, visualize it.
            #print("Detection REsults")
            enhanced_frame = draw_landmarks_on_image(image,detection_results)
            ldf = hand_landmarks_to_dataframe(detection_results,fps.total_num_frames)
            df = pd.concat([df,ldf],axis=0)

        if(show_visual):
            #print("Landmark")
            cv2.imshow(WindowName.LANDMARKS.name, enhanced_frame)

        limage = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)
        if(show_visual):
            #print("Status")
            cv2.imshow(WindowName.STATUS.name, limage)

        if(show_visual):
            key = cv2.waitKey(1)
            if key == ord('p'):
                cv2.waitKey(5000)
            elif key & 0xFF == 27:
                break
            elif key == ord('q'):
                break
            elif key == ord('p'):

                cv2.waitKey()

    if(show_visual):
        cv2.destroyAllWindows()

    return df

def main():
    # Load the image
    # Setup Logging
    set_log_level("info")
    logging.info("Starting Program")

    filename = "datasets/carmin_test/videos/018-R-3-3.mp4"

    # Put the timestamp into the file.
    timestamp = datetime.now(timezone.utc).isoformat()
    
    outputfile =  f"outputfile_+{timestamp}.csv" 
    
    frame_processor = None
    hand_model = get_hand_model()
    frame_processor = FrameProcessor(hand_model)

    base_options = python.BaseOptions(model_asset_path='./models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
        running_mode=vision.RunningMode.IMAGE, #vision.RunningMode.VIDEO,vision.RunningMode.LIVE_STREAM
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    #vision.HandLandmarkerOptions(base_options=base_options)
    detector = vision.HandLandmarker.create_from_options(options)

    try:
        with VideoCap_Info.with_no_info() as cap:
            cap.setup_capture(filename)

            test_on = False

            if( test_on ):
                process_frame_from_cap(cap, frame_processor, False)
                df = frame_processor.get_dataframe()
            
                print(f"Writing output file. {outputfile}")
                df.to_csv(outputfile, sep="\t",index=0)
            else:
                df = run_loop(cap,detector,False)
                print(f"Writing output file. {outputfile}")
                df.to_csv(outputfile, sep="\t", index=False)           

    except Exception as e:
        logging.error(f"Failed to read video or camera options. {e}")
    logging.info("End of Program")

if __name__ == "__main__":
    main()
