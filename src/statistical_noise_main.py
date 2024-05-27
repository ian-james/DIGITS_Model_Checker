import os


def nyu_video_resource_paths(dir = "NYUHandPoseDataset/"):
    dir = "/home/jame/Projects/DIGITS_Model_Checker/output/nyu_tests/videos/"
    file_type =  "mp4."

def real_hands_resource_paths(dir = "RealHands/"):
    file_type = ".jpg"
    partial_path = ["user01","user02","user03","user04_01","user04_02","user05_01","user05_02","user06_01","user06_02","user06_03","user07"]


def load_finger_paint_resource_paths(dir = "FingerPaint/"):
    file_type = ".png"
    partial_path = ["combinedSubject","globalSubject","personalSubject","posesSubject"]
    partial_tails = ["_labels","_depth"]
    partial_subjects =[ "A","B","C","D","E"]    

    paths = []
    
    for partial in partial_path:
        for subject in partial_subjects:       
            for tail in partial_tails:
                sub_dir = f"{dir}/{partial}{subject}\"_\"{tail}"                
                paths.append(sub_dir)


# An hardcoded array of locations of images directories
def load_cvpr15_resource_paths(dir = ""):

    if(dir == ""):
        dir = "cvpr15_MSRAHandGestureDB/"    
    file_type = ".jpg"    
    entries = 500
    
    subdirs = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
    # I = Index out, IP = Index pinky (bull horn), L = L shape, MP = Middle pinky, RP = Ring Pinky, T = Thumb, TIP = Thumb, Index, Pinky, Y = Y shape

    per_patient_dirs = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "I","IP","L","MP","RP","T","TIP","Y"]

    descriptive_file = "joint.txt"

def load_resource_paths():
    # An hardcoded array of locations of images directories
    base_dir = "/home/jame/Projects/Western/Western_Postdoc/Datasets/"
    resource_paths = [ 

    ]
