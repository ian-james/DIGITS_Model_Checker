import os
import re
import pandas as pd
import numpy as np

from pathlib import Path

from file_utils import setupAnyCSV, clean_spaces

from convert_mediapipe_index import get_joint_names , get_just_joint_names

# stats = ['max', 'min', 'mean', 'median', 'std', 'var', sem]
# LT, RT
# Palmer, Rad_Obl, Rad_Side, Uln_Obl, Uln_Side,
# Fist, Ext, IM
# get_joint_names()
# LT_UT, LT_UT, RT_UT, LT_MT, RT_MT, LT_MP, RT_MP

def build_vp_filename(view, pose):
    """Build a filename from the hand, name, view, and pose."""
    return f"{view}_{pose}"

def build_nvp_filename(nh, view, pose):
    """Build a filename from the hand, name, view, and pose."""
    return f"{nh}_{view}_{pose}"

def build_patient_filename(patient_number, hand, view, pose):
    """Build a filename from the hand, name, view, and pose."""
    return f"{patient_number}_{hand}_{view}_{pose}"
    

def get_hand_names():
    """Return the hand names to group by."""
    return ['Lt', 'Rt']

def get_view_names():
    """Return the view names to group by."""
    return ['Palmar', 'Rad_Obl', 'Rad_Side', 'Uln_Obl', 'Uln_Side']

def get_our_study_view_names():
   return ['Palmar', 'Rad_Obl', 'Uln_Obl' ]

def get_our_study_pose_names():
    return ['Ext', 'Fist',"IM","Abduction","Opposition","Flex","Circumduction"]

def get_pose_names(include_im=True):
    """Return the pose names to group by."""
    a = ['Ext', 'Fist']
    if include_im:
        a.append('IM')
    return a

def is_extension_file(pose):
    return 'Ext' in pose or 'extension' in pose.lower()

def is_flexion_file(pose):
    return 'Fist' in pose or 'flexion' in pose.lower()

def is_intrinsic_minus_file(pose):
    return 'IM' in pose or 'intrinsic minus' in pose.lower()

def get_pose_full_names():
    """Return the pose names to group by."""
    return ['Extension', 'Flexion', 'Intrinsic Minus']
    

def build_vp_group_names(include_im=True):
    """Build the group names from the hand, view, and pose."""
    view_names = get_view_names()
    pose_names = get_pose_names(include_im)
    group_names = []
    for view in view_names:
        for pose in pose_names:
            group_names.append(build_vp_filename(view, pose))
    return group_names

#TODO: All views or some

def build_nvp_group_names(include_im=True): 
    """Build the group names from the hand, view, and pose."""
    hand_names = get_hand_names()
    view_names = get_view_names()
    pose_names = get_pose_names(include_im)
    group_names = []
    for hand in hand_names:
        for view in view_names:
            for pose in pose_names:
                group_names.append(build_nvp_filename(hand, view, pose))
    return group_names

def opposite_hand_name(filename):
    """Replace hand name with the opposite hand."""
    if 'Lt' in filename:
        return filename.replace('Lt', 'Rt')
    elif 'Rt' in filename:
        return filename.replace('Rt', 'Lt')
    else:
        return filename

def is_left_hand(filename):
    """Check if the filename is for the left hand."""
    return 'Lt' in filename

def remove_hand_name(filename):
    """Remove the hand name from the filename."""
    return filename.replace('Lt_', '').replace('Rt_', '')    

def get_all_keywords():
    """Return all the keywords for the filenames."""
    # Split get_view_names because of _ in the names
    clean_views = [view.split('_') for view in get_view_names()]

    # Flatten the list
    clean_views = [item for sublist in clean_views for item in sublist]
    return get_hand_names() + clean_views + get_pose_names()

def keep_only_keywords( filename, keywords):
    # Remove any words not contained in the keywords list
    return '_'.join([word for word in filename.split('_') if word in keywords])

def get_patient_id(text):
    # Using regex get the patient id, the first number in the string
    return re.findall(r'\d+', text)[0]

def get_fingers():
    """Return the finger names to group by."""
    result = []
    for joint in get_joint_names().values():
        # split the joint name by the space
        joint = joint.split(' ')
        # Add the first word to the result
        result.append(joint[0])
    return list(set(result))    

def get_keywords_from_filename(filename, keywords):
    for keyword in keywords:
        if keyword in filename:
            return keyword
    return None

def get_view_pose_from_filename(filename):
    """Return the view and pose from the filename."""
    # Get the view and pose from the filename
    view = get_keywords_from_filename(filename, get_view_names())
    pose = get_keywords_from_filename(filename, get_pose_names())
    return view, pose


def get_finger_joint_from_filename(filename):
    """Return the finger joint from the filename."""
    # Get the view and pose from the filename
    joint = get_keywords_from_filename(filename, get_just_joint_names())
    finger = get_keywords_from_filename(filename, get_fingers())
    return finger, joint

def get_finger_from_filename(filename):
    return get_keywords_from_filename(filename, get_fingers())

def get_joint_from_filename(filename):
    return get_keywords_from_filename(filename, get_just_joint_names())


# Example files from process will from the following:
# 015-L-2-1_mediapipe_nh_1_md_0.5_mt_0.5_mp_0.5_model_mediapipe.csv
# to the new format:
# DIGITS_CJ_1_Lt_Palmar_Ext.csv
def change_filename_from_lab_numbers(filename):
    """Change the filename from lab numbers to the new format."""
    # Split the filename by the underscore
    parts = filename.split('_')
    # Get the first part
    
    patient_info =  parts[0].split('-')

    if( len(patient_info) < 4):
        print(f"Filename {filename} not formatted correctly.")
        return filename

    patient_number, hand, view, pose = patient_info
    hand_name = 'Lt' if 'L' in hand else 'Rt'

    # Get out study names
    views = get_our_study_view_names()
    poses = get_our_study_pose_names()

    try:
        view = views[int(view)-1]
        pose = poses[int(pose)-1]    

        # Return the new filename, Drop any leading zeroes
        return f"DIGITS_{build_patient_filename(str(int(patient_number)), hand_name, view, pose)}.csv"    
    # Array exception
    except IndexError:
        print(f"Filename {filename} not formatted correctly.")
    
    return filename

# DIGITS_CJ_1_Lt_Palmar_Ext.csv
def check_if_file_is_in_lab_format(filename, check_hand=True):
    """Check if the file is in the lab format."""
    
    view = get_keywords_from_filename(filename, get_view_names())
    pose = get_keywords_from_filename(filename, get_pose_names())
    if(check_hand):
        hand = get_keywords_from_filename(filename, get_hand_names())
        return (view and pose and hand)
    return (view and pose)
    


    
if __name__ == "__main__":
    print( change_filename_from_lab_numbers("015-L-2-1_mediapipe_nh_1_md_0.5_mt_0.5_mp_0.5_model_mediapipe.csv") )
    print( change_filename_from_lab_numbers("015-L-2-21_mediapipe_nh_1_md_0.5_mt_0.5_mp_0.5_model_mediapipe.csv") )

    if(check_if_file_is_in_lab_format("DIGITS_1_Lt_Palmar_Ext.csv") ):
        print("Filename is in the correct format.")
    
    if( check_if_file_is_in_lab_format("DIGITS_1_Palmar_Ext.csv", False) ):
        print("Filename is in the correct format.")

    if( not check_if_file_is_in_lab_format("DIGITS_Ext.csv") ):
        print("The file is not in the correct format.")