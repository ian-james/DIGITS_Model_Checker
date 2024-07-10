import math
import numpy as np

def calc_digit_length(p1, p2):
    """
    Calculate the distance between two points in 3D space.
    """
    return math.dist(p1,p2)

def calculate_angle(v1,v2):
    """
    Calculate the angle between two vectors.
    """
    print("Betweeen Vectors: ", v1, v2)
    d = np.dot(v1,v2)
    n = np.linalg.norm(v1)*np.linalg.norm(v2)
    s = min(1,max(-1,d/n))
    r = math.acos(s)
    r2 =  r * (180/math.pi)
    return math.degrees(r )

def subtract_lists(list1, list2):
    """
    Subtract two lists element wise.
    """
    return [a-b for a,b in zip(list1,list2)]

def convert_landmarks_to_vector(landmarks):
    """
    Convert a list of landmarks to a list of vectors.
    """
    return [subtract_lists(landmarks[i],landmarks[i-1]) for i in range(1,len(landmarks))]


def select_landmarks(landmarks, indices):
    """
    Select landmarks based on a list of indices.
    """
    return [landmarks[i] for i in indices]

def calculate_angle_between_each_digit_joint(landmarks, joint_indices1):
    #THUMB CMC, MCP, IP, TIP
    #MCP,PIP,DIP, TIP
    digit = select_landmarks(landmarks, joint_indices1)
    vectors = convert_landmarks_to_vector(digit)    
    r = [calculate_angle(vectors[i], vectors[i+1]) for i in range(len(vectors)-1)]
    return r


if __name__ == "__main__":
    # Test the functions
    p1 = [0,0,0]
    p2 = [1,1,1]
    print(calc_digit_length(p1,p2))

    v1 = [1,0,0]
    v2 = [0,1,0]
    print(calculate_angle(v1,v2))

    landmarks = [
        [0, 0, 0],  # Wrist
        [1, 0, 0],  # Thumb CMC
        [2, 1, 0],  # Thumb MCP
        [3, 1, 1],  # Thumb IP
        [4, 2, 1]  # Thumb Tip
    ]
    
    vectors = convert_landmarks_to_vector(landmarks)
    print(vectors)
    
    for i in range(1,len(vectors)):
        print(calculate_angle(vectors[i-1],vectors[i]))

    test_vectors = [[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 1], [4, 2, 1]]
    print("Calculations:")
    for i in range(1,len(test_vectors)):
        print(calculate_angle(vectors[i-1],vectors[i]))

    
