import math
import numpy as np
import decimal

def calc_digit_length(p1, p2):
    """
    Calculate the distance between two points in 3D space.
    """
    return math.dist(p1,p2)

def calculate_angle(v1,v2):
    """
    Calculate the angle between two vectors.
    """    
    
    v1d = np.array(v1[0:2])
    v2d = np.array(v2[0:2])

    d = np.dot(v1d,v2d)
    n = np.linalg.norm(v1d)*np.linalg.norm(v2d)
    
    if(n == 0):
        return 0
    
    s = min(1,max(-1,d/n))
    r = math.acos(s)
    r2 =  r * (180/math.pi)
    
    return math.degrees(r )

def calculate_angles2(v1,v2):
    # Convert vectors to Decimal for high precision
    v1 = np.array([decimal.Decimal(x) for x in v1])
    v2 = np.array([decimal.Decimal(x) for x in v2])
    
    # Compute dot product and norms using Decimal
    d = sum(v1 * v2)
    n = decimal.Decimal(np.linalg.norm(v1)) * decimal.Decimal(np.linalg.norm(v2))
    
    if n == 0:
        return "One or both of the vectors are zero vectors, angle calculation is undefined."
    
    # Ensure the value for acos is within the range [-1, 1]
    s = min(decimal.Decimal(1), max(decimal.Decimal(-1), d / n))
    
    # Calculate the angle in radians
    r = decimal.Decimal(math.acos(float(s)))
    
    # Convert the angle to degrees
    r2 = r * (180 / decimal.Decimal(math.pi))
    return r2
    

# Example usage with small values
v1 = [1e-10, 2e-10, 3e-10]
v2 = [4e-10, 5e-10, 6e-10]

result = calculate_angle(v1, v2)
print(result)


def calculate_angle_between_three_points(p1, p2, p3):
    """
    Calculate the angle between three points.
    """
    v1 = subtract_lists(p1,p2)
    v2 = subtract_lists(p3,p2)
    return calculate_angle(v1,v2)

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

    # Apply the calculate angle function to a dataframe of vectors
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
