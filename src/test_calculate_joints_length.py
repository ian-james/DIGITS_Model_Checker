from calculate_joints_and_length import *

#********************************************************************************************************************
# Test Section

# write a couple of tests for the functions
# Test the calculate_length function
# Test the calculate_digit_length function

# Test Main
def test_main():
    ap = setup_arguments()
    args = vars(ap.parse_args())
    set_log_level(args['log'])
    logging.info("Starting TEST Program")

    # run the test functions with print statements
    test_calculate_length()
    #test_calculate_digit_length()
    #test_calculate_all_digit_lengths()

    test_calculate_full_digit_range_df()

    test_calculate_digit_range()
    test_calculate_full_range()

    print("All Done")

# Write a test function for calculate_full_range_df(...)
def test_calculate_full_digit_range_df():
    # Create a dummy dataframe, with columns matching the hand headers listed at the top of the file.
    # The values are arbitrary, but should be consistent with the expected values.
    titles = [ "Thumb Tip","Index Tip","Middle Tip"]
    df = pd.DataFrame([[[0,0,0],[1,1,1],[2,2,2]], [[1,1,1],[2,2,2],[3,3,3]], [[2,0,2],[0,3,0],[0,0,4]]], columns=titles)
    print(df)
    res = calculate_full_digit_range_df(df, "Thumb Tip")
    print(res)

    # Res in an array of two dictionaries
    # The first dictionary contains the min values for each coordinate
    # The second dictionary contains the max values for each coordinate
    assert res[0]["Thumb Tip_x_min"] == [0,0,0]
    assert res[0]["Thumb Tip_y_min"] == [0,0,0]
    assert res[0]["Thumb Tip_z_min"] == [0,0,0]
    assert res[1]["Thumb Tip_x_max"] == [2,0,2]
    assert res[1]["Thumb Tip_y_max"] == [1,1,1]
    assert res[1]["Thumb Tip_z_max"] == [2,0,2]
    



def test_calculate_length():
    p1 = [0,0,0]
    p2 = [1,1,1]
    assert calculate_length(p1,p2) == math.sqrt(3)

def test_calculate_digit_length():
    df = pd.DataFrame({
        "Thumb Tip (X-coordinate)":0,
        "Thumb Tip (Y-coordinate)":0,
        "Thumb Tip (Z-coordinate)":0,
        "Thumb CMC (X-coordinate)":1,
        "Thumb CMC (Y-coordinate)":1,
        "Thumb CMC (Z-coordinate)":1
    },index=[0])
    assert calculate_digit_length(df, "Thumb Tip (X-coordinate)", "Thumb CMC (X-coordinate)") == 1

def test_calculate_all_digit_lengths():
    df = pd.DataFrame({
        "Thumb Tip (X-coordinate)": 0,
        "Thumb Tip (Y-coordinate)": 0,
        "Thumb Tip (Z-coordinate)": 0,
        "Thumb CMC (X-coordinate)": 1,
        "Thumb CMC (Y-coordinate)": 1,
        "Thumb CMC (Z-coordinate)": 1,
        "Index Finger Tip (X-coordinate)": 0,
        "Index Finger Tip (Y-coordinate)": 0,
        "Index Finger Tip (Z-coordinate)": 0,
        "Index Finger MCP (X-coordinate)": 1,
        "Index Finger MCP (Y-coordinate)": 1,
        "Index Finger MCP (Z-coordinate)": 1,

    }, index=[0])
    digit_lengths = calculate_all_digit_lengths(df, None)
    assert digit_lengths["Thumb_Length"] == 1
    assert digit_lengths["Index_Length"] == 1


def test_calculate_digit_range():
    df = pd.DataFrame({
        "Thumb Tip (X-coordinate)": 0,
        "Thumb Tip (Y-coordinate)": 0,
        "Thumb Tip (Z-coordinate)": 0,
        "Thumb CMC (X-coordinate)": 1,
        "Thumb CMC (Y-coordinate)": 1,
        "Thumb CMC (Z-coordinate)": 1
    },index=[0])
    assert calculate_digit_range(df, "Thumb Tip (X-coordinate)") == [0,0]

def test_calculate_full_range():
    df = pd.DataFrame([{
        "Thumb Tip (X-coordinate)": 0,
        "Thumb Tip (Y-coordinate)": 0,
        "Thumb Tip (Z-coordinate)": 0,
        "Thumb CMC (X-coordinate)": 0,
        "Thumb CMC (Y-coordinate)": 0,
        "Thumb CMC (Z-coordinate)": 0,
        "Index Finger Tip (X-coordinate)": 0,
        "Index Finger Tip (Y-coordinate)": 0,
        "Index Finger Tip (Z-coordinate)": 0,
        "Index Finger MCP (X-coordinate)": 0,
        "Index Finger MCP (Y-coordinate)": 0,
        "Index Finger MCP (Z-coordinate)": 0
    },  {

        "Thumb Tip (X-coordinate)": 1,
        "Thumb Tip (Y-coordinate)": 1,
        "Thumb Tip (Z-coordinate)": 1,
        "Thumb CMC (X-coordinate)": 1,
        "Thumb CMC (Y-coordinate)": 1,
        "Thumb CMC (Z-coordinate)": 1,
        "Index Finger Tip (X-coordinate)": 1,
        "Index Finger Tip (Y-coordinate)": 1,
        "Index Finger Tip (Z-coordinate)": 1,
        "Index Finger MCP (X-coordinate)": 1,
        "Index Finger MCP (Y-coordinate)": 1,
        "Index Finger MCP (Z-coordinate)": 1
    }])

    digit_ranges = calculate_full_range(df, None, None)
    assert digit_ranges["Thumb Tip (X-coordinate)_Range_Min"] == 0
    assert digit_ranges["Thumb Tip (X-coordinate)_Range_Max"] == 1
    assert digit_ranges["Index Finger Tip (X-coordinate)_Range_Min"] == 0
    assert digit_ranges["Index Finger Tip (X-coordinate)_Range_Max"] == 1


def test_thumb_angles(landmarks, expected_angles): 
    
    angles = calculate_angle_between_each_digit_joint(landmarks, [0,1,2,3,4])
    
    #Check that the calculates angles are within accepted error range.
    for i in range(len(expected_angles)):
        print(f"i = {i}  Expected: {expected_angles[i]}, Calculated: {angles[i]}")
        assert abs(angles[i] - expected_angles[i]) < 1e-5, f"Expected {expected_angles}, but got {angles}"    
    
    print("All tests pass")

def test_thumb_angles_all_zero():
    landmarks = [
        [0, 0, 0],  # Wrist
        [1, 1, 1],  # Thumb CMC
        [2, 2, 2],  # Thumb MCP
        [3, 3, 3],  # Thumb IP
        [4, 4, 4]  # Thumb Tip
    ]
    expected_angles = [0, 0, 0]
    test_thumb_angles(landmarks, expected_angles)

def test_thumb_angles_all_45():
    landmarks = [
        [0, 0, 0],  # Wrist
        [1, 0, 0],  # Thumb CMC
        [2, 1, 0],  # Thumb MCP
        [3, 1, 1],  # Thumb IP
        [4, 2, 1]  # Thumb Tip
    ]
    expected_angles = [45, 60, 60]
    test_thumb_angles(landmarks, expected_angles)


def test_calculate_all_finger_angles():
    landmarks = [
        [0, 0, 0],  # Wrist
        [1, 0, 0],  # Thumb CMC
        [2, 1, 0],  # Thumb MCP
        [3, 1, 1],  # Thumb IP
        [4, 2, 1],  # Thumb Tip
        [5, 0, 0],  # Index Finger MCP
        [6, 1, 0],  # Index Finger PIP
        [7, 1, 1],  # Index Finger DIP
        [8, 2, 1],  # Index Finger Tip
        [9, 0, 0],  # Middle Finger MCP
        [10, 1, 0],  # Middle Finger PIP
        [11, 1, 1],  # Middle Finger DIP
        [12, 2, 1],  # Middle Finger Tip
        [13, 0, 0],  # Ring Finger MCP
        [14, 1, 0],  # Ring Finger PIP
        [15, 1, 1],  # Ring Finger DIP
        [16, 2, 1],  # Ring Finger Tip
        [17, 0, 0],  # Pinky MCP
        [18, 1, 0],  # Pinky PIP
        [19, 1, 1],  # Pinky DIP
        [20, 2, 1]  # Pinky Tip
    ]
    expected_angles = {
        Digit.Thumb.name: [45, 60, 60],
        Digit.Index.name: [45, 60, 60],
        Digit.Middle.name: [45, 60, 60],
        Digit.Ring.name: [45, 60, 60],
        Digit.Pinky.name: [45, 60, 60]
    }
    angles = calculate_all_finger_angles(landmarks,True,False)
    for digit in angles:
        for i in range(len(angles[digit])):
            assert abs(angles[digit][i] - expected_angles[digit][i]) < 1e-5, f"Expected {expected_angles}, but got {angles}"
    print("All tests pass")




# ********************************************************************************************************************
if __name__ == "__main__":    
    #test_csv_to_landmarks()
    test_main()