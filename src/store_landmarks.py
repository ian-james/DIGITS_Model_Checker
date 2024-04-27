import csv

def write_landmarks_to_csv(file_path, hand_landmarks_list):
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header
        header = ['Timestamp', 'Frame','Handedness','Landmark', 'X', 'Y', 'Z']
        csv_writer.writerow(header)

        # Write landmarks
        for idx, hand_landmarks in enumerate(hand_landmarks_list):
            for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                row_data = [f'Hand{idx + 1}_Landmark{landmark_id + 1}', landmark.x, landmark.y, landmark.z]
                csv_writer.writerow(row_data)