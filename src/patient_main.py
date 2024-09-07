# This program will take an all_combined.csv file and create tables for each patient
# The patient digit lenght, ROM, and other statistics 


## Patient files will namned something similar to 001-L-1-1_mediapipe_nh
# Will load all the Goniometry data for ground truth comparison
# Goniometry data will have each angle twice (flexion and extension) for each joint


#Output will be something similar to this ( then all the angles in a column)
# 001-L-1-1_mediapipe_nh	1	0.8	0.8	0.8	mediapipe
# 001-L-1-2_mediapipe_nh	1	0.8	0.8	0.8	mediapipe
# 001-L-1-3_mediapipe_nh	1	0.8	0.8	0.8	mediapipe
# 001-L-1-4_mediapipe_nh	1	0.8	0.8	0.8	mediapipe
# 001-L-1-5_mediapipe_nh	1	0.8	0.8	0.8	mediapipe
# 001-L-1-6_mediapipe_nh	1	0.8	0.8	0.8	mediapipe
# 001-L-1-7_mediapipe_nh	1	0.8	0.8	0.8	mediapipe
# 001-L-3-1_mediapipe_nh	1	0.8	0.8	0.8	mediapipe


# 		Thumb			Index			Long			Ring			Little		
# Paticipant ID	Data Replicate	CMC	MP	IP	MP	PIP	DIP	MP	PIP	DIP	MP	PIP	DIP	MP	PIP	DIP
# 1	1															
# 	2															
# 	3															
# 	Mean	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!
# 	Standard Deviation	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!	#DIV/0!
# 	Manual Goniometry								
# 

# Calculations needs - Identify the max or calculated angle for each patient (001)
# Calculate each pose for the patient


# Algorithm
# Load the Goniometry Real Data and Measured Distance
# Load the All-Combined file with multiple poses or repeatitions

