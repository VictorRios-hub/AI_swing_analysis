import cv2 as cv2
import mediapipe as mp
import os 
import numpy as np
import json
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import time 
import math
from sklearn.metrics.pairwise import euclidean_distances

#This system Finds the top 3 golfers most similar to amateur average and writes the keypoint data for each moment of swing to a file

def pro_comparison():

    # Get the current directory of the script
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # load amateur keypoint data
    amateur_keypoint_files = {
        'Address': 'Keypoint_data/AmateurKeypointData/Address_keypointData.json',
        'Mid-Backswing (arm parallel)': 'Keypoint_data/AmateurKeypointData/Mid-Backswing (arm parallel)_keypointData.json',
        'Mid-Downswing (arm parallel)': 'Keypoint_data/AmateurKeypointData/Mid-Downswing (arm parallel)_keypointData.json',
        'Mid-Follow-Through (shaft parallel)': 'Keypoint_data/AmateurKeypointData/Mid-Follow-Through (shaft parallel)_keypointData.json',
        'Top': 'Keypoint_data/AmateurKeypointData/Top_keypointData.json',
        'Toe-up': 'Keypoint_data/AmateurKeypointData/Toe-up_keypointData.json',
        'Impact': 'Keypoint_data/AmateurKeypointData/Impact_keypointData.json',
        'Finish': 'Keypoint_data/AmateurKeypointData/Finish_keypointData.json'
    }

    # Join the absolute path with each folder path
    for moment, filepath in amateur_keypoint_files.items():
        amateur_keypoint_files[moment] = os.path.join(current_directory, filepath)

    average_distances = []
    # average_euclienne = []
    average_cosine = []
    amateur_keypoint_data = {}
    for moment, filepath in amateur_keypoint_files.items():
        with open(filepath, 'r') as file:
            amateur_keypoint_data[moment] = json.load(file)


    #loop through each professional
    for i in range(1, 586):
        # print("Processing player", i)
        #load keypoint data for each professional
        professional_keypoint_data = {}
        for moment in amateur_keypoint_files.keys():
            filename = f'{moment}{i}.json'
            filepath = os.path.join(f'Keypoint_data/ProfessionalKeypointData/{moment}', filename)

            # Get the directory of the current script
            current_script_directory = os.path.dirname(os.path.abspath(__file__))
            relative_directory = filepath
            
            # Construct the absolute path
            filepath = os.path.join(current_script_directory, relative_directory)

            with open(filepath, 'r') as file:
                professional_keypoint_data[moment] = json.load(file)

        #calculate euclidean distance between amateur and professionals per moment of swing
        total_distance = 0
        total_cosine_similarity  = 0

        for moment in amateur_keypoint_files.keys():
            amateur_data = amateur_keypoint_data[moment]
            professional_data = professional_keypoint_data[moment]

            distance = 0
            cosine_similarity_score = 0
            num_keypoints = len(amateur_data)  # Assuming amateur_data and professional_data have the same keypoints

            for keypoint, amateur_keypoint_values in amateur_data.items():
                professional_keypoint_values = professional_data[keypoint]

                # Keypoints normalization
                amateur_keypoint_values = normalize(np.array(amateur_keypoint_values).reshape(1, -1))
                professional_keypoint_values = normalize(np.array(professional_keypoint_values).reshape(1, -1))

                # Euclidian distance
                distance += np.linalg.norm(np.array(amateur_keypoint_values) - np.array(professional_keypoint_values))

                # Cosine Similarity
                amateur_key = np.array(amateur_keypoint_values[:2])
                pro_key = np.array(professional_keypoint_values[:2])
                cosine_similarity_score += cosine_similarity(amateur_key.reshape(1 , -1), pro_key.reshape(1 , -1))
            
            total_distance += distance / num_keypoints
            total_cosine_similarity += cosine_similarity_score / num_keypoints

        average_distance = total_distance / len(amateur_keypoint_files.keys())
        average_cosine_similarity = total_cosine_similarity  / len(amateur_keypoint_files.keys())

        # print("Average distance for player {}: {}".format(i, average_distance))
        average_distances.append((i, average_distance))
        average_cosine.append((i, average_cosine_similarity))

        # Cosine Similarity with Euclidean Distance between amateur and professionals per moment of swing
        # total_cosine_similarity  = 0
        # total_distance = 0

        # for moment in amateur_keypoint_files.keys():
        #     amateur_data = amateur_keypoint_data[moment]
        #     professional_data = professional_keypoint_data[moment]

        #     cosine_similarity_score = 0
        #     distance = 0
        #     num_keypoints = len(amateur_data)  # Assuming amateur_data and professional_data have the same keypoints

        #     for keypoint, amateur_keypoint_values in amateur_data.items():
        #         professional_keypoint_values = professional_data[keypoint]

        #         # Cosine Similarity
        #         amateur_key = np.array(amateur_keypoint_values[:2])
        #         pro_key = np.array(professional_keypoint_values[:2])
                
        #         # Reshape the data to ensure it's a 2D array
        #         amateur_key_reshaped = amateur_key.reshape(1, -1)
        #         pro_key_reshaped = pro_key.reshape(1, -1)

        #         # L2 normalize the data
        #         amateur_key_normalized = normalize(amateur_key_reshaped, norm='l2')
        #         pro_key_normalized = normalize(pro_key_reshaped, norm='l2')

        #         # cosine_similarity_score += cosine_similarity([amateur_keypoint_values], [professional_keypoint_values])[0, 0]
        #         cosine_similarity_score += cosine_similarity(amateur_key_normalized, pro_key_normalized)

        #         # Calculate euclidian distance 
        #         # distance += np.linalg.norm(np.array(amateur_keypoint_values) - np.array(professional_keypoint_values))
        #         distance += euclidean_distances(amateur_key_normalized,pro_key_normalized)

        #     total_cosine_similarity  += cosine_similarity_score / num_keypoints
        #     total_distance += distance / num_keypoints

        # average_cosine_similarity = total_cosine_similarity  / len(amateur_keypoint_files.keys())
        # average_euclide = total_distance / len(amateur_keypoint_files.keys())

        # # print("Average Cosine Similarity for player {}: {}".format(i, average_cosine_similarity))
        # # print("Euclidean Distance for player {}: {}".format(i, average_euclide))
        
        # average_distances.append((i, average_cosine_similarity))
        # average_euclienne.append((i, average_euclide))


    #sort the average distances to decending order
    # sorted_distances = sorted(average_distances, key=lambda x: x[1], reverse=True)
    sorted_cosine = sorted(average_cosine, key=lambda x: x[1], reverse=True)
    sorted_distances = sorted(average_distances, key=lambda x: x[1], reverse=False)

    Top3List = []
    Top3Score = []

    #print 3 closest matches
    print("Top 3 closest matches:")
    for i in range(1):
        player, distance = sorted_distances[i]
        print("Player {}, Average distance: {}".format(player, distance))
        Top3List.append(player)
        Top3Score.append(distance)
        player, distance = sorted_cosine[i]
        print("Player {}, Average distance: {}".format(player, distance))
        Top3List.append(player)
        Top3Score.append(distance)
    
    Top1 = Top3Score[0] 

    # Use shutil.rmtree to remove the entire directory and its contents
    
    # Get the directory of the current script
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    relative_directory = 'Keypoint_data/Top3ProfessionalKeypointData/'
    
    # Construct the absolute path and delete the repo
    del_dir = os.path.join(current_script_directory, relative_directory)
    shutil.rmtree(del_dir)

    # Recreate the empty directory          
    os.makedirs(del_dir)
    # print("Directory created")
        
    for i in Top3List:    
        #load keypoint for each moment of swing
        for moment in amateur_keypoint_files.keys():
            filename = f'{moment}{i}.json'
            filepath = os.path.join(f'Keypoint_data/ProfessionalKeypointData/{moment}', filename)
            filepath = os.path.join(current_script_directory, filepath)
            Top3filepath = os.path.join(f'Keypoint_data/Top3ProfessionalKeypointData/{moment}', filename)
            Top3filepath = os.path.join(current_script_directory, Top3filepath)

            top_dir = f'Keypoint_data/Top3ProfessionalKeypointData/{moment}'
            top_dir = os.path.join(current_script_directory, top_dir)

            if not os.path.exists(top_dir):
                os.makedirs(top_dir)
                # print("Directory created")

            #read json file 
            with open(filepath, 'r') as file:
                professional_keypoint_data = json.load(file)
            
            # print(Top3filepath)
            #write JSON data to directory
            with open(Top3filepath, 'w') as file:
                json.dump(professional_keypoint_data, file)
            
    return Top3Score

if __name__ == "__main__":
    pro_comparison()