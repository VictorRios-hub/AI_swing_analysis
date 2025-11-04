import json
import math
import openai
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#this system calculates the euclidean distance of the Top3 professional average and the amateur average, this data is entered into an OpenAI NLP prompt 
#which will return inference on the data

# Load the API key from the JSON file
def load_api_key():
    with open('System_folder/config.json', 'r') as file:
        config = json.load(file)
    return config['api_key']

def feedback():

    openai.api_key = load_api_key()

    # List of filenames
    filenames = [
        "Mid-Downswing (arm parallel)",
        "Top",
        "Impact",
        "Finish",
    ]
    #keypoint groups
    keypoint_lists = {
        "LshoulderToElbow": [11, 13],
        "LelbowToWrist": [13,15],
        "RshoulderToElbow": [12, 14],
        "RelbowToWrist": [14,16],
        "LshouldertoHip": [11, 23],
        "RshouldertoHip": [12, 24],
        "LHipToKnee": [23, 25],
        "RHipToKnee": [24, 26],
        "LKneetoAnkle":[25,27],
        "RkneetoAnkle":[26,28]
    }
    #description of each keypoint
    keypoint_descriptions = {
        11: "left shoulder",
        12: "right shoulder",
        13: "left elbow",
        14: "right elbow",
        15: "left wrist",
        16: "right wrist",
        23: "left hip",
        24: "right hip",
        25: "left knee",
        26: "right knee",
        27: "left ankle",
        28: "right ankle",
    }

    all_feedback = []
    
    # Loop through filenames
    for filename in filenames:
        
        # # Get the directory of the current script
        current_script_directory = os.path.dirname(os.path.abspath(__file__))
        am_directory = f'Pose_estimation/Keypoint_data/AmateurKeypointData/{filename}_keypointData.json'
        
        # # Construct the absolute path
        am_directory = os.path.join(current_script_directory, am_directory)

        with open(am_directory, 'r') as f:
            data1 = json.load(f)

        pro_directory = f'Pose_estimation/Keypoint_data/Top3ProfessionalKeypointData/{filename}/{filename}_averages.json'
        
        # # Construct the absolute path
        pro_directory = os.path.join(current_script_directory, pro_directory)

        with open(pro_directory, 'r') as f:
            data2 = json.load(f)

        #function for calculating euclidean distance
        def euclidean_distance(x1, y1, z1, x2, y2, z2):
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


        #finds keypoint distances through euclidean distance 
        max_avg_distance = 0
        max_avg_distance_list_name = None
        max_avg_distance_original_values = []

        for list_name, keypoint_list in keypoint_lists.items():
            total_distance = 0
            total_cosine = 0
            list_original_values = []
            v1 =[]
            v2 = []
            
            for key in keypoint_list:
                key = str(key)
                if key in data1 and key in data2:
                    x1, y1, z1 = data1[key]
                    x2, y2, z2 = data2[key]
                    distance = euclidean_distance(x1, y1, z1, x2, y2, z2)

                    # #function for calculation cosinus similarity
                    # v1 = np.array([x1, y1])
                    # v2 = np.array([x2, y2])
                    # cosine_similarity_distance = 1 - cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))

                    total_distance += distance
                    # total_cosine += cosine_similarity_distance
                    list_original_values.append(((x1, y1, z1), (x2, y2, z2)))

            avg_distance = total_distance / len(keypoint_list)

            if avg_distance > max_avg_distance:
                max_avg_distance = avg_distance
                max_avg_distance_list_name = list_name
                max_avg_distance_original_values = list_original_values

        #generates feedback by prompting gpt 3.5 turbo with keypoint differences between amateur and professional average
        def generate_feedback(list_name, original_values):
            feedback = []

            for i, values in enumerate(original_values):
                amateur_coords, pro_coords = values
                keypoint = keypoint_lists[list_name][i]
                keypoint_description = keypoint_descriptions[keypoint]

                diff_x = amateur_coords[0] - pro_coords[0]
                diff_y = amateur_coords[1] - pro_coords[1]
                diff_z = amateur_coords[2] - pro_coords[2]
                #print(diff_x , diff_y , diff_z)
                # prompt = (f"You are a profesionnal master of swing golf who help the user become a master too."
                #     f"For the {keypoint_description} (keypoint {keypoint}), the amateur data coordinates are {amateur_coords}, and the professional data coordinates are {pro_coords}. "
                #     f"The differences in x, y, and z are {diff_x}, {diff_y}, and {diff_z}, respectively. "
                #     f"How can the user improve their golf swing based on this data?, do not return to data that references specific x, y or z values, or the professional swing, instead make inferences based on these values."
                #     f"Give advices on how to improve the amateur swing in 3 sentences, do not refer to the coordinates only on the human joints")
                
                # prompt = (f"As a professional golf swing master, I've analyzed the user's swing data. For the {keypoint_description} (keypoint {keypoint}), observe the disparities in coordinates ({amateur_coords} vs. {pro_coords}). The euclidian distance differences in x, y, and z are {diff_x}, {diff_y}, and {diff_z}. "
                # f"Based on this data, outline general rules for refining the {keypoint_description} alignment, ensuring smoother transitions, and maintaining consistent posture and follow-through. Avoid explicit references to coordinates or specific x, y, or z values.")

                # prompt = (f"As a professional golf swing expert, I've scrutinized the user's swing data. Focusing on the {keypoint_description} (keypoint {keypoint}), note the disparities in coordinates between the amateur ({amateur_coords}) and the professional ({pro_coords}). Analyze the differences in x, y, and z, which are {diff_x}, {diff_y}, and {diff_z}, respectively. "
                #         f"Provide precise guidelines, without numerical references, for refining {keypoint_description} alignment, ensuring seamless transitions, and maintaining consistent posture and follow-through. Your expertise will guide the user toward an improved golf swing."
                #         f"Give only few sentences for advices and don't focus on explanation")

                # prompt = (f"You are a profesionnal master of swing golf who help the golf student user."
                #     f"For the {keypoint_description} (keypoint {keypoint}), the amateur data coordinates are {amateur_coords}, and the professional data coordinates are {pro_coords}."
                #     f"Analyze the disparities in coordinates between amateur and pro data. Differences in x, y, and z: {diff_x}, {diff_y}, {diff_z}. "
                #     f"Offer succinct, rule-based advice for refining {keypoint_description} alignment, ensuring smooth transitions, and maintaining posture. Your expertise guides the user to an improved golf swing."
                #     f"Give only 2 advices for differents keypoints.")
                
                # prompt = (f"You are a profesionnal master of swing golf who help the golf student user."
                #     f"For the {keypoint_description} (keypoint {keypoint}), the amateur data coordinates are {amateur_coords}, and the professional data coordinates are {pro_coords}."
                #     f"Find the first advice. Then for the {keypoint_descriptions} and the {keypoint_lists} analyze the disparities in coordinates between amateur and pro data"
                #     f"The amateur data coordinates are {data1}, and the professional data coordinates are {data2}."
                #     f"The differences in x, y, and z: {diff_x}, {diff_y}, {diff_z}."
                #     f"Offer succinct; rule-based advice for refining specific {keypoint_descriptions} alignment, ensuring smooth transitions, and maintaining posture."
                #     f"Your expertise guides the user to an improved golf swing. Give 2 specific advices for differents keypoints in only several words. Don't specify keypoints numbers, specify the body joints and keypoints name instead"
                #     f"Use bullet points to list the advices. "
                #     f"Before the output, read your response again and make sure to give me the best response following my rules, this is very important the greatest possible accuracy")

                prompt = (f"As a professional golf swing master, you assist golf students in refining their technique. Focus on the {keypoint_description} (keypoint {keypoint}), comparing amateur ({amateur_coords}) and professional ({pro_coords}) data coordinates. Differences in x, y, z: {diff_x}, {diff_y}, {diff_z}."
                    f"First advice: Analyze disparities concerning {keypoint_descriptions} and {keypoint_lists} in amateur and professional data. Amateur data coordinates ({data1}), professional data coordinates ({data2}). "
                    f"Provide two specific, succinct, rule-based advices for refining {keypoint_descriptions} alignment, ensuring smooth transitions, and maintaining posture. Use bullet points to list the advices."
                    f"Your expertise guides the user to an improved golf swing in real life, don't use keypoint number in your answer. Before output, ensure adherence to rules for optimal accuracy.")

                response = openai.Completion.create(
                    engine="gpt-3.5-turbo-instruct",
                    prompt=prompt,
                    max_tokens=300,
                    n=1,
                    stop=None,
                    temperature=0.6,
                    
                    )

                feedback_item = response.choices[0].text.strip()
                feedback.append(feedback_item)
                
                #num_tokens = num_tokens_from_string(item, "p50k_base")
                #print(num_tokens)
            return feedback 

        #feedback is returned
        feedback = generate_feedback(max_avg_distance_list_name, max_avg_distance_original_values)
        # print("-------------------------------------------------------------------------------")
        print(f" Advice for : {filename}")
        for item in feedback:
            print("\n" + item)
            # all_feedback.append(f" Advice for : {filename}")
            all_feedback.append(item)
        # print("-------------------------------------------------------------------------------")
        #num_tokens = num_tokens_from_string(feedback, "p50k_base")
        #print(num_tokens)
    
    return all_feedback

if __name__ == "__main__":
    feedback()