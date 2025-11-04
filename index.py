import streamlit as st
import os
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch
from PIL import Image
import csv
import cv2
import numpy as np

sys.path.insert(0, 'System_folder/golfdb/')
from System_folder.golfdb.AmateurSegmentation import process_segmentation

from System_folder.Pose_estimation.AmateurKeypointToJson import keypoints

from System_folder.Pose_estimation.JsonComparison import pro_comparison

from System_folder.Pose_estimation.JsonAvg import average_keypoints

from System_folder.FeedBack import feedback

def display_image_with_keypoints(image_path, keypoints_json_path):
    # Load keypoints from JSON file
    with open(keypoints_json_path, 'r') as json_file:
        keypoints_data = json.load(json_file)

     # Extract category name from image_path
    title = os.path.basename(os.path.dirname(image_path))

    # Handle pro dir name 
    if title == "ImageData":
        title = os.path.basename(os.path.dirname(keypoints_json_path))

    # Load the original image
    image = Image.open(image_path)

    # Create a figure and plot the image
    fig, ax = plt.subplots()
    ax.imshow(image, extent=[0, 1, 1, 0])

    # Set the limits of the axes between 0 and 1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Invert the y-axis
    ax.invert_yaxis()

    # Extract and plot keypoints
    for keypoint, coordinates in keypoints_data.items():
        x, y, _ = coordinates  # Ignore the third coordinate for simplicity

        # Plot keypoints on the image
        ax.add_patch(Circle((x, y), radius=0.005, color='red'))  # You can customize the appearance

    # Define the connections between keypoints for creating the skeleton
    connections = [
        ("LshoulderToElbow", [11, 13]),
        ("LelbowToWrist", [13, 15]),
        ("RshoulderToElbow", [12, 14]),
        ("RelbowToWrist", [14, 16]),
        ("LshouldertoHip", [11, 23]),
        ("RshouldertoHip", [12, 24]),
        ("LHipToKnee", [23, 25]),
        ("RHipToKnee", [24, 26]),
        ("LKneetoAnkle", [25, 27]),
        ("RkneetoAnkle", [26, 28]),
    ]

    # Draw lines between specified keypoints
    for connection_name, keypoints_pair in connections:
        keypoint1, keypoint2 = keypoints_pair
        x1, y1, _ = keypoints_data[str(keypoint1)]
        x2, y2, _ = keypoints_data[str(keypoint2)]

        # Plot line between keypoints
        ax.add_patch(ConnectionPatch((x1, y1), (x2, y2), "data", "data", color='blue'))
    
    # Set the title of the plot
    ax.set_title(title)

    # Display the image with keypoints in Streamlit
    # st.pyplot(fig)
    return fig


def display_images(image_paths, width=150):
    # Specify the number of columns based on the number of images
    num_columns = len(image_paths)

    # Create columns layout
    columns = st.columns(num_columns)

    for path, column in zip(image_paths, columns):
        column.image(path, width=width)

def process_video(image_paths):
    
    # Process the video and return the result
    process_segmentation()

    # # Display images side by side
    display_images(image_paths)

    result = "This is the processed result."
    return result

def resize_video(input_path, output_path, dimension=160):
    """
    Resize a video to the specified dimension (width x height).
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (dimension, dimension))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (dimension, dimension))
        out.write(resized_frame)

    cap.release()
    out.release()

def resize_images(input_paths, output_paths, dimension=160):
    """
    Resize multiple images while maintaining the original aspect ratios.
    """
    for input_path, output_path in zip(input_paths, output_paths):
        image = cv2.imread(input_path)
        
        # Get the original image dimensions
        height, width, _ = image.shape

        # Calculate the aspect ratio
        aspect_ratio = width / height

        # Calculate the new dimensions while maintaining the aspect ratio
        new_width = int(dimension * aspect_ratio)
        new_height = dimension

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height))

        # Save the resized image
        cv2.imwrite(output_path, resized_image)

def main():
    st.title("AI golf swing processing app")
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        # Save the video 
        video_path = save_uploaded_file(uploaded_file)
        # print(video_path)

        # Resize the video
        # resize_video(video_path, video_path, 160) 

        image_paths = ["System_folder/golfdb/ImageData/Am_Swing_events/Address/Address1.jpg",
                   "System_folder/golfdb/ImageData/Am_Swing_events/Toe-up/Toe-up1.jpg",
                   "System_folder/golfdb/ImageData/Am_Swing_events/Mid-Backswing (arm parallel)/Mid-backswing (arm parallel)1.jpg",
                   "System_folder/golfdb/ImageData/Am_Swing_events//Top/Top1.jpg",
                   "System_folder/golfdb/ImageData/Am_Swing_events/Mid-Downswing (arm parallel)/Mid-downswing (arm parallel)1.jpg",
                   "System_folder/golfdb/ImageData/Am_Swing_events/Impact/Impact1.jpg",
                   "System_folder/golfdb/ImageData/Am_Swing_events/Mid-Follow-Through (shaft parallel)/Mid-follow-through (shaft parallel)1.jpg",
                   "System_folder/golfdb/ImageData/Am_Swing_events/Finish/Finish1.jpg"]
                
        # Display the text 
        st.subheader("Swing sequences:")

        # Process the video
        processed_result = process_video(image_paths)

        keypoints_paths = ["System_folder/Pose_estimation/Keypoint_data/AmateurKeypointData/Address_keypointData.json",
                   "System_folder/Pose_estimation/Keypoint_data/AmateurKeypointData/Toe-up_keypointData.json",
                   "System_folder/Pose_estimation/Keypoint_data/AmateurKeypointData/Mid-Backswing (arm parallel)_keypointData.json",                  
                   "System_folder/Pose_estimation/Keypoint_data/AmateurKeypointData/Top_keypointData.json",
                   "System_folder/Pose_estimation/Keypoint_data/AmateurKeypointData/Mid-Downswing (arm parallel)_keypointData.json",                   
                   "System_folder/Pose_estimation/Keypoint_data/AmateurKeypointData/Impact_keypointData.json",                   
                   "System_folder/Pose_estimation/Keypoint_data/AmateurKeypointData/Mid-Follow-Through (shaft parallel)_keypointData.json",                  
                   "System_folder/Pose_estimation/Keypoint_data/AmateurKeypointData/Finish_keypointData.json"]                   

        # keypoints_paths = ["System_folder/Pose_estimation/Keypoint_data/ProfessionalKeypointData/Address/Address1.json",
        #            "System_folder/Pose_estimation/Keypoint_data/ProfessionalKeypointData/Toe-up/Toe-up1.json",
        #            "System_folder/Pose_estimation/Keypoint_data/ProfessionalKeypointData/Mid-Backswing (arm parallel)/Mid-Backswing (arm parallel)1.json",                  
        #            "System_folder/Pose_estimation/Keypoint_data/ProfessionalKeypointData/Top/Top1.json",
        #            "System_folder/Pose_estimation/Keypoint_data/ProfessionalKeypointData/Mid-Downswing (arm parallel)/Mid-Downswing (arm parallel)1.json",                   
        #            "System_folder/Pose_estimation/Keypoint_data/ProfessionalKeypointData/Impact/Impact1.json",                   
        #            "System_folder/Pose_estimation/Keypoint_data/ProfessionalKeypointData/Mid-Follow-Through (shaft parallel)/Mid-Follow-Through (shaft parallel)1.json",                  
        #            "System_folder/Pose_estimation/Keypoint_data/ProfessionalKeypointData/Finish/Finish1.json"]     
        
        # Extract keypoints from amateur frames 
        st.subheader("Swing keypoints:")
        # resize_images(image_paths, image_paths, 160)
        keypoints()

        # Specify the number of columns based on the number of images
        num_columns = len(image_paths)

        # Create columns layout
        columns = st.columns(num_columns // 2)

        # Display images with keypoints in Streamlit on three lines, pyplotting the column once every two elements
        for i in range(0, len(image_paths), 2):
            image_path1, keypoints_path1 = image_paths[i], keypoints_paths[i]
            image_path2, keypoints_path2 = image_paths[i + 1], keypoints_paths[i + 1]

            fig1 = display_image_with_keypoints(image_path1, keypoints_path1)
            columns[i // 2].pyplot(fig1)

            fig2 = display_image_with_keypoints(image_path2, keypoints_path2)
            columns[i // 2].pyplot(fig2)
            

        # Professional comparison
        st.header("Pro swing:")
                
        # Create a loading spinner
        with st.spinner("Processing professional swings..."):
            top3 = pro_comparison()

        # Graduation
        top_value = top3[0]

        # Calculate the original grade in the range [7, 10]
        original_grade = (1 - top_value - 0.8) * 10/0.05

        # Round the rescaled grade to keep only two decimal places
        rounded_rescaled_grade = round(original_grade, 2)

        # Display the rounded rescaled grade in a subheader using f-string
        st.subheader(f"Swing grade: {rounded_rescaled_grade}")


        # Pro keypoints
        average_keypoints()
        st.text(" The best swing for your profile :")
        
        keypoints_pro = ["System_folder/Pose_estimation/Keypoint_data/Top3ProfessionalKeypointData/Address/Address_averages.json",
            "System_folder/Pose_estimation/Keypoint_data/Top3ProfessionalKeypointData/Toe-up/Toe-up_averages.json",
            "System_folder/Pose_estimation/Keypoint_data/Top3ProfessionalKeypointData/Mid-Backswing (arm parallel)/Mid-Backswing (arm parallel)_averages.json",                  
            "System_folder/Pose_estimation/Keypoint_data/Top3ProfessionalKeypointData/Top/Top_averages.json",
            "System_folder/Pose_estimation/Keypoint_data/Top3ProfessionalKeypointData/Mid-Downswing (arm parallel)/Mid-Downswing (arm parallel)_averages.json",                   
            "System_folder/Pose_estimation/Keypoint_data/Top3ProfessionalKeypointData/Impact/Impact_averages.json",                   
            "System_folder/Pose_estimation/Keypoint_data/Top3ProfessionalKeypointData/Mid-Follow-Through (shaft parallel)/Mid-Follow-Through (shaft parallel)_averages.json",                  
            "System_folder/Pose_estimation/Keypoint_data/Top3ProfessionalKeypointData/Finish/Finish_averages.json"]    
        
        image_pro = ["System_folder/golfdb/ImageData/Sports_1393.jpg",
                     "System_folder/golfdb/ImageData/Sports_1393.jpg",
                     "System_folder/golfdb/ImageData/Sports_1393.jpg",
                     "System_folder/golfdb/ImageData/Sports_1393.jpg",
                     "System_folder/golfdb/ImageData/Sports_1393.jpg",
                     "System_folder/golfdb/ImageData/Sports_1393.jpg",
                     "System_folder/golfdb/ImageData/Sports_1393.jpg",
                     "System_folder/golfdb/ImageData/Sports_1393.jpg",]
        
        # Specify the number of columns based on the number of images
        num_columns = len(image_pro)

        # Create columns layout
        columns = st.columns(num_columns // 2)

        # Display images with keypoints in Streamlit on three lines, pyplotting the column once every two elements
        for i in range(0, len(image_pro), 2):
            image_path1, keypoints_path1 = image_pro[i], keypoints_pro[i]
            image_path2, keypoints_path2 = image_pro[i + 1], keypoints_pro[i + 1]

            fig1 = display_image_with_keypoints(image_path1, keypoints_path1)
            columns[i // 2].pyplot(fig1)

            fig2 = display_image_with_keypoints(image_path2, keypoints_path2)
            columns[i // 2].pyplot(fig2)

        # Feedback
        # Display the text output
        st.header("Recommendations:")

        # Create a loading spinner
        with st.spinner("Processing recommendations..."):
            recommandations = feedback()

        # List of filenames
        filenames = [
            "Mid-Downswing (arm parallel)",
            "Top",
            "Impact",
            "Finish",
        ]
        i = 0
        for filename in filenames:
            st.write(f" Advice for : {filename}")
            st.write(recommandations[i])
            i += 1

def save_uploaded_file(uploaded_file):

    # Save the uploaded file to the Amateur-vids location
    file_path = "System_folder/golfdb/VideoData/Amateur-vids/video.mp4"

    # Create the directories if they don't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
   
    with open(file_path, "wb") as file:
        file.write(uploaded_file.read())

    return file_path

    
if __name__ == "__main__":
    main()
