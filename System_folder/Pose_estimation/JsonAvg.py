import json
import os

def average_keypoints():

    # Get the directory of the current script
    root_directory = 'Keypoint_data/Top3ProfessionalKeypointData'
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the absolute path
    root_directory = os.path.join(current_script_directory, root_directory)

    # Check if the directory exists
    if not os.path.exists(root_directory):
        # If not, create the directory
        os.makedirs(root_directory)
        print(f"Directory '{root_directory}' created.")
    else:
        print(f"Directory '{root_directory}' already exists.")


    #loop through all subdirectories in the root
    for directory, subdirectories, filenames in os.walk(root_directory):
        if directory == root_directory:
            
            continue

        print(filenames)
        #initialize dictionaries to track total and count
        totals = {}
        counts = {}
        max_len = {}

        for filename in filenames:
            if filename.endswith('.json'):
                with open(os.path.join(directory, filename), 'r') as f:
                    data = json.load(f)

                    
                    #loop through each key-value pair in the file
                    for obj_id, values in data.items():

                        
                        #initialize the total and count per item
                        for i, value in enumerate(values):
                            item_key = f"{obj_id}_{i}"
                            if item_key not in totals:
                                totals[item_key] = 0
                                counts[item_key] = 0

                            
                            #add the value to the running total and increment the count for the item
                            totals[item_key] += value
                            counts[item_key] += 1

                        
                            #track the maximum length of the list per object ID
                            if obj_id not in max_len or i >= max_len[obj_id]:
                                max_len[obj_id] = i + 1

        #calculate the averages
        result = {}
        for item_key in totals:
            average = totals[item_key] / counts[item_key]
            obj_id, i = item_key.split('_')
            i = int(i)

            
            #initialize list for object ID if it doesnt exist
            if obj_id not in result:
                result[obj_id] = [0] * max_len[obj_id]

            
            #update value for this item in the result dict
            result[obj_id][i] = average

        
        #get the folder name from the directory path
        folder_name = os.path.basename(directory)

        
        #write the output to a new JSON file
        output_filepath = os.path.join(directory, f'{folder_name}_averages.json')
        with open(output_filepath, 'w') as f:
            json.dump(result, f)

if __name__ == "__main__":
    average_keypoints()






