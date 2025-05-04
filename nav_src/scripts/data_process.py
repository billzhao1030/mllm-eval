import os
import json
import numpy as np

def filter_longest_action_plans(path):
    samples = json.load(open(path, "r"))
    instr_id_to_sample = {}
    with open("./datasets/R2R/annotations/R2R_val_72_action_plan_filtered.json", "r") as f:
        data = json.load(f)
    instr_id_list = []
    for item in data:
        instr_id_list.append(item["instr_id"])

    for sample in samples:
        instr_id = sample["instr_id"]

        if instr_id not in instr_id_list:
            instr_id_to_sample[instr_id] = sample

    return list(instr_id_to_sample.values())

def modify_heading_angles(cur_heading, observation):
    # Split the observation string into lines
    lines = observation.split('\n')
    
    # Initialize an empty list to store the modified lines
    modified_lines = []
    
    # Iterate through the lines
    for line in lines:
        # If the line starts with 'heading', modify the heading angles
        if line.startswith('heading'):
            # Extract the two angles from the line
            angle1, angle2 = map(int, line[8:-1].split(' - '))
            
            # Calculate the new angles relative to the current heading
            new_angle1 = angle1 - cur_heading
            new_angle2 = angle2 - cur_heading
            
            # Construct the modified line
            modified_line = f'heading {new_angle1} - {new_angle2}:'
        else:
            # If the line doesn't start with 'heading', keep it as it is
            modified_line = line
        
        # Add the modified line to the list
        modified_lines.append(modified_line)
    
    # Join the modified lines into a single string
    modified_observation = '\n'.join(modified_lines)
    
    return modified_observation

def observation_to_dict(observation):
    # Split the observation string into lines
    lines = observation.split('\n')
    
    # Initialize an empty dictionary to store the heading ranges and their corresponding strings
    observation_dict = {}
    
    # Initialize a variable to store the current heading range
    current_heading = None
    
    # Iterate through the lines
    for line in lines:
        # If the line starts with 'heading', update the current heading range
        if line.startswith('heading'):
            current_heading = line
            # Initialize an empty list to store the strings for the current heading range
            observation_dict[current_heading] = []
        else:
            # If the line doesn't start with 'heading', add it to the list of strings for the current heading range
            observation_dict[current_heading].append(line)
    
    # Join the strings for each heading range into a single string
    for heading in observation_dict:
        observation_dict[heading] = '\n'.join(observation_dict[heading])
    
    return observation_dict

def observation_to_list(observation):
    # Split the observation string into lines
    lines = observation.split('\n')
    
    # Initialize an empty list to store the heading strings
    observation_list = []
    
    # Initialize a variable to store the current heading string
    current_heading_str = ''
    
    # Iterate through the lines
    for line in lines:
        # If the line starts with 'heading', add the current heading string to the list (without the heading part) and reset it
        if line.startswith('heading'):
            if current_heading_str:
                observation_list.append(current_heading_str.strip())
            current_heading_str = ''
        else:
            # If the line doesn't start with 'heading', add it to the current heading string
            current_heading_str += line + '\n'
    
    # Add the last heading string to the list (without the heading part)
    observation_list.append(current_heading_str.strip())
    
    return observation_list

def format_observation_output(observation_list, heading_angle):
    # Define the directions
    directions = ['front', 'front right', 'right', 'rear right', 'rear', 'rear left', 'left', 'front left']

    # Calculate the range of heading angles belonging to each direction
    range_idx = int((heading_angle - 22.5) // 45) + 1
    obs_idx = [(i + range_idx) % 8 for i in range(8)]
    
    # Calculate the relative angle ranges based on the heading angle
    angle_ranges = [(angle - 22.5 - heading_angle, angle + 22.5 - heading_angle) for angle in range(0, 360, 45)]
    
    # Function to normalize an angle to the range of -180 to 180
    def normalize_angle(angle):
        while angle > 180:
            angle -= 360
        while angle <= -180:
            angle += 360
        return angle
    
    def angle_to_left_right(angle):
        return f"left {-angle}" if angle < 0 else f"right {angle}"
    
    # Initialize an empty list to store the formatted strings
    formatted_strings = []
    
    # Iterate through the directions, angle ranges, and observation strings
    for direction, idx in zip(directions, obs_idx):
        # Calculate the relative angles and normalize them
        rel_angle1 = normalize_angle(angle_ranges[idx][0])
        rel_angle2 = normalize_angle(angle_ranges[idx][1])

        # Convert the angles to "left n" or "right n"
        left_right1 = angle_to_left_right(rel_angle1)
        left_right2 = angle_to_left_right(rel_angle2)
        
        # Create the formatted string
        formatted_string = f"{direction}, range ({left_right1} to {left_right2}): \n'{observation_list[idx]}'"
        
        # Add the formatted string to the list
        formatted_strings.append(formatted_string)
    
    # Join the formatted strings into a single output string
    output_string = '\n'.join(formatted_strings)
    
    return output_string

if __name__ == "__main__":
    # main()
    # filtered_data = filter_longest_action_plans("./datasets/R2R/exprs/default/preds/submit_val_72_action_plan.json")
    # with open("./datasets/R2R/exprs/default/preds/submit_val_72_action_plan_filtered.json", "w") as f:
    #     json.dump(filtered_data, f)

    # obs_dir = '../datasets/R2R/observations/'
    # obs_files = os.listdir(obs_dir)
    # output_dir = '../datasets/R2R/observations_list/'

    # for obs_file in obs_files:
    #     obs_path = os.path.join(obs_dir, obs_file)
    #     with open(obs_path) as f:
    #         obs = json.load(f)
    #     for viewpointID, observation in obs.items():
    #         obs_list = observation_to_list(observation)
    #         obs[viewpointID] = obs_list
        
    #     output_path = os.path.join(output_dir, obs_file)
    #     # make sure the output directory exists
    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #     with open(output_path, 'w') as f:
    #         json.dump(obs, f, indent=2)


    observation = "heading 0 - 45:\ndown: a yellow car with a yellow top\nmiddle: a room with a large window and a tv\ntop: a ceiling with a television hanging from it\nheading 45 - 90:\ndown: a white counter with a light on it\nmiddle: a room with a wooden ceiling and wooden walls\ntop: a wooden ceiling with a light shining on it\nheading 90 - 135:\ndown: a bathroom with a purple floor and a door\nmiddle: a wooden closet with a wooden door\ntop: a bathroom with a wooden ceiling and a light\nheading 135 - 180:\ndown: a bathroom with a yellow tiled wall\nmiddle: a bathroom with a tiled shower and a clock\ntop: a bathroom with a window with a small house in it\nheading 180 - 225:\ndown: a tiled ceiling with white tiles and a white floor\nmiddle: a bathroom with a tiled wall and a white tile\ntop: a tiled bathroom with yellow tiles and white grout\nheading 225 - 270:\ndown: a bathroom with a yellow tiled wall\nmiddle: a bathroom with a tiled wall and a toilet\ntop: a bathroom with a yellow tiled wall\nheading 270 - 315:\ndown: a purple floor in a bathroom with a window\nmiddle: a bathroom with a window and a white vase\ntop: a bathroom with a wooden ceiling and a light fixture\nheading 315 - 360:\ndown: a glass counter with a light on it\nmiddle: a room with a yellow glass table and chairs\ntop: a wooden ceiling with a light fixture and a fan\n"
    observation_list = observation_to_list(observation)
    heading_angle =124.345

    formatted_output = format_observation_output(observation_list, heading_angle)
    print(formatted_output)