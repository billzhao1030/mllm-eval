"""Agent that interacts with Matterport3D simulator via a hierarchical planning approach."""
import json
import yaml
import re
import warnings
import torch
import numpy as np
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple, Dict, Union

from env import R2RNavBatch
from argparse import Namespace
from agent_base import BaseAgent

from utils.data import AgentAction, AgentFinish, OutputParserException
from prompt.planner_prompt import NAVGPT_2_PROMPT

FINAL_ANSWER_ACTION = "Final Answer:"
EXCEPTION_TOOL_NAME = "_Exception"
MAX_SCRATCHPAD_LENGTH = 7000

MISSING_ANSWER_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Answer:' in LLM output"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)
VIEWPOINT_NOT_IN_NAVIGABLE_ERROR_MESSAGE = (
    "Invalid Choice: Viewpoint not in navigable viewpoints"
)
ANSWER_NOT_VALID_ERROR_MESSAGE = (
    "Invalid Choice: Answer is not a valid number"
)

class NavAgent(BaseAgent):
    def __init__(
            self,
            env: R2RNavBatch,
            config: Namespace):
        """
        Initialize the LLM Navigation Agent.

        Args:
            env: The Matterport3D environment.
            config: The configuration.
        """
        super().__init__(env)
        self.config = config

        if self.config.llm_model_name == "NavGPT_InstructBLIP":
            from LLMs.NavGPT_InstructBLIP import NavGPTInstruct

            self.model = NavGPTInstruct(
                config = self.config,
            )
        elif self.config.llm_model_name == "Emu-14B":
            from LLMs.NavGPT_EMU import Custom_Emu
            
            self.model = Custom_Emu(
                config = self.config,
            )

        self.history: List[str] = None

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        if self.config.llm_model_name == "Emu-14B":
            regex = (
                r"(?<=The answer is \{[\"'])[a-f0-9]{32}(?=[\"']:)"
            )
        else:
            regex =(
                r"(?<=Answer:\s)(-?\d+(\.\d+)?)"
            )
        action_match = re.search(regex, text)
        if action_match:
            action = 'make_action'
            tool_input = action_match.group().strip()
            if self.config.action_space == 'viewpoint' and tool_input not in self.env.env.sims[0].navigable_dict.keys():
                raise OutputParserException(
                    f"Could not make corresponding action: `{text}`",
                    observation=VIEWPOINT_NOT_IN_NAVIGABLE_ERROR_MESSAGE,
                    llm_output=text,
                    send_to_llm=True,
                )

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        if not re.search(r"Answer:", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ANSWER_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(r'\nAnswer:\s*(\d+(\.\d+)?)\s*$', text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=ANSWER_NOT_VALID_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")

    def get_history(self, obs, angle) -> str:
        '''Return the history of actions taken.'''
        # history = f'{angle}\nCurrent viewpoint "{obs["viewpoint"]}": Scene from the viewpoint is a {obs["obs_summary"]}'
        history = f'{angle}\n Scene from the viewpoint is a {obs["obs_summary"]}'

        return history

    def get_navigable_str(self, cur_heading, cur_elevation, navigable) -> str:
        '''Return the navigable viewpoints as a string.'''
        navigable_str = ''

        for vp, items in navigable.items():
            heading = np.rad2deg(items['heading'])
            elevation = np.rad2deg(items['elevation'])
            distance = items['distance']
            rel_heading = heading - cur_heading
            rel_elevation = elevation - cur_elevation

            if self.config.use_relative_angle:
                navigable_str += f"'{vp}':\nheading: {rel_heading:.2f}, elevation: {rel_elevation:.2f}, distance: {distance:.2f}\n"
            else:
                navigable_str += f"'{vp}':\nheading: {heading:.2f}, elevation: {elevation:.2f}, distance: {distance:.2f}\n"

        return navigable_str

    def modify_heading_angles(self, heading_angle, observation_list, candidate_dict, object_list):
        # Function to normalize an angle to the range of -180 to 180
        def normalize_angle(angle):
            while angle > 180:
                angle -= 360
            while angle <= -180:
                angle += 360
            return angle

        def angle_to_left_right(angle):
            return f"left {-angle:.2f}" if angle < 0 else f"right {angle:.2f}"

        # Define the directions
        directions = ['Front', 'Front Right', 'Right', 'Rear Right', 'Rear', 'Rear Left', 'Left', 'Front Left']

        # Calculate the range of heading angles belonging to each direction
        range_idx = int((heading_angle - 22.5) // 45) + 1
        obs_idx = [(i + range_idx) % 8 for i in range(8)]

        # Initialize a dictionary to store the candidate viewpoints for each direction
        candidate_range = {}
        if not self.config.use_navigable:
            for viewpoint_id, viewpoint_data in candidate_dict.items():
                viewpoint_heading = np.rad2deg(viewpoint_data['heading'])
                vp_range_idx = int((viewpoint_heading - 22.5) // 45) + 1
                rel_viewpoint_heading = viewpoint_heading - heading_angle
                rel_viewpoint_heading = normalize_angle(rel_viewpoint_heading)
                rel_viewpoint_heading = angle_to_left_right(rel_viewpoint_heading)
                vp_description = rel_viewpoint_heading + f', {viewpoint_data["distance"]:.2f}m'
                # rel_range_idx = (vp_range_idx - range_idx) % 8
                candidate_range.setdefault(vp_range_idx, {}).update({viewpoint_id: vp_description})

        # Calculate the relative angle ranges based on the heading angle
        angle_ranges = [(angle - 22.5 - heading_angle, angle + 22.5 - heading_angle) for angle in range(0, 360, 45)]

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

            # Add the objects to the formatted string
            object_dict = {}
            if len(object_list[idx]) > 0:
                object = object_list[idx]
                for obj, obj_data in object.items():
                    rel_obj_heading = obj_data['heading'] - heading_angle
                    rel_obj_heading = normalize_angle(rel_obj_heading)
                    rel_obj_heading = angle_to_left_right(rel_obj_heading)
                    object_dict[obj] = f'{rel_obj_heading}, {obj_data["distance"]:.2f}m'
                formatted_string += f'\n{direction} Objects in 3m: {object_dict}'
            else:
                formatted_string += f'\n{direction} Objects in 3m: None'

            # Add the candidate viewpoints to the formatted string
            if candidate_range.get(idx):
                formatted_string += f'\n{direction} Navigable Viewpoints:{candidate_range[idx]}'
            else:
                formatted_string += f'\n{direction} Navigable Viewpoints: None'

            # Add the formatted string to the list
            formatted_strings.append(formatted_string)

        # Join the formatted strings into a single output string
        output_string = '\n'.join(formatted_strings)

        return output_string

    def init_trajecotry(self, obs: List[dict]):
        """Initialize the trajectory with the given observation."""
        # Record the navigation path
        self.traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'llm_outputs': [],
            'details': [],
        } for ob in obs]
        # Record the history of actions taken
        self.history = [f'Navigation start, no actions taken yet.\nCurrent viewpoint "{obs[0]["viewpoint"]}": Scene from the viewpoint is a {obs[0]["obs_summary"]}']

        print(f"\nExcuating instruction:\n{obs[0]['instr_id']}: {obs[0]['instruction']}")

    def make_equiv_action(self, actions: List[str]) -> str:
        """
        Interface between Panoramic view and Egocentric view
        Take in the next viewpoint ID and move the agent to that viewpoint
        return the turned angle and new observation
        """
        def normalize_angle(angle):
            while angle > 180:
                angle -= 360
            while angle <= -180:
                angle += 360
            return angle

        def angle_to_left_right(angle):
            return f"left {-angle:.2f}" if angle < 0 else f"right {angle:.2f}"

        # Get current agent facing angle
        cur_obs = self.env._get_obs()[0]
        cur_heading = np.rad2deg(cur_obs['heading'])
        # Get the target viewpoint ID
        target = cur_obs['viewpoint']
        if self.config.action_space == 'angle':
            res = float('inf')
            for key, val in cur_obs['candidate'].items():
                cur_res = np.abs(np.rad2deg(val['heading']) - float(actions[0]))
                if cur_res < res:
                    res = cur_res
                    target = [key]
        elif self.config.action_space == 'viewpoint':
            target = actions

        # Make the action
        new_obs = self.env.step(target)[0]
        new_heading = np.rad2deg(new_obs['heading'])
        # Record the trajectory
        self.traj[0]['path'].append(self.env.env.sims[0].gmap.bfs_shortest_path(cur_obs['viewpoint'], target[0])[1:])
        # Calculate the turned angle
        turned_angle = new_heading - cur_heading
        # Generate action description
        cur_heading = angle_to_left_right(normalize_angle(cur_heading))
        new_heading = angle_to_left_right(normalize_angle(new_heading))
        action_description = f'Turn heading direction {turned_angle:.2f} degrees from {cur_heading} to {new_heading}.'
        return action_description, new_obs

    def make_action(self, action) -> str:
        '''Make single step action in MatterSim.'''

        if action.tool == 'make_action':
            # Extract viewpoint ID
            target = action.tool_input.strip(" ").strip('"').strip("'")
            # Make action
            turned_angle, new_obs = self.make_equiv_action([target])
            # Update history
            history = self.get_history(new_obs, turned_angle)
        else:
            # For non-valid actions
            history = action.tool_input
            turned_angle = None
            new_obs = self.env._get_obs()[0]

        self.history.append(history)

        detail = {
            "viewpointID": action.tool_input,
            "turned_angle": turned_angle,
            "obs_summary": new_obs["obs_summary"],
            "history": self.history[-1],
        }
        self.traj[0]['details'].append(detail)

        return [new_obs]

    def rollout(self, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        # Initialize the trajectory
        self.init_trajecotry(obs)

        # Load the instruction
        instructions = [ob['instruction'] for ob in obs]

        # Initilize
        self.traj[0]['instruction'] = instructions[0]

        # Rollout
        step = 0
        while step < self.config.max_iterations:

            # TODO: batchlization
            for i, ob in enumerate(obs):
                instruction = instructions[i]
                views = ob['obs']
                map = ob['map']
                # position = ob['position']
                heading = np.rad2deg(ob['heading'])
                elevation = np.rad2deg(ob['elevation'])

            # Construct input
            input = {
                "instruction": instruction,
                "views": views,
                "map": map,
                "heading": heading,
                "elevation": elevation,
                "history": self.history,
                "candidate": ob['candidate'],
            }

            # Model inference
            output = self.model(prompt = NAVGPT_2_PROMPT, input = input)
            self.traj[0]['llm_outputs'].append(output)

            # Parse LLMs output
            try:
                action = self.parse(output)
            except OutputParserException as e:

                observation = str(e.observation)
                text = str(e.llm_output)

                action = AgentAction("_Exception", observation, text)

            # If the tool chosen is the finishing tool, then we end and return.
            if isinstance(action, AgentFinish):
                print(f"\nStep {step}:\nLLM output: {action.log}\nAction: Finished!")
                break
            if isinstance(action, AgentAction):
                # Make action
                obs = self.make_action(action)
                print(f"\nStep {step}:\nLLM output: {action.log}\nAction: {self.history[-1]}")

            # Update history
            self.traj[0]['history'] = self.history
            step += 1

        return self.traj