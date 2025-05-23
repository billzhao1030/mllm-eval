"""Agent that interacts with Matterport3D simulator via a hierarchical planning approach."""

import re
import numpy as np
from agent_base import BaseAgent
from models.base_mllm import BaseMLLM
from tasks.mp3d_dataset import MP3DDataset
from prompt import PROMPT
from models import get_models
from typing import List

from utils.data import AgentAction, AgentFinish, OutputParserException

FINAL_ANSWER_ACTION = "Final Answer:"

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
    def __init__(self, env, config, logger):
        super().__init__(env)
        
        self.config = config
        self.logger = logger

        self.model: BaseMLLM = get_models(config.experiment.model)

        self.history: List[str] = None

    def parse(self, text: str, id_viewpoint: dict):
        final_answer = FINAL_ANSWER_ACTION in text

        regex = (r"(?<=Answer:\s)(-?\d+(\.\d+)?)")

        action_match = re.search(regex, text)

        if action_match:
            action = "make_action"
            tool_input = id_viewpoint[action_match.group().strip()]

            if tool_input not in self.env.env.sims[0].navigable.keys():
                raise OutputParserException(
                    f"Could not make corresponding action: `{text}`",
                    observation=VIEWPOINT_NOT_IN_NAVIGABLE_ERROR_MESSAGE,
                    llm_output=text,
                    send_to_llm=True,
                )
            
            return AgentAction(action, action_match.group().strip(), text)
        elif final_answer:
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
        

    def init_trajectory(self, obs: List[dict]):
        """Initialize the trajectory with the given observation."""
        # Record the navigation path
        self.traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'llm_outputs': [],
            'details': [],
        } for ob in obs]
        # Record the history of actions taken
        self.history = ["Navigation starts."]

        print(f"\nExcuating instruction:\n{obs[0]['instr_id']}: {obs[0]['instruction']}")


    def make_equiv_action(self, actions: List[dict]) -> str:
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

        # Get the target caption
        target_caption = cur_obs['caption'][actions[0]]

        # Extract viewpoint ID
        tool_input = cur_obs['id_viewpoint'][actions[0]]
        target = [tool_input.strip(" ").strip('"').strip("'")]

        # Get the distance from current viewpoint to target
        distance = cur_obs['candidate'][target[0]]['distance']

        # Make the action
        new_obs = self.env.step(target)[0]
        new_heading = np.rad2deg(new_obs['heading'])

        # Record the trajectory
        self.traj[0]['path'].append([target[0]])

        # Calculate the turned angle
        turned_angle = new_heading - cur_heading

        # Generate action description
        cur_heading = angle_to_left_right(normalize_angle(cur_heading))
        new_heading = angle_to_left_right(normalize_angle(new_heading))

        action_description = f'Turn heading direction {turned_angle:.2f} degrees from {cur_heading} to {new_heading}, and forward {distance:.2f} meters towards {target_caption}.'

        return action_description, new_obs

    def make_action(self, action: AgentAction) -> str:
        """Make single action in Simulator"""
        if action.tool == 'make_action':
            # Make action
            action_description, new_obs = self.make_equiv_action([action.tool_input])

            # Update history
            history = action_description
        else:
            # For non-valid actions
            history = action.tool_input # Error observation message
            turned_angle = None
            new_obs = self.env._get_obs()[0]

        self.history.append(history)

        detail = {
            "viewpointID": action.tool_input,
            "turned_angle": turned_angle,
            "summary": new_obs["summary"], # TODO, check this if needed
            "history": self.history[-1],
        }
        self.traj[0]['details'].append(detail)

        return [new_obs]


    def rollout(self, reset=True):
        if reset: # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        # Initialize the trajectory
        self.init_trajectory(obs)

        # Load the instruction
        instructions = [ob['instruction'] for ob in obs]

        self.traj[0]['instruction'] = instructions[0]

        # Rollout
        step = 0
        while step < self.config.experiment.max_step:
            for i, ob in enumerate(obs):
                instruction = instructions[i]
                image_obs = ob['img_obs']['egocentirc'] # Single egocentric plot
                marker_caption = ob['caption']

                map = ob['map']

                heading = np.rad2deg(ob['heading'])
                elevation = np.rad2deg(ob['elevation'])

            input = {
                "instruction": instruction,
                "caption": marker_caption,
                "map": map,
                "heading": heading,
                "elevation": elevation,
                "history": self.history,
                "candidate": ob['candidate'],
                'image': image_obs
            }

            # Model Inference
            output = self.model(prompt=PROMPT, input=input)
            self.traj[0]['llm_output'].append(output)

            # Parse LLMs output
            try:
                action = self.parse(output, ob['id_viewpoint'])
            except OutputParserException as e:
                observation = str(e.observation)
                text = str(e.llm_output)

                action = AgentAction("_Exception", observation, text)

            # If the tool chosen is the finishing tool, then we end and return.
            if isinstance(action, AgentFinish):
                self.logger.info(f"\nStep {step}:\nLLM output: {action.log}\nAction: Finished!")

            if isinstance(action, AgentAction):
                # Make action
                obs = self.make_action(action)
                print(f"\nStep {step}:\nLLM output: {action.log}\nAction: {self.history[-1]}")

            # Update history
            self.traj[0]['history'] = self.history
            step += 1

        return self.traj
