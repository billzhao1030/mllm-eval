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

    def parse(self, text: str):
        final_answer = FINAL_ANSWER_ACTION in text

        regex = (r"(?<=Answer:\s)(-?\d+(\.\d+)?)")

        action_match = re.search(regex, text)

        if action_match:
            action = "make_action"
            tool_input = action_match.group().strip()

            if self.config.experiment.action_space == 'viewpoint' and tool_input not in self.env.env.sims[0].navigable.keys():
                raise OutputParserException(
                    f"Could not make corresponding action: `{text}`",
                    observation=VIEWPOINT_NOT_IN_NAVIGABLE_ERROR_MESSAGE,
                    llm_output=text,
                    send_to_llm=True,
                )
            
            return AgentAction(action, tool_input, text)
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
        
    def get_history(self, obs, angle) -> str:
        pass

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
        self.history = [f'Navigation start, no actions taken yet.\nCurrent viewpoint "{obs[0]["viewpoint"]}": Scene from the viewpoint is a {obs[0]["obs_summary"]}']

        print(f"\nExcuating instruction:\n{obs[0]['instr_id']}: {obs[0]['instruction']}")

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
                image_obs = ob['img_obs']
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
                "candidate": ob['candidate']
            }

            # Model Inference
            output = self.model(prompt=PROMPT, input=input)
            self.traj[0]['llm_output'].append(output)

            # Parse LLMs output
            try:
                action = self.parse(output)
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
