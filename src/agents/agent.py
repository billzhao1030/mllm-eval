"""Agent that interacts with Matterport3D simulator via a hierarchical planning approach."""

import re
from agent_base import BaseAgent
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

        self.model = get_models(config.experiment.model)

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

