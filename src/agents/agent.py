"""Agent that interacts with Matterport3D simulator via a hierarchical planning approach."""
from agent_base import BaseAgent

class NavAgent(BaseAgent):
    def __init__(self, env, config, logger):
        super().__init__(env)
        
        self.config = config
        self.logger = logger

        
