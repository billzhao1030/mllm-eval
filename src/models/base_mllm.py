from abc import ABC, abstractmethod

class BaseMLLM(ABC):
    @abstractmethod
    def __init__(self, config, **kwargs):
        """Load model, set device, build preprocessors."""

    @abstractmethod
    def __call__(self, prompt: str, input: dict) -> str:
        """Run inference, return LLM text output."""
        
    @property
    @abstractmethod
    def _identifying_params(self) -> dict:
        """Return dict of key hyperparameters."""