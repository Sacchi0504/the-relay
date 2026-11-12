"""Abstract Model Provider Interface."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
from app.core.models import ModelRequest, ModelResponse, ProviderSource


class BaseModelProvider(ABC):
    """Abstract interface for all LLM providers (Real Groq and Simulator)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @property
    @abstractmethod
    def source(self) -> ProviderSource:
        """Provider source: REAL or STAND_IN."""
        pass

    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Execute non-streaming inference request."""
        pass

    @abstractmethod
    async def generate_stream(
        self, request: ModelRequest
    ) -> AsyncGenerator[str, None]:
        """Execute streaming inference request, yielding chunks."""
        pass

