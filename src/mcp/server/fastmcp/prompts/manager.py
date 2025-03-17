"""Prompt management functionality."""

from typing import Any, Union, Dict, List

from mcp.server.fastmcp.prompts.base import Message, Prompt
from mcp.server.fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)


class PromptManager:
    """Manages FastMCP prompts."""

    def __init__(self, warn_on_duplicate_prompts: bool = True):
        self._prompts: Dict[str, Prompt] = {}
        self.warn_on_duplicate_prompts = warn_on_duplicate_prompts

    def get_prompt(self, name: str) -> Union[Prompt, None]:
        """Get prompt by name."""
        return self._prompts.get(name)

    def list_prompts(self) -> List[Prompt]:
        """List all registered prompts."""
        return list(self._prompts.values())

    def add_prompt(
        self,
        prompt: Prompt,
    ) -> Prompt:
        """Add a prompt to the manager."""

        # Check for duplicates
        existing = self._prompts.get(prompt.name)
        if existing:
            if self.warn_on_duplicate_prompts:
                logger.warning(f"Prompt already exists: {prompt.name}")
            return existing

        self._prompts[prompt.name] = prompt
        return prompt

    async def render_prompt(
        self, name: str, arguments: Union[Dict[str, Any], None] = None
    ) -> List[Message]:
        """Render a prompt by name with arguments."""
        prompt = self.get_prompt(name)
        if not prompt:
            raise ValueError(f"Unknown prompt: {name}")

        return await prompt.render(arguments)