from dataclasses import dataclass
from typing import Union


@dataclass
class ReadResourceContents:
    """Contents returned from a read_resource call."""

    content: Union[str, bytes]
    mime_type: Union[str, None] = None
