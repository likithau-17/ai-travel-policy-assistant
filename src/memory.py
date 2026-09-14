"""
Conversation memory helpers.

LangGraph's MemorySaver stores conversation state by thread_id.
These helpers keep memory-related logic separate from the Flask layer.
"""

from langgraph.checkpoint.memory import MemorySaver


def create_memory():
    """Create the in-memory checkpointer used by the agent."""
    return MemorySaver()
