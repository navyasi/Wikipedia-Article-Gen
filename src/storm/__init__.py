"""
STORM: Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking

A system for automatically generating Wikipedia-like articles from scratch using LLMs.
"""

from .core import StormSystem, StormConfig
from .models.llama_client import LlamaClient, LlamaConfig
from .search.web_search import WebSearchManager, SearchProvider, SearchResult

__version__ = "0.1.0"
__all__ = [
    "StormSystem",
    "StormConfig", 
    "LlamaClient",
    "LlamaConfig",
    "WebSearchManager",
    "SearchProvider",
    "SearchResult"
]