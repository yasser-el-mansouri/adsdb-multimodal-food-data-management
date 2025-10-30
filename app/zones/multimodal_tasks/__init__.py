"""
Multimodal Tasks Zone

This zone handles multimodal tasks such as retrieval, querying, and analysis
on both text and image data.
"""

from app.zones.multimodal_tasks.task1_retrieval import Task1RetrievalProcessor
from app.zones.multimodal_tasks.task2 import ExploitationMultiModalSearcher
from app.zones.multimodal_tasks.task3_rag import Task3RAGProcessor

__all__ = [
    "Task1RetrievalProcessor",
    "ExploitationMultiModalSearcher",
    "Task3RAGProcessor",
]

