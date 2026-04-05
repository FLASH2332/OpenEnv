# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pydantic schemas for the Rag Optimizer environment.
These define the API contract between the client (agent) and the server.
"""

from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field


class RagOptimizerAction(BaseModel):
    """
    Actions the agent can take to interact with the Knowledge Base.
    """
    
    action_type: Literal["read_document", "update_document", "delete_document", "add_metadata", "submit"] = Field(
        ..., 
        description="The RAG optimization tool to execute."
    )
    
    doc_id: Optional[str] = Field(
        None, 
        description="The ID of the document to target."
    )
    text: Optional[str] = Field(
        None, 
        description="The text content (used for update_document)."
    )
    metadata_key: Optional[str] = Field(
        None, 
        description="The key of the metadata tag (used for add_metadata)."
    )
    metadata_value: Optional[str] = Field(
        None, 
        description="The value of the metadata tag (used for add_metadata)."
    )


class RagOptimizerObservation(BaseModel):
    """
    The environment's response to an action, including the state of the KB.
    """
    
    message: str = Field(
        ..., 
        description="Feedback from the last action (e.g., success/error messages)."
    )
    
    current_docs: Dict[str, Dict] = Field(
        ..., 
        description="A live summary of the documents currently inside the KB (doc_id -> metadata/length)."
    )
    
    # Required OpenEnv standard fields
    done: bool = Field(False, description="Whether the episode has finished.")
    reward: float = Field(0.0, description="The reward obtained from the last step.")
    metadata: Dict = Field(default_factory=dict, description="Additional optional information.")
