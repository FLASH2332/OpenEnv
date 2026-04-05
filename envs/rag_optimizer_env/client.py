# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rag Optimizer Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import RagOptimizerAction, RagOptimizerObservation


class RagOptimizerEnvClient(EnvClient[RagOptimizerAction, RagOptimizerObservation, State]):
    """
    Client for the Rag Optimizer Environment.
    Translates local Pydantic objects to JSON for the OpenEnv WebSocket.
    """

    def _step_payload(self, action: RagOptimizerAction) -> Dict:
        """Convert RagOptimizerAction to JSON payload."""
        return {
            "action_type": action.action_type,
            "doc_id": action.doc_id,
            "text": action.text,
            "metadata_key": action.metadata_key,
            "metadata_value": action.metadata_value,
        }

    def _parse_result(self, payload: Dict) -> StepResult[RagOptimizerObservation]:
        """Parse server response back into RagOptimizerObservation."""
        obs_data = payload.get("observation", {})
        observation = RagOptimizerObservation(
            message=obs_data.get("message", ""),
            current_docs=obs_data.get("current_docs", {}),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse the hidden tracking state."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
