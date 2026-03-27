# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Energy Survival Grid Environment.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import Field
from openenv.core.env_server.types import Action, Observation  # type: ignore


class ModerationDecision(str, Enum):
    """Available moderation decisions."""
    ALLOW = "ALLOW"
    DELETE = "DELETE"
    ESCALATE = "ESCALATE"


class ModerationAction(Action):
    """Action for the Content Moderation environment."""

    decision: ModerationDecision = Field(..., description="The moderation decision for the content.")
    rationale: str = Field(..., description="Detailed explanation for the decision based on policy.")


class ModerationObservation(Observation):
    """Observation from the Content Moderation environment."""

    content_id: str = Field(..., description="Unique ID for the piece of content.")
    content_text: str = Field(..., description="The actual text or description of the content to moderate.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context (e.g., author history, timestamp).")
    policy_guidelines: str = Field(..., description="Relevant policy snippet for the current task.")
    queue_remaining: int = Field(..., description="Number of items remaining in the moderation queue.")
    current_score: float = Field(default=0.0, description="Cumulative score in the current session.")

