# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Professional Content Moderation & Policy Enforcement Environment.
"""

import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ModerationAction, ModerationDecision, ModerationObservation
except (ImportError, ValueError):
    from models import ModerationAction, ModerationDecision, ModerationObservation # type: ignore

class ContentModerationEnv(Environment):
    """
    An environment for training and evaluating content moderation agents.
    Agents must review content items and decide whether to ALLOW, DELETE, or ESCALATE.
    """

    def __init__(self):
        super().__init__()
        self.level = 1
        self.queue: List[Dict[str, Any]] = []
        self.policy = ""
        self.current_index = 0
        self.total_reward = 0.0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.reset(level=1)

    def reset(self, level: Optional[int] = None, task_id: Optional[str] = None, seed: Optional[int] = None, **kwargs) -> ModerationObservation:
        """
        Resets the environment. Supports both 'level' (numeric) and 'task_id' (string).
        """
        if seed is not None:
            random.seed(seed)
            
        # Prioritize task_id from ResetRequest if provided
        if task_id:
            task_map = {
                "easy": 1, "level_1": 1, "level1": 1,
                "medium": 2, "level_2": 2, "level2": 2,
                "hard": 3, "level_3": 3, "level3": 3
            }
            self.level = task_map.get(task_id.lower(), 1)
        elif level is not None:
            self.level = level
        else:
            self.level = 1

        self.queue, self.policy = self._load_data(self.level)
        self.current_index = 0
        self.total_reward = 0.0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        return self._get_observation(reward=0.0, done=False)

    def step(self, action: ModerationAction) -> ModerationObservation:
        if self.current_index >= len(self.queue):
            return self._get_observation(reward=0.0, done=True)

        current_item = self.queue[self.current_index]
        reward = self._calculate_reward(action, current_item)
        
        self.total_reward += reward
        self.current_index += 1
        self._state.step_count = self.current_index
        
        done = self.current_index >= len(self.queue)
        return self._get_observation(reward=reward, done=done)

    @property
    def state(self) -> State:
        """Returns the current internal state of the environment."""
        is_done = self.current_index >= len(self.queue)
        # Update custom data for the state object
        self._state.is_done = is_done
        self._state.custom_data = {
            "level": self.level,
            "score": self.total_reward,
            "queue_remaining": len(self.queue) - self.current_index,
            "current_index": self.current_index
        }
        return self._state

    def _calculate_reward(self, action: ModerationAction, item: Dict[str, Any]) -> float:
        """Calculates reward based on alignment with ground truth and rationale quality."""
        # 1. Decision Match (Base Reward: 10)
        if action.decision == item["ground_truth"]:
            reward = 10.0
        else:
            # Major penalty for wrong decision
            reward = -10.0
            
        # 2. Rationale Quality (Bonus: up to 5)
        # Check for presence of key technical terms (simulating high-quality reasoning)
        rationale_lower = action.rationale.lower()
        keyword_hits = sum(1 for kw in item.get("keywords", []) if kw.lower() in rationale_lower)
        reward += min(keyword_hits, 5.0)
        
        return reward

    def _get_observation(self, reward: float, done: bool) -> ModerationObservation:
        """Constructs the observation object for the current state."""
        if self.current_index < len(self.queue):
            item = self.queue[self.current_index]
            obs = ModerationObservation(
                content_id=item["id"],
                content_text=item["text"],
                policy_guidelines=self.policy,
                queue_remaining=len(self.queue) - self.current_index,
                current_score=self.total_reward,
                reward=reward,
                done=done,
                metadata={
                    "author": item.get("metadata", {}).get("author", "Unknown"),
                    "steps": self.current_index
                }
            )
        else:
            # Tail observation when queue is empty
            obs = ModerationObservation(
                content_id="EOF",
                content_text="No more content in queue.",
                policy_guidelines=self.policy,
                queue_remaining=0,
                current_score=self.total_reward,
                reward=reward,
                done=done,
                metadata={"steps": self.current_index}
            )
        return obs

    def _load_data(self, level: int) -> Tuple[List[Dict[str, Any]], str]:
        if level <= 1:
            policy = "POLICY: Remove any content containing explicit profanity, insults, or commercial spam (links to shady sites)."
            data = [
                {"id": "p1", "text": "Buy cheap watches at scam-site.com! Best prices!", "ground_truth": ModerationDecision.DELETE, "keywords": ["spam", "commercial", "link", "scam"], "metadata": {"author": "bot123"}},
                {"id": "p2", "text": "I really love the new design of this app, great job!", "ground_truth": ModerationDecision.ALLOW, "keywords": ["positive", "feedback"], "metadata": {"author": "user_45"}},
                {"id": "p3", "text": "You are a total idiot and nobody likes your posts.", "ground_truth": ModerationDecision.DELETE, "keywords": ["insult", "harassment"], "metadata": {"author": "troll_x"}},
                {"id": "p4", "text": "Can someone help me with the login issue? I keep getting a 404.", "ground_truth": ModerationDecision.ALLOW, "keywords": ["support", "question"], "metadata": {"author": "newbie_01"}},
                {"id": "p5", "text": "Get rich quick! Click here ==> bit.ly/fake-money", "ground_truth": ModerationDecision.DELETE, "keywords": ["spam", "link"], "metadata": {"author": "spam_master"}},
            ]
        elif level == 2:
            policy = "POLICY: Prohibit hate speech (attacks based on race, religion, gender) and targeted harassment. Allow political criticism if not hateful."
            data = [
                {"id": "p1", "text": "I think the current government's economic policy is a disaster.", "ground_truth": ModerationDecision.ALLOW, "keywords": ["criticism", "political"], "metadata": {"author": "citizen_a"}},
                {"id": "p2", "text": "All people from [Country X] are naturally inferior and should be banned.", "ground_truth": ModerationDecision.DELETE, "keywords": ["hate speech", "discrimination", "inferior"], "metadata": {"author": "bigot_88"}},
                {"id": "p3", "text": "This specific journalist is a snake and we should haunt them at their home.", "ground_truth": ModerationDecision.DELETE, "keywords": ["harassment", "threat", "journalist"], "metadata": {"author": "angry_mob"}},
                {"id": "p4", "text": "The latest move by the Senator is highly questionable but legal.", "ground_truth": ModerationDecision.ALLOW, "keywords": ["opinion", "public figure"], "metadata": {"author": "pundit_99"}},
                {"id": "p5", "text": "Followers of [Religion Y] have no place in a civilized society.", "ground_truth": ModerationDecision.DELETE, "keywords": ["hate speech", "religion", "civilized"], "metadata": {"author": "extremist_1"}},
            ]
        else:
            policy = "POLICY: Moderate misinformation regarding public health, self-harm promotion, and deepfake/manipulated media. Escalate complex cases to human supervisors."
            data = [
                {"id": "p1", "text": "Drinking bleach is a proven cure for the common cold. Pass it on!", "ground_truth": ModerationDecision.DELETE, "keywords": ["medical misinformation", "health", "dangerous"], "metadata": {"author": "doc_fake"}},
                {"id": "p2", "text": "I'm feeling really hopless and considering giving up on everything. Life isn't worth it.", "ground_truth": ModerationDecision.ESCALATE, "keywords": ["self-harm", "help", "distress"], "metadata": {"author": "lonely_soul"}},
                {"id": "p3", "text": "Check out this video of the President saying things he never actually said! [attached: deepfake.mp4]", "ground_truth": ModerationDecision.DELETE, "keywords": ["manipulated media", "deepfake", "misinformation"], "metadata": {"author": "chaos_bot"}},
                {"id": "p4", "text": "The moon is actually made of cheese and the government is hiding it.", "ground_truth": ModerationDecision.ALLOW, "keywords": ["conspiracy", "harmless"], "metadata": {"author": "tinfoil_hat"}},
                {"id": "p5", "text": "Instructions on how to bypass security at the local hospital to steal supplies.", "ground_truth": ModerationDecision.DELETE, "keywords": ["illegal", "safety", "crime"], "metadata": {"author": "heist_guy"}},
            ]
        
        # Add some variation by shuffling
        random.shuffle(data)
        return data, policy
