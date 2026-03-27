# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Professional Content Moderation - Baseline Inference Script.
Uses OpenAI API to moderate content based on environmental observations.
"""

import os
import json
from typing import Dict, Any, List
from openai import OpenAI
from .server.moderation_env import ContentModerationEnv
from .models import ModerationAction, ModerationObservation, ModerationDecision

# Initialize OpenAI Client (reads OPENAI_API_KEY from env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "mock_key"))

def get_llm_decision(obs: ModerationObservation) -> ModerationAction:
    """Queries the LLM to make a moderation decision."""
    
    prompt = f"""
You are a Professional Content Moderator. Review the following content based on the provided policy.

POLICY:
{obs.policy_guidelines}

CONTENT TO REVIEW (ID: {obs.content_id}):
"{obs.content_text}"

METADATA:
{json.dumps(obs.metadata, indent=2)}

DIRECTIONS:
1. Choose one decision: ALLOW, DELETE, or ESCALATE.
2. Provide a brief rationale based on the policy.
3. Respond ONLY in valid JSON format:
{{
  "decision": "ALLOW" | "DELETE" | "ESCALATE",
  "rationale": "your explanation here"
}}
"""

    if os.getenv("OPENAI_API_KEY") is None:
        # Fallback for environment without API key (reproducible mock logic)
        text = obs.content_text.lower()
        if "buy" in text or "scam" in text or "idiot" in text:
            return ModerationAction(decision=ModerationDecision.DELETE, rationale="Filtered by baseline heuristic (No API key found).")
        return ModerationAction(decision=ModerationDecision.ALLOW, rationale="Allowed by baseline heuristic (No API key found).")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful and precise content moderator."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        res_data = json.loads(response.choices[0].message.content)
        return ModerationAction(
            decision=ModerationDecision(res_data["decision"]),
            rationale=res_data["rationale"]
        )
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return ModerationAction(decision=ModerationDecision.ALLOW, rationale="Error fallback.")

def run_evaluation(level: int):
    """Runs the full evaluation for a specific level."""
    env = ContentModerationEnv()
    obs = env.reset(level=level)
    
    print(f"\n--- Starting Level {level} Evaluation ---")
    total_reward = 0.0
    
    while not obs.done:
        print(f"Reviewing: {obs.content_text[:50]}...")
        action = get_llm_decision(obs)
        print(f"Decision: {action.decision} | Rationale: {action.rationale[:60]}...")
        
        obs = env.step(action)
        total_reward += obs.reward
        print(f"Step Reward: {obs.reward}")

    print(f"Level {level} Complete. Total Reward: {total_reward}")
    print(f"Final Session Score: {obs.current_score}")
    return obs.current_score

if __name__ == "__main__":
    print("====================================================")
    print("   CONTENT MODERATION - BASELINE INFERENCE          ")
    print("====================================================")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. Running with heuristic mock.")
    
    scores = []
    for level in [1, 2, 3]:
        score = run_evaluation(level)
        scores.append(score)
    
    print("\n====================================================")
    print(f"Overall Baseline Scores: L1={scores[0]}, L2={scores[1]}, L3={scores[2]}")
    print("====================================================")
