# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Grader for the Professional Content Moderation Environment.
"""

try:
    from .server.moderation_env import ContentModerationEnv
    from .models import ModerationAction, ModerationDecision
except (ImportError, ValueError):
    from server.moderation_env import ContentModerationEnv # type: ignore
    from models import ModerationAction, ModerationDecision # type: ignore
import random

class GroundTruthAgent:
    """An agent that always makes the correct decision based on internal knowledge (cheating for grading testing)."""
    def act(self, observation):
        # This is just for environment verification. In real usage, the agent won't have access to ground truth.
        # Here we simulate a 'smart' agent by looking at the content and making a likely correct choice.
        text = observation.content_text.lower()
        if "buy" in text or "scam" in text or "click here" in text:
            decision = ModerationDecision.DELETE
            rationale = "Content contains commercial spam and suspicious links."
        elif "idiot" in text or "inferior" in text or "snake" in text:
            decision = ModerationDecision.DELETE
            rationale = "Content contains insults and targeted harassment violating community standards."
        elif "bleach" in text or "deepfake" in text:
            decision = ModerationDecision.DELETE
            rationale = "Content promotes dangerous medical misinformation or manipulated media."
        elif "giving up" in text:
            decision = ModerationDecision.ESCALATE
            rationale = "Content indicates potential self-harm. Escalating to safety team."
        else:
            decision = ModerationDecision.ALLOW
            rationale = "Content is within community guidelines and does not violate policies."
        
        return ModerationAction(decision=decision, rationale=rationale)

def grade(level: int, agent_class) -> dict:
    """
    Evaluates an agent on the specified environment level.
    Returns a score between 0.0 and 1.0.
    """
    env = ContentModerationEnv()
    obs = env.reset(level=level)
    
    agent = agent_class()
    total_reward = 0.0
    steps = 0
    max_steps = 10  # Content moderation batches are small
    
    while not obs.done and steps < max_steps:
        action = agent.act(obs)
        obs = env.step(action)
        total_reward += obs.reward
        steps += 1
        
    # Max possible reward per item is 15 (10 base + 5 rationale/keywords)
    # Average items per level is 5. Max potential reward ~75.
    # We normalize to 0.0 - 1.0. 
    # Minimum reward can be negative (e.g. -15 * 5 = -75).
    # Normalization: (total_reward - min_possible) / (max_possible - min_possible)
    
    max_potential = steps * 15.0
    min_potential = steps * -15.0
    
    if max_potential == min_potential:
        norm_score = 0.0
    else:
        norm_score = (total_reward - min_potential) / (max_potential - min_potential)
    
    return {
        "level": level,
        "reward": total_reward,
        "steps": steps,
        "score": round(norm_score, 3), # 0.0 - 1.0
        "final_score": obs.current_score
    }

if __name__ == "__main__":
    print("====================================================")
    print("   CONTENT MODERATION - GRADER PERFORMANCE          ")
    print("====================================================")
    
    for level in [1, 2, 3]:
        print(f"\n--- Evaluating Level {level} ---")
        res = grade(level, GroundTruthAgent)
        print(f"Items Moderated: {res['steps']}")
        print(f"Total Reward:    {res['reward']:.1f}")
        print(f"Normalized Score: {res['score']} (Range 0.0-1.0)")
        
        if res['score'] > 0.8:
            print("Status: PASS (Elite Performance)")
        elif res['score'] > 0.5:
            print("Status: PASS (Acceptable Performance)")
        else:
            print("Status: FAIL (Below Threshold)")
            
    print("\n====================================================")
