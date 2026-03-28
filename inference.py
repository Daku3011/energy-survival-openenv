# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Inference Script - Professional Content Moderation
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import re
import base64
import textwrap
import logging
from io import BytesIO
from typing import List, Optional, Dict

from openai import OpenAI
import numpy as np
from PIL import Image

# Import BrowserGym (Assuming it's available in the evaluation environment)
try:
    from browsergym_env import BrowserGymAction, BrowserGymEnv
except ImportError:
    # Fallback for local development if browsergym is not installed
    class BrowserGymAction:
        def __init__(self, action_str): self.action_str = action_str
    class BrowserGymEnv:
        @staticmethod
        def from_docker_image(**kwargs): return None

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
MAX_STEPS = 15
MAX_DOM_CHARS = 3500
TEMPERATURE = 0.2
MAX_TOKENS = 300
FALLBACK_ACTION = "noop()"

DEBUG = True
ACTION_PREFIX_RE = re.compile(
    r"^(action|next action)\s*[:\-]\s*",
    re.IGNORECASE,
)
ACTION_PATTERN = re.compile(r"[A-Za-z_]+\s*\(.*\)", re.DOTALL)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You control a web browser through BrowserGym to perform Professional Content Moderation.
    The goal is to review the content against the provided policy and take one of three actions:
    1. ALLOW: Click the 'btn-allow' button.
    2. DELETE: Click the 'btn-delete' button.
    3. ESCALATE: Click the 'btn-escalate' button.
    
    Before clicking a button, you MUST type a rationale into the textarea with ID 'rationale-input'.
    
    Reply with exactly one action string.
    The action must be a valid BrowserGym command such as:
    - noop()
    - click('element_id')
    - fill('selector', 'text to enter')
    - type('selector', 'text to enter')
    
    Example sequence:
    1. fill('rationale-input', 'This content violates the policy because...')
    2. click('btn-delete')
    
    Use single quotes around string arguments.
    Do not include explanations or additional text.
    """
).strip()

def build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-4:])

def extract_screenshot_uri(observation) -> Optional[str]:
    if not hasattr(observation, 'screenshot') or observation.screenshot is None:
        return None
    screen_array = np.array(observation.screenshot, dtype=np.uint8)
    image = Image.fromarray(screen_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    data_uri = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{data_uri}"

def extract_clickable_elements(observation) -> List[Dict[str, str]]:
    """Collect BrowserGym element IDs that can be clicked."""
    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}
    extra_props = obs_dict.get("extra_element_properties", {}) or {}

    clickables: List[Dict[str, str]] = []
    for bid, props in extra_props.items():
        if not props.get("clickable"):
            continue
        bbox = props.get("bbox") or []
        bbox_str = ", ".join(map(str, bbox)) if bbox else "?"
        clickables.append({"bid": str(bid), "bbox": bbox_str})
    
    clickables.sort(key=lambda item: item["bid"])
    return clickables

def build_user_prompt(step: int, observation, history: List[str]) -> str:
    goal = getattr(observation, 'goal', "Moderate the pending content queue.")
    url = getattr(observation, 'url', "http://localhost:7860/web")
    error_note = "Yes" if getattr(observation, 'last_action_error', None) else "No"

    clickables = extract_clickable_elements(observation)
    if clickables:
        actions_hint = "\n".join(f"    - {item['bid']} (bbox: {item['bbox']})" for item in clickables)
    else:
        actions_hint = "    (none detected - use IDs: rationale-input, btn-allow, btn-delete, btn-escalate)"

    prompt = textwrap.dedent(
        f"""
        Step: {step}
        Goal: {goal}
        Current URL: {url}
        Previous steps:
        {build_history_lines(history)}
        Last action error: {error_note}
        Available clickable element IDs: {actions_hint}
        
        Task Reference:
        - Textarea: rationale-input
        - Buttons: btn-allow, btn-delete, btn-escalate
        
        Reply with exactly one BrowserGym action string.
        """
    ).strip()
    return prompt

def parse_model_action(response_text: str) -> str:
    if not response_text:
        return FALLBACK_ACTION
    lines = response_text.splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line: continue
        line = ACTION_PREFIX_RE.sub("", line)
        match = ACTION_PATTERN.search(line)
        if match:
            action = match.group(0).strip()
            return re.sub(r"\s+", " ", action)
    match = ACTION_PATTERN.search(response_text)
    if match:
        action = match.group(0).strip()
        return re.sub(r"\s+", " ", action)
    return FALLBACK_ACTION

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Use the browsergym environment configured for this task
    # For submisson, organizers will provide the correctly configured image
    env = BrowserGymEnv.from_docker_image(
        image="browsergym-env:latest",
        env_vars={
            "BROWSERGYM_BENCHMARK": "custom",
            "BROWSERGYM_TASK_NAME": "professional-moderation",
            "TASK_URL": "http://localhost:7860/web"
        },
    )

    if env is None:
        print("Error: BrowserGymEnv not initialized. Ensure browsergym is installed.")
        return

    history: List[str] = []

    try:
        result = env.reset()
        observation = result.observation
        print(f"Episode goal: {getattr(observation, 'goal', 'Moderate Content')}")

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                print("Environment signalled done. Stopping early.")
                break

            user_prompt = build_user_prompt(step, observation, history)
            user_content = [{"type": "text", "text": user_prompt}]
            screenshot_uri = extract_screenshot_uri(observation)
            if screenshot_uri:
                user_content.append({"type": "image_url", "image_url": {"url": screenshot_uri}})

            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": user_content},
            ]

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"Model request failed ({exc}). Using fallback action.")
                response_text = FALLBACK_ACTION

            action_str = parse_model_action(response_text)
            print(f"Step {step}: model suggested -> {action_str}")

            result = env.step(BrowserGymAction(action_str=action_str))
            observation = result.observation

            reward = getattr(result, 'reward', 0.0) or 0.0
            error_flag = " ERROR" if getattr(observation, 'last_action_error', None) else ""
            history_line = f"Step {step}: {action_str} -> reward {reward:+.2f}{error_flag}"
            history.append(history_line)
            print(f"  Reward: {reward:+.2f} | Done: {result.done}")

            if result.done:
                print("Episode complete.")
                break
        else:
            print(f"Reached max steps ({MAX_STEPS}).")

    finally:
        env.close()

if __name__ == "__main__":
    main()
