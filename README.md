---
title: Professional Content Moderation and Policy Enforcement
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# Professional Content Moderation & Policy Enforcement (PCMPE)

<div align="center">
  <p><strong>A high-fidelity OpenEnv reinforcement learning environment for automated content moderation.</strong></p>
</div>


Welcome to **PCMPE**, a real-world task simulation designed for the OpenEnv framework. This environment evaluates an AI agent's ability to act as a **Content Moderator**, enforcing platform policies across a variety of user-generated content scenarios, from simple spam filtering to nuanced hate speech detection and misinformation triage.

---

## 🖼️ Dashboard Preview
The environment includes a premium, glassmorphic **Moderator Console** (available at `/web`) for manual testing and real-time monitoring of agent decisions.

- **Real-time Queue**: Watch items flow through the moderation pipeline.
- **Policy Context**: Agents are provided with relevant policy snippets for every decision.
- **Rationale Analysis**: Rewards are based not just on the decision (ALLOW/DELETE/ESCALATE), but also on the quality of the reasoning.

---

## 🏆 Environment Design

### ⚡ The Moderation Protocol
Every action (`ModerationAction`) requires:
1.  **Decision**: `ALLOW`, `DELETE`, or `ESCALATE`.
2.  **Rationale**: A textual explanation justifying the choice based on the active policy.

### 🧪 Task Levels & Difficulty

| Level | Name | Description | Dataset Size |
|:------|:-----|:------------|:-------------|
| **1 (Easy)** | Filter & Spam | Obvious profanity and commercial spam. | 5 Items |
| **2 (Medium)** | Policy Enforcement | Nuanced hate speech vs political criticism. | 5 Items |
| **3 (Hard)** | Complex Triage | Misinformation, self-harm, and deepfakes. | 5 Items |

### 💰 Reward Function
- **Correct Decision**: `+10 points`
- **Incorrect Decision**: `-15 points` (High penalty for moderation failure)
- **High-Quality Rationale**: `+2 to +5 points` (Bonus for length and keyword matches)

---

## 🚀 Getting Started

### 1. Run the Environment Server
PCMPE is a standard FastAPI-based OpenEnv environment.
```bash
uv run server
```
Visit **`http://localhost:8000/web`** to access the interactive console.

### 2. Run the Programmatic Grader
Benchmark the environment logic and baseline performance:
```bash
python3 grader.py
```

### 3. Run the LLM Baseline (Submission Schema)
To produce a reproducible score using an LLM agent, run the submission-compliant `inference.py` script. This script uses the standard OpenAI SDK but is compatible with any OpenAI-compliant provider (like Google Gemini).

#### Configure Environment:
```bash
export API_BASE_URL="https://api.openai.com/v1"      # Or Gemini endpoint
export MODEL_NAME="gpt-4o-mini"                        # Or gemini-1.5-flash
export HF_TOKEN="your-api-key"                         # Your API Key (used as HF_TOKEN)
```

#### Execute Inference:
```bash
# This environment uses BrowserGym for high-fidelity evaluation.
# Ensure the server is running (uv run server) before executing inference.
# You will also need the 'browsergym' and 'playwright' packages.

export API_BASE_URL="https://router.huggingface.co/v1" 
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your-huggingface-token"

python3 inference.py
```
*Note: The script is designed to interact directly with the Web UI at `/web` to simulate human moderation workflow.*

---

## 📦 Containerization & Deployment
This environment is designed to be deployed as a **Hugging Face Space**.
- **Base OS**: Linux (Ubuntu 22.04)
- **Runtime**: Python 3.10+ (via `uv`)
- **Container**: `Dockerfile` included for standard build/run.

```bash
docker build -t moderation_env .
docker run -p 8000:8000 moderation_env
```

---

<div align="center">
  <i>Part of the Meta OpenEnv Competition &middot; Engineering the future of Agentic Safety</i> <br />
  <strong>Developed by Dwarkesh 🚀</strong>
</div>
