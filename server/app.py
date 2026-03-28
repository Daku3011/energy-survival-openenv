# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the Professional Content Moderation Environment.
"""

import os
from fastapi import Request
from fastapi.responses import HTMLResponse

import sys
from pathlib import Path

# Add project root to sys.path to allow consistent imports
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from openenv.core.env_server.http_server import create_app  # type: ignore
except ImportError:
    pass

try:
    # Try absolute imports first (standard when running as package)
    from models import ModerationAction, ModerationObservation
    from server.moderation_env import ContentModerationEnv
except (ImportError, ValueError):
    try:
        # Fallback for local direct execution
        from moderation_env import ContentModerationEnv
        sys.path.append(str(Path(__file__).parent))
        from models import ModerationAction, ModerationObservation
    except (ImportError, ValueError):
        # Last resort: absolute from server
        try:
            from server.moderation_env import ContentModerationEnv
            from server.models import ModerationAction, ModerationObservation
        except ImportError:
            # Final fallback to package structure
            from treasure_env.models import ModerationAction, ModerationObservation
            from treasure_env.server.moderation_env import ContentModerationEnv

# Ensure Web Interface is enabled for Gradio if create_app uses it
os.environ["ENABLE_WEB_INTERFACE"] = "true"

app = create_app(
    ContentModerationEnv,
    ModerationAction,
    ModerationObservation,
    env_name="moderation_env",
    max_concurrent_envs=1,
)

# Stateful backend for the custom Web UI (separate from Gradio /web)
# This allows the custom premium UI to have its own session state
web_env = ContentModerationEnv()

@app.post("/web/reset")
async def web_reset(data: dict):
    try:
        level = int(data.get("level", 1))
    except (ValueError, TypeError):
        level = 1
        
    obs = web_env.reset(level=level)
    return {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
        "backend_v": "2.0"
    }

@app.post("/web/step")
async def web_step(data: dict):
    action_data = data.get("action", {})
    decision = action_data.get("decision", "ALLOW")
    rationale = action_data.get("rationale", "Automated moderation review.")
    
    action = ModerationAction(decision=decision, rationale=rationale)
    obs = web_env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done
    }

@app.get("/web/state")
async def web_state():
    # Return the current observation from the web_env
    is_done = web_env.current_index >= len(web_env.queue)
    obs = web_env._get_observation(reward=0.0, done=is_done)
    return {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": is_done
    }

# Override the default health check to satisfy custom grader requirements
# We do this by inserting it at the beginning of the routes list
@app.get("/health", tags=["Health"], include_in_schema=False)
async def custom_health():
    return {"status": "ok", "app": "pcmpe"}

# Helper to move the custom health check to the front so it takes precedence
for i, route in enumerate(app.routes):
    if hasattr(route, "path") and route.path == "/health" and i > 0:
        # If we find our custom health check later in the list, move it to index 0
        if getattr(route, "name", "") == "custom_health":
            app.routes.insert(0, app.routes.pop(i))
            break

@app.get("/", response_class=HTMLResponse)
def root_redirect():
    # Redirect or serve the premium UI
    return web_ui()

@app.get("/web", response_class=HTMLResponse)
def web_ui():
    # This is the premium custom UI
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=HTML_CONTENT)

HTML_CONTENT = """<!DOCTYPE html>
<html class="dark" lang="en"><head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>MODERATOR_AGENT - Professional Content Moderation</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<style>
    .glass-panel { background: rgba(22, 26, 33, 0.7); backdrop-filter: blur(16px); border: 1px solid rgba(153, 247, 255, 0.1); }
    body { background: radial-gradient(circle at 50% 50%, #161a21 0%, #0b0e14 100%); color: #ecedf6; font-family: 'Inter', sans-serif; }
    .policy-box { background: rgba(0, 241, 254, 0.05); border-left: 4px solid #00f1fe; }
    .animate-pulse-cyan { animation: pulse-cyan 2s infinite; }
</style>
</head>
<body class="min-h-screen p-8">
<main class="max-w-6xl mx-auto flex flex-col gap-8 h-[90vh]">
    <header class="flex justify-between items-end">
        <div>
            <span class="text-[10px] font-bold text-[#00f1fe] tracking-[0.3em] uppercase mb-1 block">Shield & Protocol</span>
            <h1 class="text-4xl font-bold tracking-tight font-['Space_Grotesk']">CONTENT_MODERATION_LENS</h1>
        </div>
        <div class="flex gap-4">
            <div class="glass-panel px-6 py-3 rounded-xl flex flex-col items-center">
                <span class="text-[10px] text-gray-400 uppercase tracking-widest">Score</span>
                <span id="stat-score" class="text-2xl font-bold text-[#00fd93]">0.0</span>
            </div>
            <div class="glass-panel px-6 py-3 rounded-xl flex flex-col items-center">
                <span class="text-[10px] text-gray-400 uppercase tracking-widest">Step</span>
                <span id="stat-steps" class="text-2xl font-bold">0</span>
            </div>
        </div>
    </header>

    <div class="grid grid-cols-12 gap-8 flex-1 overflow-hidden">
        <section class="col-span-4 flex flex-col gap-6 overflow-hidden">
            <div class="glass-panel rounded-2xl p-6 flex flex-col gap-4">
                <h2 class="text-xs font-bold uppercase tracking-widest text-[#00f1fe]">Protocol Guidelines</h2>
                <div id="policy-text" class="policy-box p-4 rounded-lg text-sm italic h-48 overflow-y-auto text-gray-300">
                    Awaiting initialization...
                </div>
            </div>
            <div class="glass-panel rounded-2xl p-6 flex-1 flex flex-col gap-4 overflow-hidden">
                <h2 class="text-xs font-bold uppercase tracking-widest text-gray-400 flex justify-between">
                    <span>Queue Status</span>
                    <span id="queue-count" class="text-[#00f1fe]">0 Items</span>
                </h2>
                <div id="queue-visual" class="flex-1 space-y-2 overflow-y-auto pr-2"></div>
            </div>
        </section>

        <section class="col-span-8 flex flex-col gap-6 overflow-hidden">
            <div class="glass-panel rounded-2xl p-8 flex-1 flex flex-col gap-8 relative overflow-hidden">
                <div id="end-screen" class="absolute inset-0 z-10 bg-[#0b0e14]/90 flex flex-col items-center justify-center gap-4 hidden">
                    <h2 class="text-2xl font-bold">Batch Complete</h2>
                    <button onclick="startEpisode(currentLevel)" class="px-8 py-3 bg-[#00f1fe] text-black font-bold rounded-full">Initialize New Batch</button>
                </div>

                <div class="flex justify-between items-start">
                    <div>
                        <span id="content-id" class="text-[10px] font-mono text-gray-500">ID: P-000000</span>
                        <h3 id="author-name" class="font-bold text-sm">Author Unknown</h3>
                    </div>
                    <div class="flex gap-2">
                        <button onclick="startEpisode(1)" id="lvl-1" class="px-3 py-1 bg-gray-800 rounded text-xs font-bold border border-transparent">Easy</button>
                        <button onclick="startEpisode(2)" id="lvl-2" class="px-3 py-1 bg-gray-800 rounded text-xs font-bold border border-transparent">Medium</button>
                        <button onclick="startEpisode(3)" id="lvl-3" class="px-3 py-1 bg-gray-800 rounded text-xs font-bold border border-transparent">Hard</button>
                    </div>
                </div>

                <div class="flex-1 bg-white/[0.02] border border-white/[0.05] rounded-xl p-8 flex items-center justify-center">
                    <p id="content-text" class="text-2xl font-medium text-center italic">"Your content will appear here..."</p>
                </div>

                <div class="flex flex-col gap-4">
                    <textarea id="rationale-input" class="w-full bg-black/40 border border-white/10 rounded-xl p-4 text-sm h-24" placeholder="Explain your decision..."></textarea>
                    <div class="grid grid-cols-3 gap-4">
                        <button onclick="takeStep('ALLOW')" class="p-4 bg-[#005b31]/20 border border-[#005b31]/40 rounded-xl text-[#00fd93] font-bold uppercase text-xs">Allow</button>
                        <button onclick="takeStep('DELETE')" class="p-4 bg-[#9f0519]/20 border border-[#9f0519]/40 rounded-xl text-[#ff716c] font-bold uppercase text-xs">Delete</button>
                        <button onclick="takeStep('ESCALATE')" class="p-4 bg-[#390050]/20 border border-[#390050]/40 rounded-xl text-[#d674ff] font-bold uppercase text-xs">Escalate</button>
                    </div>
                </div>
            </div>
        </section>
    </div>
</main>

<script>
    let currentLevel = 1;
    let isProcessing = false;

    async function startEpisode(level) {
        currentLevel = level;
        isProcessing = true;
        document.getElementById('end-screen').classList.add('hidden');
        
        [1,2,3].forEach(i => {
            const btn = document.getElementById(`lvl-${i}`);
            btn.style.borderColor = i === level ? '#00f1fe' : 'transparent';
        });

        const res = await fetch('/web/reset', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({level})
        });
        const data = await res.json();
        updateUI(data.observation);
        isProcessing = false;
    }

    async function takeStep(decision) {
        if(isProcessing) return;
        const rationale = document.getElementById('rationale-input').value;
        if(!rationale) { alert('Rationale required.'); return; }

        isProcessing = true;
        const res = await fetch('/web/step', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ action: { decision, rationale } })
        });
        const data = await res.json();
        document.getElementById('rationale-input').value = '';
        updateUI(data.observation);
        if(data.done) document.getElementById('end-screen').classList.remove('hidden');
        isProcessing = false;
    }

    function updateUI(obs) {
        document.getElementById('content-text').innerText = `"${obs.content_text}"`;
        document.getElementById('policy-text').innerText = obs.policy_guidelines;
        document.getElementById('stat-score').innerText = obs.current_score.toFixed(1);
        document.getElementById('stat-steps').innerText = obs.metadata?.steps || 0;
        document.getElementById('queue-count').innerText = `${obs.queue_remaining} Items`;
        document.getElementById('content-id').innerText = `ID: ${obs.content_id}`;
    }

    startEpisode(1);
</script>
</body></html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
