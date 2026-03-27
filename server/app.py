# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Energy Survival Grid Environment.
"""

try:
    from openenv.core.env_server.http_server import create_app  # type: ignore
except ImportError:
    pass

try:
    from ..models import ModerationAction, ModerationObservation
    from .moderation_env import ContentModerationEnv
except (ImportError, ValueError):
    from models import ModerationAction, ModerationObservation # type: ignore
    from server.moderation_env import ContentModerationEnv

app = create_app(
    ContentModerationEnv,
    ModerationAction,
    ModerationObservation,
    env_name="moderation_env",
    max_concurrent_envs=1,
)

# Stateful backend for the Web UI
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

from fastapi.responses import HTMLResponse  # type: ignore # noqa: E402

@app.get("/", response_class=HTMLResponse)
def root():
    return web_ui()

@app.get("/web", response_class=HTMLResponse)
def web_ui():
    return HTMLResponse("""<!DOCTYPE html>
<html class="dark" lang="en"><head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>MODERATOR_AGENT - Professional Content Moderation</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<style>
    .material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; }
    .glass-panel { background: rgba(22, 26, 33, 0.7); backdrop-filter: blur(16px); border: 1px solid rgba(153, 247, 255, 0.1); }
    body { background: radial-gradient(circle at 50% 50%, #161a21 0%, #0b0e14 100%); color: #ecedf6; font-family: 'Inter', sans-serif; }
    .policy-box { background: rgba(0, 241, 254, 0.05); border-left: 4px solid #00f1fe; }
    .content-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); }
    .animate-pulse-cyan { animation: pulse-cyan 2s infinite; }
    @keyframes pulse-cyan { 0% { box-shadow: 0 0 0 0 rgba(0, 241, 254, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(0, 241, 254, 0); } 100% { box-shadow: 0 0 0 0 rgba(0, 241, 254, 0); } }
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
        <!-- Left: Policy & Queue -->
        <section class="col-span-4 flex flex-col gap-6 overflow-hidden">
            <div class="glass-panel rounded-2xl p-6 flex flex-col gap-4">
                <h2 class="text-xs font-bold uppercase tracking-widest text-[#00f1fe] flex items-center gap-2">
                    <span class="material-symbols-outlined text-sm">gavel</span> Protocol Guidelines
                </h2>
                <div id="policy-text" class="policy-box p-4 rounded-lg text-sm leading-relaxed text-gray-300 italic h-48 overflow-y-auto">
                    Awaiting initialization...
                </div>
            </div>

            <div class="glass-panel rounded-2xl p-6 flex-1 flex flex-col gap-4 overflow-hidden">
                <h2 class="text-xs font-bold uppercase tracking-widest text-gray-400 flex items-center justify-between">
                    <span>Queue Status</span>
                    <span id="queue-count" class="text-[#00f1fe]">0 Items</span>
                </h2>
                <div id="queue-visual" class="flex-1 space-y-2 overflow-y-auto pr-2">
                    <!-- Queue items visual -->
                </div>
            </div>
        </section>

        <!-- Right: Moderation Interface -->
        <section class="col-span-8 flex flex-col gap-6 overflow-hidden">
            <div class="glass-panel rounded-2xl p-8 flex-1 flex flex-col gap-8 relative overflow-hidden">
                <div id="end-screen" class="absolute inset-0 z-10 bg-[#0b0e14]/90 flex flex-col items-center justify-center gap-4 hidden">
                    <span class="material-symbols-outlined text-6xl text-[#00fd93]">verified</span>
                    <h2 class="text-2xl font-bold">Batch Complete</h2>
                    <button onclick="startEpisode(currentLevel)" class="px-8 py-3 bg-[#00f1fe] text-black font-bold rounded-full hover:scale-105 transition-transform">Initialize New Batch</button>
                </div>

                <div class="flex justify-between items-start">
                    <div class="flex flex-col gap-1">
                        <span id="content-id" class="text-[10px] font-mono text-gray-500">ID: P-000000</span>
                        <div class="flex items-center gap-3">
                            <div class="w-10 h-10 rounded-full bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center">
                                <span class="material-symbols-outlined text-xl">person</span>
                            </div>
                            <div>
                                <h3 id="author-name" class="font-bold text-sm">Author Unknown</h3>
                                <span class="text-[10px] text-gray-500">Post Timestamp: Just Now</span>
                            </div>
                        </div>
                    </div>
                    <div class="flex gap-2">
                        <button onclick="startEpisode(1)" id="lvl-1" class="px-3 py-1 bg-gray-800 rounded text-xs font-bold border border-transparent hover:border-[#00f1fe] transition-all">Easy</button>
                        <button onclick="startEpisode(2)" id="lvl-2" class="px-3 py-1 bg-gray-800 rounded text-xs font-bold border border-transparent hover:border-[#00f1fe] transition-all">Medium</button>
                        <button onclick="startEpisode(3)" id="lvl-3" class="px-3 py-1 bg-gray-800 rounded text-xs font-bold border border-transparent hover:border-[#00f1fe] transition-all">Hard</button>
                    </div>
                </div>

                <div class="flex-1 bg-white/[0.02] border border-white/[0.05] rounded-xl p-8 flex items-center justify-center">
                    <p id="content-text" class="text-2xl font-medium tracking-tight text-center max-w-lg italic">
                        "Your content will appear here for review..."
                    </p>
                </div>

                <div class="flex flex-col gap-4">
                    <label class="text-[10px] font-bold uppercase tracking-widest text-gray-500">Review Rationale</label>
                    <textarea id="rationale-input" class="w-full bg-black/40 border border-white/10 rounded-xl p-4 text-sm focus:border-[#00f1fe] focus:ring-0 transition-all h-24" placeholder="Explain your decision based on the protocol guidelines..."></textarea>
                    
                    <div class="grid grid-cols-3 gap-4">
                        <button onclick="takeStep('ALLOW')" class="group flex flex-col items-center gap-2 p-4 bg-[#005b31]/20 border border-[#005b31]/40 rounded-xl hover:bg-[#005b31]/40 transition-all">
                            <span class="material-symbols-outlined text-[#00fd93] group-hover:scale-110 transition-transform">check_circle</span>
                            <span class="text-xs font-bold uppercase tracking-wider text-[#00fd93]">Allow</span>
                        </button>
                        <button onclick="takeStep('DELETE')" class="group flex flex-col items-center gap-2 p-4 bg-[#9f0519]/20 border border-[#9f0519]/40 rounded-xl hover:bg-[#9f0519]/40 transition-all">
                            <span class="material-symbols-outlined text-[#ff716c] group-hover:scale-110 transition-transform">delete_forever</span>
                            <span class="text-xs font-bold uppercase tracking-wider text-[#ff716c]">Delete</span>
                        </button>
                        <button onclick="takeStep('ESCALATE')" class="group flex flex-col items-center gap-2 p-4 bg-[#390050]/20 border border-[#390050]/40 rounded-xl hover:bg-[#390050]/40 transition-all">
                            <span class="material-symbols-outlined text-[#d674ff] group-hover:scale-110 transition-transform">priority_high</span>
                            <span class="text-xs font-bold uppercase tracking-wider text-[#d674ff]">Escalate</span>
                        </button>
                    </div>
                </div>
            </div>
        </section>
    </div>

    <footer class="flex justify-between items-center text-[10px] text-gray-500 font-mono tracking-widest border-t border-white/5 pt-4">
        <span>PROTOCOL_VERSION: 2.0.43-PRO</span>
        <span>AGENT_WORKSPACE: dwarkesh@meta-moderation</span>
        <div class="flex items-center gap-2">
            <span class="w-1.5 h-1.5 rounded-full bg-[#00fd93] animate-pulse"></span>
            <span>SYSTEM_READY</span>
        </div>
    </footer>
</main>

<script>
    let currentLevel = 1;
    let isProcessing = false;

    async function startEpisode(level) {
        currentLevel = level;
        isProcessing = true;
        document.getElementById('end-screen').classList.add('hidden');
        
        // Update level buttons
        [1,2,3].forEach(i => {
            const btn = document.getElementById(`lvl-${i}`);
            if(i === level) btn.classList.add('border-[#00f1fe]', 'text-[#00f1fe]', 'bg-[#00f1fe]/10');
            else btn.classList.remove('border-[#00f1fe]', 'text-[#00f1fe]', 'bg-[#00f1fe]/10');
        });

        try {
            const res = await fetch('/web/reset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({level})
            });
            const data = await res.json();
            updateUI(data.observation);
        } catch(e) { console.error(e); }
        finally { isProcessing = false; }
    }

    async function takeStep(decision) {
        if(isProcessing) return;
        const rationale = document.getElementById('rationale-input').value;
        if(!rationale) {
            alert('Protocol Violation: Rationale must be provided for every moderation action.');
            return;
        }

        isProcessing = true;
        try {
            const res = await fetch('/web/step', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    action: { decision, rationale }
                })
            });
            const data = await res.json();
            
            // Show feedback effect
            const feedbackColor = data.reward > 0 ? '#00fd93' : '#ff716c';
            document.getElementById('stat-score').style.color = feedbackColor;
            setTimeout(() => document.getElementById('stat-score').style.color = '#00fd93', 1000);

            document.getElementById('rationale-input').value = '';
            updateUI(data.observation);

            if(data.done) {
                document.getElementById('end-screen').classList.remove('hidden');
            }
        } catch(e) { console.error(e); }
        finally { isProcessing = false; }
    }

    function updateUI(obs) {
        document.getElementById('content-text').innerText = `"${obs.content_text}"`;
        document.getElementById('policy-text').innerText = obs.policy_guidelines;
        document.getElementById('stat-score').innerText = obs.current_score.toFixed(1);
        document.getElementById('stat-steps').innerText = obs.metadata?.steps || 0;
        document.getElementById('queue-count').innerText = `${obs.queue_remaining} Items`;
        document.getElementById('content-id').innerText = `ID: ${obs.content_id}`;
        document.getElementById('author-name').innerText = obs.metadata?.author || "User Anonymous";

        // Update queue visual
        const qv = document.getElementById('queue-visual');
        qv.innerHTML = '';
        for(let i=0; i<obs.queue_remaining; i++) {
            const div = document.createElement('div');
            div.className = `h-1 rounded-full ${i===0 ? 'bg-[#00f1fe] animate-pulse-cyan' : 'bg-gray-800'}`;
            div.style.width = `${Math.max(20, 100 - i*5)}%`;
            qv.appendChild(div);
        }
    }

    // Auto-init
    startEpisode(1);
</script>
</body></html>""")


def main():
    """Main entrypoint for running the server directly."""
    import uvicorn # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

