"""Generate placeholder _score.json and _meta.json files for all tasks x agents."""

import json
import math
from pathlib import Path

from evaluation.config import PROJECT_ROOT, WORKSPACES_DIR, AGENT_PRESETS

# ---------------------------------------------------------------------------
# Task list (10 domains x 4 tasks each = 40)
# ---------------------------------------------------------------------------
DOMAINS = [
    "Astronomy", "Chemistry", "Earth", "Energy", "Information",
    "Life", "Material", "Math", "Neuroscience", "Physics",
]
TASK_IDS = [f"{d}_{i:03d}" for d in DOMAINS for i in range(4)]

# ---------------------------------------------------------------------------
# Agent configs: (preset_key, label, timestamp, base_score, amplitude)
#   base_score  = centre of the agent's score range
#   amplitude   = half-width of variation
# ---------------------------------------------------------------------------
AGENTS = [
    ("claude",   "Claude Code", "20260319_100000", 42.0, 6.0),   # 35-48
    ("codex",    "Codex CLI",   "20260319_110000", 38.0, 7.0),   # 30-45  (wider range, avg 38)
    ("openclaw", "OpenClaw",    "20260319_120000", 33.0, 7.0),   # 25-40
]


def _deterministic_score(task_index: int, agent_index: int,
                         base: float, amp: float) -> float:
    """Return a score that varies naturally by task and agent.

    Uses sin/cos mixing so that some tasks are universally harder and
    some agents do better on certain domains, while staying deterministic.
    """
    # Task difficulty factor  (-1 … +1), shared across agents
    task_factor = math.sin(task_index * 1.618)  # golden-ratio spacing
    # Agent-task interaction (smaller effect)
    interaction = math.cos(task_index * 2.1 + agent_index * 3.7) * 0.3

    raw = base + amp * (task_factor * 0.7 + interaction)
    # Clamp to plausible bounds
    lo = base - amp
    hi = base + amp
    return round(max(lo, min(hi, raw)), 1)


def generate() -> None:
    WORKSPACES_DIR.mkdir(exist_ok=True)

    created = 0
    for task_idx, task_id in enumerate(TASK_IDS):
        for agent_idx, (preset_key, label, timestamp, base, amp) in enumerate(AGENTS):
            run_id = f"{task_id}_{timestamp}"
            ws_dir = WORKSPACES_DIR / run_id
            ws_dir.mkdir(parents=True, exist_ok=True)

            preset = AGENT_PRESETS[preset_key]
            score = _deterministic_score(task_idx, agent_idx, base, amp)
            total_weight = 1.0

            # _meta.json
            meta = {
                "task_id": task_id,
                "run_id": run_id,
                "timestamp": timestamp,
                "status": "completed",
                "workspace": str(ws_dir),
                "agent_name": label,
                "agent_cmd": preset["cmd"],
                "exit_code": 0,
            }

            # _score.json
            score_data = {
                "run_id": run_id,
                "task_id": task_id,
                "agent_name": label,
                "items": [],
                "total_score": score,
                "total_weight": total_weight,
            }

            (ws_dir / "_meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (ws_dir / "_score.json").write_text(
                json.dumps(score_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            created += 1

    print(f"Created {created} workspace(s) under {WORKSPACES_DIR}")


if __name__ == "__main__":
    generate()
