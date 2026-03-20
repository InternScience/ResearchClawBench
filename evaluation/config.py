"""Paths and constants for the evaluation system."""

import os
from pathlib import Path

# Project root (parent of evaluation/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tasks directory containing all benchmark tasks
TASKS_DIR = PROJECT_ROOT / "tasks"

# Workspaces directory for run outputs
WORKSPACES_DIR = PROJECT_ROOT / "workspaces"
WORKSPACES_DIR.mkdir(exist_ok=True)

# Scoring model (OpenAI-compatible; override via SCORER_MODEL env var)
SCORER_MODEL = os.environ.get("SCORER_MODEL", "gpt-5.1")

# Agent presets: name -> {cmd, label, icon}
# {prompt_file} and {workspace} are substituted at runtime
# Command runs with cwd=workspace, {prompt_file} points to INSTRUCTIONS.md
AGENT_PRESETS = {
    "claude": {
        "label": "Claude Code",
        "icon": "C",
        "logo": "/static/logos/anthropic.svg",
        "cmd": 'claude --dangerously-skip-permissions -p "{prompt_file}" --output-format stream-json --verbose',
    },
    "codex": {
        "label": "Codex CLI",
        "icon": "X",
        "logo": "/static/logos/openai.svg",
        "cmd": 'codex exec --full-auto "$(cat \'{prompt_file}\')"',
    },
    "openclaw": {
        "label": "OpenClaw",
        "icon": "O",
        "logo": "/static/logos/openclaw.svg",
        "cmd": "openclaw agent --agent main --timeout 3600 --message \"$(printf 'You are an autonomous research agent. Do NOT output a plan or ask for confirmation. Immediately start using your tools (exec, write, read) to complete the task step by step. Install packages with pip, write Python code, run it, generate figures, and write the report. Work inside: {workspace}\\n\\n'; cat '{prompt_file}')\"",
    },
}

# Image extensions recognized for vision scoring
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"}

# Max image size for base64 encoding (10MB)
MAX_IMAGE_SIZE = 10 * 1024 * 1024
