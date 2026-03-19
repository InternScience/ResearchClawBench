"""Utility functions for task/run discovery, file trees, and path safety."""

import json
import os
import re
from pathlib import Path
from typing import Optional

from .config import TASKS_DIR, WORKSPACES_DIR


def list_tasks():
    """Return list of task IDs sorted alphabetically."""
    if not TASKS_DIR.exists():
        return []
    return sorted(
        d.name for d in TASKS_DIR.iterdir()
        if d.is_dir() and (d / "task_info.json").exists()
    )


def list_tasks_grouped():
    """Return tasks grouped by domain: {domain: [task_id, ...]}."""
    groups = {}
    for task_id in list_tasks():
        domain = re.match(r"([A-Za-z]+)_", task_id)
        domain_name = domain.group(1) if domain else "Other"
        groups.setdefault(domain_name, []).append(task_id)
    return groups


def load_task_info(task_id: str) -> dict:
    """Load task_info.json for a task."""
    path = TASKS_DIR / task_id / "task_info.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checklist(task_id: str) -> list:
    """Load checklist.json from target_study/."""
    path = TASKS_DIR / task_id / "target_study" / "checklist.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_paper_path(task_id: str) -> Optional[Path]:
    """Return path to the target paper PDF if it exists."""
    target_dir = TASKS_DIR / task_id / "target_study"
    if not target_dir.exists():
        return None
    for f in target_dir.iterdir():
        if f.suffix == ".pdf" and f.name.startswith("paper"):
            return f
    return None


def list_runs(task_id: Optional[str] = None):
    """List all runs, optionally filtered by task_id.

    Returns list of dicts: {run_id, task_id, timestamp, status, workspace}.
    """
    if not WORKSPACES_DIR.exists():
        return []
    runs = []
    for d in sorted(WORKSPACES_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "_meta.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if task_id and meta.get("task_id") != task_id:
            continue
        runs.append({
            "run_id": d.name,
            "task_id": meta.get("task_id"),
            "timestamp": meta.get("timestamp"),
            "status": meta.get("status", "unknown"),
            "agent_name": meta.get("agent_name", ""),
            "model": meta.get("model", ""),
            "duration_seconds": meta.get("duration_seconds"),
            "workspace": str(d),
        })
    return runs


def get_run_workspace(run_id: str) -> Optional[Path]:
    """Return workspace path for a run."""
    ws = WORKSPACES_DIR / run_id
    if ws.is_dir():
        return ws
    return None


def safe_resolve(base: Path, user_path: str) -> Optional[Path]:
    """Resolve user_path relative to base, preventing directory traversal.

    Returns resolved Path if safe, None if traversal detected.
    """
    try:
        resolved = (base / user_path).resolve()
        base_resolved = base.resolve()
        if resolved == base_resolved or str(resolved).startswith(str(base_resolved) + os.sep):
            return resolved
    except (ValueError, OSError):
        pass
    return None


def build_file_tree(root: Path, prefix: str = "") -> list:
    """Build a file tree as a list of dicts for a directory.

    Returns a flat list where directories appear before their children.
    Skips hidden files and internal metadata files.
    """
    skip_names = {"_meta.json", "_agent_output.jsonl", "_score.json", ".claude"}
    tree = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return tree

    for entry in entries:
        if entry.name.startswith(".") or entry.name in skip_names:
            continue
        rel = f"{prefix}/{entry.name}" if prefix else entry.name
        if entry.is_dir():
            tree.append({"name": entry.name, "path": rel, "type": "directory"})
            tree.extend(build_file_tree(entry, rel))
        else:
            try:
                stat = entry.stat()
            except OSError:
                continue
            tree.append({
                "name": entry.name,
                "path": rel,
                "type": "file",
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })
    return tree
