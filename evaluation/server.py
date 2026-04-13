"""Flask API server with SSE streaming for the evaluation system."""

import json
import mimetypes
import os
import re
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from .config import AGENT_PRESETS, TASKS_DIR, WORKSPACES_DIR
from .run_task import TaskRunner
from .score import score_run
from .utils import (
    build_file_tree,
    get_paper_path,
    get_run_workspace,
    list_runs,
    list_tasks_grouped,
    load_checklist,
    load_task_info,
    safe_resolve,
)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable static file caching
app.json.sort_keys = False  # Preserve JSON key order (e.g. agent presets)
CORS(app)

# Track active runners
_active_runners: dict[str, TaskRunner] = {}
RESEARCHHARNESS_LABEL = "ResearchHarness"


def _order_agent_labels(agent_names):
    names = set(agent_names)
    ordered = []
    for preset in AGENT_PRESETS.values():
        label = preset.get("label", "")
        if label in names and label != RESEARCHHARNESS_LABEL and label not in ordered:
            ordered.append(label)
    ordered.extend(
        sorted(
            name for name in names
            if name not in ordered and name != RESEARCHHARNESS_LABEL
        )
    )
    if RESEARCHHARNESS_LABEL in names:
        ordered.append(RESEARCHHARNESS_LABEL)
    return ordered


# --- Pages ---

@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.route("/api/config")
def api_config():
    """Return agent presets for the UI."""
    presets = {k: {"label": v["label"], "icon": v["icon"], "logo": v.get("logo", "")} for k, v in AGENT_PRESETS.items()}
    # Name→logo mapping for use in run history, leaderboard, etc.
    agent_logos = {v["label"]: v.get("logo", "") for v in AGENT_PRESETS.values()}
    return jsonify({"presets": presets, "agent_logos": agent_logos})


# --- Task APIs ---

@app.route("/api/tasks")
def api_tasks():
    """List all tasks grouped by domain."""
    return jsonify(list_tasks_grouped())


@app.route("/api/tasks/<task_id>/info")
def api_task_info(task_id):
    """Get task_info.json for a task."""
    try:
        return jsonify(load_task_info(task_id))
    except FileNotFoundError:
        return jsonify({"error": "Task not found"}), 404


@app.route("/api/tasks/<task_id>/checklist")
def api_task_checklist(task_id):
    """Get checklist.json for a task."""
    try:
        return jsonify(load_checklist(task_id))
    except FileNotFoundError:
        return jsonify({"error": "Checklist not found"}), 404


@app.route("/api/tasks/<task_id>/paper")
def api_task_paper(task_id):
    """Serve the target paper PDF."""
    paper_path = get_paper_path(task_id)
    if paper_path and paper_path.exists():
        return send_file(paper_path, mimetype="application/pdf")
    return jsonify({"error": "Paper not found"}), 404


@app.route("/api/tasks/<task_id>/files")
def api_task_files(task_id):
    """Get task file tree: data/, related_work/, INSTRUCTIONS.md + empty workspace dirs."""
    task_dir = TASKS_DIR / task_id
    if not task_dir.exists():
        return jsonify({"error": "Task not found"}), 404

    # Collect all top-level entries: real dirs + empty workspace dirs
    # We'll build them in alphabetical order
    top_dirs = {}

    # Real dirs with children
    for subdir in ["data", "related_work"]:
        sub_path = task_dir / subdir
        if sub_path.exists():
            top_dirs[subdir] = build_file_tree(sub_path, subdir, max_per_dir=10, max_depth=3)

    # Empty workspace dirs (no children)
    for d in ["code", "outputs", "report"]:
        if d not in top_dirs:
            top_dirs[d] = []
    # report/images as child of report
    top_dirs["report"].insert(0, {"name": "images", "path": "report/images", "type": "directory"})

    # Build final tree in alphabetical order
    tree = []
    for name in sorted(top_dirs.keys()):
        tree.append({"name": name, "path": name, "type": "directory"})
        tree.extend(top_dirs[name])

    # INSTRUCTIONS.md at the end (root-level file)
    tree.append({"name": "INSTRUCTIONS.md", "path": "INSTRUCTIONS.md", "type": "file", "size": 0})

    return jsonify(tree)


@app.route("/api/tasks/<task_id>/file")
def api_task_file(task_id):
    """Serve a file from the task directory, or generate INSTRUCTIONS.md."""
    task_dir = TASKS_DIR / task_id
    if not task_dir.exists():
        return jsonify({"error": "Task not found"}), 404

    file_path = request.args.get("path", "")
    if not file_path:
        return jsonify({"error": "path required"}), 400

    # Special: generate INSTRUCTIONS.md on-the-fly
    if file_path == "INSTRUCTIONS.md":
        runner = TaskRunner(task_id)
        content = runner._build_instructions()
        return content, 200, {"Content-Type": "text/markdown; charset=utf-8"}

    resolved = safe_resolve(task_dir, file_path)
    if not resolved or not resolved.exists() or not resolved.is_file():
        return jsonify({"error": "File not found"}), 404

    mime_type, _ = mimetypes.guess_type(str(resolved))
    return send_file(resolved, mimetype=mime_type or "application/octet-stream")


# --- Run APIs ---

@app.route("/api/runs", methods=["GET"])
def api_list_runs():
    """List all runs, optionally filtered by task_id."""
    task_id = request.args.get("task_id")
    return jsonify(list_runs(task_id))


@app.route("/api/runs", methods=["POST"])
def api_start_run():
    """Start a new run for a task."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400
    task_id = data.get("task_id")
    if not task_id:
        return jsonify({"error": "task_id required"}), 400

    # Validate task exists
    task_dir = TASKS_DIR / task_id
    if not task_dir.exists():
        return jsonify({"error": "Task not found"}), 404

    # Resolve agent command from presets
    agent = data.get("agent", "")
    if agent in AGENT_PRESETS:
        agent_cmd = AGENT_PRESETS[agent]["cmd"]
        agent_name = AGENT_PRESETS[agent]["label"]
    else:
        return jsonify({"error": "Unknown agent preset"}), 400

    runner = TaskRunner(task_id, agent_cmd=agent_cmd, agent_name=agent_name)
    run_id = runner.run_async()
    _active_runners[run_id] = runner
    # Clean up finished runners (process started AND exited) to prevent memory growth
    finished = [rid for rid, r in _active_runners.items()
                if rid != run_id and r.process is not None and r.process.poll() is not None]
    for rid in finished:
        del _active_runners[rid]

    return jsonify({
        "run_id": run_id,
        "task_id": task_id,
        "status": "running",
        "workspace": str(runner.workspace),
    })


@app.route("/api/runs/<run_id>/stop", methods=["POST"])
def api_stop_run(run_id):
    """Stop a running task."""
    runner = _active_runners.get(run_id)
    if runner and runner.process:
        try:
            runner.process.terminate()
        except OSError:
            pass
        return jsonify({"status": "stopped"})
    return jsonify({"error": "Run not found or not running"}), 404


@app.route("/api/runs/<run_id>/output")
def api_run_output(run_id):
    """Return saved agent output lines (last N lines via ?tail=)."""
    workspace = get_run_workspace(run_id)
    if not workspace:
        return jsonify({"error": "Run not found"}), 404
    output_path = workspace / "_agent_output.jsonl"
    if not output_path.exists():
        return jsonify([])
    lines = []
    with open(output_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    # Allow ?tail=N to return only the last N lines (default: all)
    tail = request.args.get("tail", type=int)
    if tail and tail > 0 and len(lines) > tail:
        lines = lines[-tail:]
    return jsonify(lines)


@app.route("/api/runs/<run_id>", methods=["DELETE"])
def api_delete_run(run_id):
    """Delete a run and its workspace."""
    # Stop the process first if still running
    runner = _active_runners.pop(run_id, None)
    if runner and runner.process:
        try:
            runner.process.terminate()
        except OSError:
            pass
    workspace = get_run_workspace(run_id)
    if not workspace:
        return jsonify({"error": "Run not found"}), 404
    import shutil
    shutil.rmtree(workspace, ignore_errors=True)
    return jsonify({"status": "deleted", "run_id": run_id})


@app.route("/api/runs/<run_id>/stream")
def api_run_stream(run_id):
    """SSE endpoint: stream Claude Code output in real-time."""
    workspace = get_run_workspace(run_id)
    if not workspace:
        return jsonify({"error": "Run not found"}), 404

    output_path = workspace / "_agent_output.jsonl"
    meta_path = workspace / "_meta.json"

    def generate():
        """Generator that yields SSE events."""
        MAX_INITIAL_LINES = 500  # Skip old history to prevent browser freeze on large outputs
        line_offset = 0
        keepalive_counter = 0
        initial = True
        while True:
            # Read new lines from output file
            if output_path.exists():
                try:
                    with open(output_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    if initial:
                        # On first connection, skip to last MAX_INITIAL_LINES to avoid flooding
                        line_offset = max(0, len(lines) - MAX_INITIAL_LINES)
                        initial = False
                    new_lines = lines[line_offset:]
                    for line in new_lines:
                        line = line.strip()
                        if line:
                            yield f"data: {line}\n\n"
                    line_offset = len(lines)
                except (OSError, IOError):
                    pass

            # Check if run is done
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("status") in ("completed", "failed"):
                        yield f"data: {json.dumps({'type': 'system', 'subtype': 'done', 'status': meta['status']})}\n\n"
                        break
                except (json.JSONDecodeError, OSError):
                    pass

            # Send keepalive every 20s to prevent connection timeout
            keepalive_counter += 1
            if keepalive_counter >= 40:
                yield ": keepalive\n\n"
                keepalive_counter = 0

            time.sleep(0.5)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/runs/<run_id>/meta")
def api_run_meta(run_id):
    """Get run metadata."""
    workspace = get_run_workspace(run_id)
    if not workspace:
        return jsonify({"error": "Run not found"}), 404
    meta_path = workspace / "_meta.json"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({"error": "Meta not found"}), 404


@app.route("/api/runs/<run_id>/input-files")
def api_run_input_files(run_id):
    """Get input files (data/, related_work/). These don't change during a run."""
    workspace = get_run_workspace(run_id)
    if not workspace:
        return jsonify({"error": "Run not found"}), 404
    tree = []
    for subdir in ["data", "related_work"]:
        sub = workspace / subdir
        if sub.exists():
            tree.append({"name": subdir, "path": subdir, "type": "directory"})
            tree.extend(build_file_tree(sub, subdir, max_per_dir=10, max_depth=3))
    return jsonify(tree)


@app.route("/api/runs/<run_id>/output-files")
def api_run_output_files(run_id):
    """Get agent-generated files (code/, outputs/, report/) + INSTRUCTIONS.md."""
    workspace = get_run_workspace(run_id)
    if not workspace:
        return jsonify({"error": "Run not found"}), 404
    tree = []
    for subdir in ["code", "outputs", "report"]:
        sub = workspace / subdir
        if sub.exists():
            tree.append({"name": subdir, "path": subdir, "type": "directory"})
            if subdir == "report":
                tree.extend(build_file_tree(sub, subdir))
            else:
                tree.extend(build_file_tree(sub, subdir, max_per_dir=10, max_depth=3))
    instr = workspace / "INSTRUCTIONS.md"
    if instr.exists():
        st = instr.stat()
        tree.append({"name": "INSTRUCTIONS.md", "path": "INSTRUCTIONS.md", "type": "file", "size": st.st_size, "mtime": st.st_mtime})
    return jsonify(tree)


@app.route("/api/runs/<run_id>/file")
def api_run_file(run_id):
    """Serve a file from the workspace with path traversal protection."""
    workspace = get_run_workspace(run_id)
    if not workspace:
        return jsonify({"error": "Run not found"}), 404

    file_path = request.args.get("path", "")
    if not file_path:
        return jsonify({"error": "path parameter required"}), 400

    resolved = safe_resolve(workspace, file_path)
    if not resolved or not resolved.exists() or not resolved.is_file():
        return jsonify({"error": "File not found or access denied"}), 404

    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(str(resolved))
    if not mime_type:
        mime_type = "application/octet-stream"

    return send_file(resolved, mimetype=mime_type)


def _xlsx_to_json(file_path):
    """Convert xlsx file to JSON rows (max 200 rows)."""
    try:
        if file_path.stat().st_size > 50 * 1024 * 1024:
            return {"error": "File too large to preview (>50MB)"}
        import openpyxl
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        ws = wb.active
        rows = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i >= 200:
                break
            rows.append([str(c) if c is not None else "" for c in row])
        wb.close()
        return {"rows": rows}
    except ImportError:
        return {"error": "openpyxl not installed"}
    except Exception as e:
        return {"error": str(e)}


@app.route("/api/runs/<run_id>/xlsx_preview")
def api_run_xlsx_preview(run_id):
    workspace = get_run_workspace(run_id)
    if not workspace:
        return jsonify({"error": "Run not found"}), 404
    file_path = request.args.get("path", "")
    resolved = safe_resolve(workspace, file_path)
    if not resolved or not resolved.exists():
        return jsonify({"error": "File not found"}), 404
    return jsonify(_xlsx_to_json(resolved))


@app.route("/api/tasks/<task_id>/xlsx_preview")
def api_task_xlsx_preview(task_id):
    task_dir = TASKS_DIR / task_id
    if not task_dir.exists():
        return jsonify({"error": "Task not found"}), 404
    file_path = request.args.get("path", "")
    resolved = safe_resolve(task_dir, file_path)
    if not resolved or not resolved.exists():
        return jsonify({"error": "File not found"}), 404
    return jsonify(_xlsx_to_json(resolved))


@app.route("/api/runs/<run_id>/report")
def api_run_report(run_id):
    """Get the report markdown with rewritten image URLs."""
    workspace = get_run_workspace(run_id)
    if not workspace:
        return jsonify({"error": "Run not found"}), 404

    report_path = workspace / "report" / "report.md"
    if not report_path.exists():
        # Try any .md in report/
        report_dir = workspace / "report"
        if not report_dir.exists():
            return jsonify({"error": "No report found"}), 404
        for md in report_dir.glob("*.md"):
            report_path = md
            break
        else:
            return jsonify({"error": "No report found"}), 404

    report_text = report_path.read_text(encoding="utf-8", errors="replace")

    # Rewrite relative image paths to API URLs
    # Match patterns like ![alt](../outputs/fig.png) or ![alt](outputs/fig.png)
    def rewrite_image_url(match):
        alt = match.group(1)
        raw_path = match.group(2)
        # Resolve relative to report/ directory
        img_resolved = (report_path.parent / raw_path).resolve()
        try:
            rel_to_workspace = img_resolved.relative_to(workspace.resolve())
            api_url = f"/api/runs/{run_id}/file?path={str(rel_to_workspace).replace(os.sep, '/')}"
            return f"![{alt}]({api_url})"
        except ValueError:
            return match.group(0)

    report_text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", rewrite_image_url, report_text)

    return jsonify({"markdown": report_text})


# --- Scoring APIs ---

@app.route("/api/runs/<run_id>/score", methods=["POST"])
def api_score_run(run_id):
    """Trigger scoring for a run."""
    workspace = get_run_workspace(run_id)
    if not workspace:
        return jsonify({"error": "Run not found"}), 404

    # Check if already scored
    score_path = workspace / "_score.json"
    if score_path.exists() and not request.args.get("force"):
        try:
            with open(score_path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass  # Fall through to re-score

    # Run scoring in background thread
    def do_score():
        try:
            score_run(run_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Write error to score file so UI can show it
            score_path = workspace / "_score.json"
            with open(score_path, "w", encoding="utf-8") as f:
                json.dump({"error": str(e)}, f)

    thread = threading.Thread(target=do_score, daemon=True)
    thread.start()

    return jsonify({"status": "scoring", "message": "Scoring started"})


@app.route("/api/runs/<run_id>/score", methods=["GET"])
def api_get_score(run_id):
    """Get scoring results."""
    workspace = get_run_workspace(run_id)
    if not workspace:
        return jsonify({"error": "Run not found"}), 404

    score_path = workspace / "_score.json"
    if not score_path.exists():
        return jsonify({"status": "not_scored"}), 404

    try:
        with open(score_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except (json.JSONDecodeError, OSError):
        return jsonify({"error": "Score file corrupted"}), 500


# --- Target study images ---

@app.route("/api/tasks/<task_id>/target_image")
def api_target_image(task_id):
    """Serve a target study image."""
    image_path = request.args.get("path", "")
    if not image_path:
        return jsonify({"error": "path required"}), 400
    target_dir = TASKS_DIR / task_id / "target_study"
    resolved = safe_resolve(target_dir, image_path)
    if not resolved or not resolved.exists():
        return jsonify({"error": "Image not found"}), 404
    mime_type, _ = mimetypes.guess_type(str(resolved))
    return send_file(resolved, mimetype=mime_type or "image/png")


# --- Leaderboard / Frontier ---

@app.route("/api/leaderboard")
def api_leaderboard():
    """Aggregate best scores per (task, agent) pair.

    Returns:
      tasks: [task_id, ...]
      agents: [agent_name, ...]
      scores: {agent_name: {task_id: {score, run_id, duration_seconds, model}, ...}, ...}
      frontier: {task_id: max_score_or_null, ...}
    """
    all_runs = list_runs()
    # For each (task, agent), keep only the best scored run.
    best = {}  # (task_id, agent_name) -> leaderboard cell metadata
    for run in all_runs:
        ws = get_run_workspace(run["run_id"])
        if not ws:
            continue
        score_path = ws / "_score.json"
        if not score_path.exists():
            continue
        try:
            with open(score_path, "r", encoding="utf-8") as f:
                score_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        task_id = run["task_id"]
        agent = score_data.get("agent_name", run.get("agent_name", "Unknown"))
        total = score_data.get("total_score", 0)
        entry = {
            "score": total,
            "run_id": run["run_id"],
            "duration_seconds": run.get("duration_seconds"),
            "model": run.get("model", ""),
        }
        key = (task_id, agent)
        if key not in best or total > best[key]["score"]:
            best[key] = entry

    # Build structured response
    tasks_set = set()
    agents_set = set()
    for (t, a) in best:
        tasks_set.add(t)
        agents_set.add(a)

    tasks_list = sorted(tasks_set)
    agents_list = _order_agent_labels(agents_set)

    scores = {}
    for agent in agents_list:
        scores[agent] = {}
        for task in tasks_list:
            key = (task, agent)
            if key in best:
                scores[agent][task] = best[key]

    # Frontier: max score per task across all agents
    frontier = {}
    for task in tasks_list:
        best_entry = None
        for agent in agents_list:
            key = (task, agent)
            if key in best and (best_entry is None or best[key]["score"] > best_entry["score"]):
                best_entry = best[key]
        frontier[task] = best_entry["score"] if best_entry else None

    return jsonify({
        "tasks": tasks_list,
        "agents": agents_list,
        "scores": scores,
        "frontier": frontier,
    })


def main():
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
