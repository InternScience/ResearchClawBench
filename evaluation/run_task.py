"""TaskRunner: workspace setup and agent subprocess execution."""

import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from .config import AGENT_PRESETS, TASKS_DIR, WORKSPACES_DIR
from .utils import load_task_info


class TaskRunner:
    """Sets up a workspace and runs an agent on a research task."""

    def __init__(self, task_id: str, agent_cmd: str = None, agent_name: str = "Unknown"):
        self.task_id = task_id
        self.task_dir = TASKS_DIR / task_id
        self.task_info = load_task_info(task_id)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{task_id}_{self.timestamp}"
        self.workspace = WORKSPACES_DIR / self.run_id
        self.meta_path = self.workspace / "_meta.json"
        self.output_path = self.workspace / "_agent_output.jsonl"
        self.instructions_path = self.workspace / "INSTRUCTIONS.md"
        self.agent_cmd = agent_cmd or AGENT_PRESETS["claude"]["cmd"]
        self.agent_name = agent_name
        self.process = None

    def setup_workspace(self):
        """Create workspace with data, related_work copies and working dirs."""
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Copy data/ directory
        src_data = self.task_dir / "data"
        if src_data.exists():
            shutil.copytree(src_data, self.workspace / "data", dirs_exist_ok=True)

        # Copy related_work/ directory
        src_rw = self.task_dir / "related_work"
        if src_rw.exists():
            shutil.copytree(src_rw, self.workspace / "related_work", dirs_exist_ok=True)

        # Create working directories
        for dirname in ("code", "outputs", "report", "report/images"):
            (self.workspace / dirname).mkdir(parents=True, exist_ok=True)

        # Write INSTRUCTIONS.md
        self.instructions_path.write_text(self._build_instructions(), encoding="utf-8")

        # Write initial meta with agent info
        self._write_meta("running")

    def _build_instructions(self) -> str:
        """Build the full INSTRUCTIONS.md content."""
        task_desc = self.task_info.get("task", "")
        data_parts = []
        for d in self.task_info.get("data", []):
            orig_path = d.get("path", "")
            ws_path = orig_path
            if "/data/" in orig_path:
                ws_path = "data/" + orig_path.split("/data/", 1)[1]
            data_parts.append(f"- **{d['name']}** (`{ws_path}`): {d.get('description', '')}")
        data_text = "\n".join(data_parts) if data_parts else "No specific data files."

        return f"""# Research Task

## Task Description
{task_desc}

## Available Data Files
{data_text}

## Workspace Layout
- `data/` — Input datasets (read-only, do not modify)
- `related_work/` — Reference papers and materials
- `code/` — Write your analysis code here
- `outputs/` — Save intermediate results
- `report/` — Write your final research report here
- `report/images/` — Save all report figures here

## Deliverables
1. Write analysis code in `code/` that processes the data
2. Save intermediate outputs to `outputs/`
3. Write a comprehensive research report as `report/report.md`
   - Include methodology, results, and discussion
   - Use proper academic writing style
   - **You MUST include figures in your report.** Generate plots, charts, and visualizations that support your analysis
   - Save all report figures to `report/images/` and reference them in the report using relative paths: `images/figure_name.png`
   - Include at least: data overview plots, main result figures, and comparison/validation plots

## Guidelines
- Install any needed Python packages via pip
- Use matplotlib/seaborn for visualization
- Ensure all code is reproducible
- Document your approach clearly in the report
"""

    def _write_meta(self, status: str, extra: dict = None):
        """Write or update _meta.json."""
        meta = {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "status": status,
            "workspace": str(self.workspace),
            "agent_name": self.agent_name,
            "agent_cmd": self.agent_cmd,
        }
        if extra:
            meta.update(extra)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # Env vars that should NOT leak into agent subprocesses (scorer-only)
    _STRIP_ENV_VARS = {"OPENAI_API_KEY", "OPENAI_BASE_URL", "SCORER_MODEL"}

    def run(self):
        """Launch agent subprocess, capture stdout to output file."""
        start_time = time.time()
        prompt_file_path = str(self.instructions_path.resolve())
        workspace_path = str(self.workspace.resolve())

        cmd_str = self.agent_cmd.format(
            prompt_file=prompt_file_path,
            workspace=workspace_path,
        )

        clean_env = {k: v for k, v in os.environ.items() if k not in self._STRIP_ENV_VARS}

        try:
            self.process = subprocess.Popen(
                cmd_str,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self.workspace),
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=True,
                env=clean_env,
            )

            with open(self.output_path, "w", encoding="utf-8") as out_f:
                for line in self.process.stdout:
                    line = line.rstrip("\n")
                    if line:
                        out_f.write(line + "\n")
                        out_f.flush()

            self.process.wait()
            exit_code = self.process.returncode
            detected_model = self._detect_model()
            duration = round(time.time() - start_time)
            self._write_meta(
                "completed" if exit_code == 0 else "failed",
                {"exit_code": exit_code, "model": detected_model, "duration_seconds": duration}
            )
        except Exception as e:
            if self.process and self.process.poll() is None:
                try:
                    self.process.terminate()
                except OSError:
                    pass
            duration = round(time.time() - start_time)
            self._write_meta("failed", {"error": str(e), "duration_seconds": duration})
        finally:
            # Guarantee meta is never left as "running"
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("status") == "running":
                    self._write_meta("failed", {"error": "Process ended unexpectedly"})
            except Exception:
                self._write_meta("failed", {"error": "Process ended unexpectedly"})

    def _detect_model(self) -> str:
        """Try to detect the model name from agent output."""
        import re
        try:
            with open(self.output_path, "r", encoding="utf-8", errors="replace") as f:
                # Only scan first 50 lines for efficiency
                for i, line in enumerate(f):
                    if i > 50:
                        break
                    # Codex: "model: gpt-5.4"
                    m = re.search(r'model:\s*(\S+)', line, re.IGNORECASE)
                    if m and not m.group(1).startswith('{'):
                        return m.group(1)
                    # Claude Code stream-json: look for model in JSON
                    if '"model"' in line:
                        try:
                            d = json.loads(line)
                            if isinstance(d, dict) and 'model' in d:
                                return d['model']
                        except (json.JSONDecodeError, KeyError):
                            pass
        except (OSError, IOError):
            pass
        return ""

    def run_async(self):
        """Launch in a daemon thread for non-blocking use."""
        self.setup_workspace()
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return self.run_id
