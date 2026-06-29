from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from evaluation import utils
from evaluation import cli_eval


class FakeResearchClawBenchAgent:
    def __init__(self, *, trace_dir: str | None, **kwargs):
        self.trace_dir = Path(trace_dir) if trace_dir else None

    def _run_session(self, prompt: str, workspace_root: str, event_callback):
        workspace = Path(workspace_root)
        event_callback({"type": "assistant", "content": "smoke run"})
        report_path = workspace / "report" / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("# Smoke report\n", encoding="utf-8")
        if self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = self.trace_dir / "trace.jsonl"
            trace_path.write_text(json.dumps({"prompt_seen": bool(prompt)}) + "\n", encoding="utf-8")
            trace_path_text = str(trace_path)
        else:
            trace_path_text = ""
        return {"termination": "result", "trace_path": trace_path_text, "session_state_path": ""}


def fake_default_llm_config(model_name: str | None = None, extra_body: dict | None = None) -> dict:
    config = {
        "model": model_name or "fake-model",
        "api_base": "",
        "api_key": "",
        "generate_cfg": {},
    }
    if extra_body:
        config["extra_body"] = dict(extra_body)
    return config


def fake_researchharness():
    return FakeResearchClawBenchAgent, fake_default_llm_config, "fake role prompt"


def write_config(path: Path, *, repeats: int = 1, concurrency: int = 1) -> None:
    path.write_text(
        f"""
name: cli_smoke
agent_model:
  name: fake-model
  api_base: http://example.invalid/v1
  api_key: fake-key
tasks:
  - id: Material_002
    repeats: {repeats}
repeats_per_task: 1
max_concurrent_runs: {concurrency}
judge_model:
  enabled: false
""".strip()
        + "\n",
        encoding="utf-8",
    )


class CliEvalSmokeTests(TestCase):
    def setUp(self):
        cli_eval._ALLOCATED_RUN_IDS.clear()
        with cli_eval._ACTIVE_RUNS_LOCK:
            cli_eval._ACTIVE_RUNS.clear()

    def test_public_example_configs_dry_run(self):
        repo_root = Path(__file__).resolve().parents[1]
        examples = [
            "researchharness_example_1_single_task.yaml",
            "researchharness_example_2_mixed_repeats.yaml",
            "researchharness_example_3_all_tasks.yaml",
            "researchharness_example_4_qwen_thinking.yaml",
        ]
        for example in examples:
            with self.subTest(example=example):
                with (
                    patch.object(cli_eval, "_load_researchharness", fake_researchharness),
                    redirect_stdout(StringIO()),
                ):
                    code = cli_eval.run_eval(
                        repo_root / "eval_configs" / example,
                        dry_run=True,
                        no_score=True,
                        skip_secret_check=True,
                    )
                self.assertEqual(code, 0)

    def test_mixed_repeats_and_all_tasks_are_resolved(self):
        repo_root = Path(__file__).resolve().parents[1]
        mixed = cli_eval._load_yaml(repo_root / "eval_configs" / "researchharness_example_2_mixed_repeats.yaml")
        mixed_plan = cli_eval._resolve_task_plan(mixed, default_repeats=1)
        self.assertEqual(
            [(item.task_id, item.repeats) for item in mixed_plan],
            [("Earth_000", 1), ("Physics_003", 2), ("Chemistry_002", 3)],
        )

        all_tasks = cli_eval._load_yaml(repo_root / "eval_configs" / "researchharness_example_3_all_tasks.yaml")
        all_plan = cli_eval._resolve_task_plan(all_tasks, default_repeats=1)
        self.assertGreaterEqual(len(all_plan), 40)
        self.assertTrue(all(item.repeats == 1 for item in all_plan))

    def test_dry_run_does_not_create_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "eval.yaml"
            workspace_root = tmp_path / "workspaces"
            write_config(config_path)

            with (
                patch.object(cli_eval, "WORKSPACES_DIR", workspace_root),
                patch.object(cli_eval, "_load_researchharness", fake_researchharness),
                redirect_stdout(StringIO()),
            ):
                code = cli_eval.run_eval(
                    config_path,
                    dry_run=True,
                    no_score=True,
                    skip_secret_check=False,
                )

            self.assertEqual(code, 0)
            self.assertFalse(workspace_root.exists())

    def test_legacy_model_sections_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "eval.yaml"
            config_path.write_text(
                """
name: legacy_config
model:
  name: fake-model
  api_base: http://example.invalid/v1
  api_key: fake-key
tasks:
  - id: Material_002
scorer:
  enabled: false
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with (
                patch.object(cli_eval, "_load_researchharness", fake_researchharness),
                redirect_stdout(StringIO()),
            ):
                with self.assertRaises(cli_eval.EvalConfigError) as ctx:
                    cli_eval.run_eval(
                        config_path,
                        dry_run=True,
                        no_score=True,
                        skip_secret_check=False,
                    )

            self.assertIn("agent_model", str(ctx.exception))

    def test_agent_api_env_names_are_required_even_when_secret_check_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "eval.yaml"
            config_path.write_text(
                """
name: missing_env_names
agent_model:
  name: fake-model
tasks:
  - id: Material_002
judge_model:
  enabled: false
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with (
                patch.object(cli_eval, "_load_researchharness", fake_researchharness),
                redirect_stdout(StringIO()),
            ):
                with self.assertRaises(cli_eval.EvalConfigError) as ctx:
                    cli_eval.run_eval(
                        config_path,
                        dry_run=True,
                        no_score=True,
                        skip_secret_check=True,
                    )

            self.assertIn("agent_model.api_base", str(ctx.exception))

    def test_removed_max_llm_calls_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "eval.yaml"
            config_path.write_text(
                """
name: removed_max_llm_calls
agent_model:
  name: fake-model
  api_base: http://example.invalid/v1
  api_key: fake-key
tasks:
  - id: Material_002
researchharness:
  max_llm_calls: 100
  max_rounds: 500
judge_model:
  enabled: false
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with (
                patch.object(cli_eval, "_load_researchharness", fake_researchharness),
                redirect_stdout(StringIO()),
            ):
                with self.assertRaises(cli_eval.EvalConfigError) as ctx:
                    cli_eval.run_eval(
                        config_path,
                        dry_run=True,
                        no_score=True,
                        skip_secret_check=False,
                    )

            self.assertIn("max_llm_calls", str(ctx.exception))

    def test_model_name_envs_are_resolved_explicitly(self):
        config = {
            "agent_model": {
                "name_env": "AGENT_MODEL_NAME",
                "api_base": "http://example.invalid/agent",
                "api_key": "agent-key",
                "extra_body": {"enable_thinking": False},
            },
            "judge_model": {
                "enabled": True,
                "name_env": "JUDGE_MODEL_NAME",
                "api_base": "http://example.invalid/judge",
                "api_key": "judge-key",
            },
        }
        with patch.dict(os.environ, {"AGENT_MODEL_NAME": "agent-test", "JUDGE_MODEL_NAME": "judge-test"}):
            model = cli_eval._load_model_config(config, require_secrets=True)
            scorer = cli_eval._load_scorer_config(config, require_secrets=True, force_disabled=False)

        self.assertEqual(model.name, "agent-test")
        self.assertEqual(model.name_source, "AGENT_MODEL_NAME")
        self.assertEqual(model.extra_body, {"enable_thinking": False})
        llm = cli_eval._build_llm_config(fake_default_llm_config, {"researchharness": {}}, model)
        self.assertEqual(llm["extra_body"], {"enable_thinking": False})
        self.assertEqual(scorer.model, "judge-test")
        self.assertEqual(scorer.model_source, "JUDGE_MODEL_NAME")

    def test_researchharness_timeout_fields_are_specific(self):
        model = cli_eval.ModelConfig(
            name="fake-model",
            api_base="http://example.invalid/v1",
            api_key="fake-key",
            extra_body={},
            name_source="literal",
            api_key_source="literal",
            api_base_source="literal",
        )
        config = {
            "researchharness": {
                "llm_request_timeout_seconds": 1200,
                "webfetch_tool_timeout_seconds": 300,
                "readpdf_tool_timeout_seconds": 301,
            }
        }

        llm = cli_eval._build_llm_config(fake_default_llm_config, config, model)
        self.assertEqual(llm["timeout_seconds"], 1200.0)
        self.assertEqual(
            cli_eval._researchharness_tool_timeout_env(config),
            {
                "WEBFETCH_TIMEOUT_SECONDS": "300",
                "READPDF_TIMEOUT_SECONDS": "301",
            },
        )

        rejected = [
            ("timeout_seconds", "llm_request_timeout_seconds"),
            ("webfetch_timeout_seconds", "webfetch_tool_timeout_seconds"),
            ("readpdf_timeout_seconds", "readpdf_tool_timeout_seconds"),
        ]
        for old_name, new_name in rejected:
            with self.subTest(old_name=old_name):
                with self.assertRaises(cli_eval.EvalConfigError) as ctx:
                    cli_eval._validate_researchharness_config({"researchharness": {old_name: 1}})
                self.assertIn(new_name, str(ctx.exception))

    def test_missing_model_name_env_value_is_rejected_for_real_runs(self):
        config = {
            "agent_model": {
                "name_env": "MISSING_AGENT_MODEL_NAME",
                "api_base": "http://example.invalid/agent",
                "api_key": "agent-key",
            }
        }
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(cli_eval.EvalConfigError) as ctx:
                cli_eval._load_model_config(config, require_secrets=True)

        self.assertIn("MISSING_AGENT_MODEL_NAME", str(ctx.exception))

    def test_real_run_requires_researchharness_tool_env_before_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "eval.yaml"
            workspace_root = tmp_path / "workspaces"
            write_config(config_path)

            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(cli_eval, "WORKSPACES_DIR", workspace_root),
                patch.object(cli_eval, "_load_researchharness", fake_researchharness),
                redirect_stdout(StringIO()),
            ):
                with self.assertRaises(cli_eval.EvalConfigError) as ctx:
                    cli_eval.run_eval(
                        config_path,
                        dry_run=False,
                        no_score=True,
                        skip_secret_check=False,
                    )

            self.assertIn("SERPER_KEY", str(ctx.exception))
            self.assertIn("JINA_KEY", str(ctx.exception))
            self.assertIn("MINERU_TOKEN", str(ctx.exception))
            self.assertFalse(workspace_root.exists())

    def test_tool_preflight_uses_current_environment_without_proxy_rewrites(self):
        seen_proxy_values = []

        def check_result(name: str):
            seen_proxy_values.append(
                (
                    os.environ.get("HTTP_PROXY"),
                    os.environ.get("HTTPS_PROXY"),
                    os.environ.get("NO_PROXY"),
                )
            )
            return {"name": name, "status": "PASS", "detail": "ok"}

        env = {
            "HTTP_PROXY": "http://proxy.example:7890",
            "HTTPS_PROXY": "http://proxy.example:7890",
            "NO_PROXY": "127.0.0.1,localhost",
            "SERPER_KEY": "serper",
            "JINA_KEY": "jina",
            "MINERU_TOKEN": "mineru",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(cli_eval, "_check_serper_tool", side_effect=lambda: check_result("SERPER_KEY/WebSearch")),
            patch.object(cli_eval, "_check_jina_tool", side_effect=lambda: check_result("JINA_KEY/WebFetch")),
            patch.object(cli_eval, "_check_mineru_tool", side_effect=lambda: check_result("MINERU_TOKEN/ReadPDF")),
        ):
            ok, results, reason = cli_eval._run_researchharness_tool_preflight()

            self.assertTrue(ok)
            self.assertEqual(reason, "")
            self.assertEqual(len(results), 3)
            self.assertEqual(
                seen_proxy_values,
                [
                    ("http://proxy.example:7890", "http://proxy.example:7890", "127.0.0.1,localhost"),
                    ("http://proxy.example:7890", "http://proxy.example:7890", "127.0.0.1,localhost"),
                    ("http://proxy.example:7890", "http://proxy.example:7890", "127.0.0.1,localhost"),
                ],
            )
            self.assertEqual(os.environ["HTTP_PROXY"], "http://proxy.example:7890")
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://proxy.example:7890")
            self.assertEqual(os.environ["NO_PROXY"], "127.0.0.1,localhost")

    def test_tool_timeout_yaml_is_injected_for_preflight_and_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "eval.yaml"
            workspace_root = tmp_path / "workspaces"
            config_path.write_text(
                """
name: timeout_injection
agent_model:
  name: fake-model
  api_base: http://example.invalid/v1
  api_key: fake-key
tasks:
  - id: Material_002
repeats_per_task: 1
max_concurrent_runs: 1
researchharness:
  webfetch_tool_timeout_seconds: 321
  readpdf_tool_timeout_seconds: 322
judge_model:
  enabled: false
""".strip()
                + "\n",
                encoding="utf-8",
            )
            seen_timeouts = []

            def fake_preflight():
                seen_timeouts.append(
                    (
                        os.environ.get("WEBFETCH_TIMEOUT_SECONDS"),
                        os.environ.get("READPDF_TIMEOUT_SECONDS"),
                    )
                )
                return True, [{"name": "all", "status": "PASS", "detail": "ok"}], ""

            env = {
                "SERPER_KEY": "serper",
                "JINA_KEY": "jina",
                "MINERU_TOKEN": "mineru",
                "WEBFETCH_TIMEOUT_SECONDS": "111",
            }
            with (
                patch.dict(os.environ, env, clear=True),
                patch.object(cli_eval, "WORKSPACES_DIR", workspace_root),
                patch.object(cli_eval, "_load_researchharness", fake_researchharness),
                patch.object(cli_eval, "_run_researchharness_tool_preflight", side_effect=fake_preflight),
                redirect_stdout(StringIO()),
            ):
                code = cli_eval.run_eval(
                    config_path,
                    dry_run=False,
                    no_score=True,
                    skip_secret_check=False,
                )

                self.assertEqual(code, 0)
                self.assertEqual(seen_timeouts, [("321", "322")])
                self.assertEqual(os.environ["WEBFETCH_TIMEOUT_SECONDS"], "111")
                self.assertNotIn("READPDF_TIMEOUT_SECONDS", os.environ)

    def test_real_run_skips_when_researchharness_tool_preflight_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "eval.yaml"
            workspace_root = tmp_path / "workspaces"
            write_config(config_path)

            with (
                patch.dict(
                    os.environ,
                    {"SERPER_KEY": "serper", "JINA_KEY": "jina", "MINERU_TOKEN": "mineru"},
                ),
                patch.object(cli_eval, "WORKSPACES_DIR", workspace_root),
                patch.object(cli_eval, "_load_researchharness", fake_researchharness),
                patch.object(
                    cli_eval,
                    "_run_researchharness_tool_preflight",
                    return_value=(
                        False,
                        [{"name": "JINA_KEY/WebFetch", "status": "FAIL", "detail": "Jina balance is insufficient."}],
                        "ResearchHarness tool preflight failed: JINA_KEY/WebFetch: Jina balance is insufficient.",
                    ),
                ),
                redirect_stdout(StringIO()),
            ):
                code = cli_eval.run_eval(
                    config_path,
                    dry_run=False,
                    no_score=True,
                    skip_secret_check=False,
                )

            self.assertEqual(code, 1)
            batch_dir = next((workspace_root / "cli_runs").iterdir())
            run_dir = next(path for path in batch_dir.iterdir() if path.is_dir() and not path.name.endswith("_trace"))
            meta = json.loads((run_dir / "_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["status"], "skipped")
            self.assertEqual(meta["termination"], "tool_preflight_failed")
            self.assertIn("JINA_KEY", meta["skip_reason"])
            self.assertFalse((run_dir / "data").exists())
            self.assertFalse((run_dir / "related_work").exists())
            report_text = next(batch_dir.glob("eval_report_*.md")).read_text(encoding="utf-8")
            self.assertIn("## Tool Preflight Issues", report_text)
            self.assertIn("failed_tools", report_text)
            self.assertIn("failure_details", report_text)
            self.assertIn("JINA_KEY/WebFetch", report_text)
            self.assertIn("Jina balance is insufficient", report_text)

    def test_concurrent_smoke_uses_cli_workspace_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "eval.yaml"
            workspace_root = tmp_path / "workspaces"
            cli_root = workspace_root / "cli_runs"
            write_config(config_path, repeats=2, concurrency=2)
            stdout = StringIO()

            with (
                patch.dict(
                    os.environ,
                    {"SERPER_KEY": "serper", "JINA_KEY": "jina", "MINERU_TOKEN": "mineru"},
                ),
                patch.object(cli_eval, "WORKSPACES_DIR", workspace_root),
                patch.object(cli_eval, "_load_researchharness", fake_researchharness),
                patch.object(
                    cli_eval,
                    "_run_researchharness_tool_preflight",
                    return_value=(True, [{"name": "all", "status": "PASS", "detail": "ok"}], ""),
                ),
                redirect_stdout(stdout),
            ):
                code = cli_eval.run_eval(
                    config_path,
                    dry_run=False,
                    no_score=True,
                    skip_secret_check=False,
                )

            self.assertEqual(code, 0)
            self.assertEqual([path.name for path in workspace_root.iterdir()], ["cli_runs"])
            batch_dirs = sorted(path for path in cli_root.iterdir() if path.is_dir())
            self.assertEqual(len(batch_dirs), 1)
            batch_dir = batch_dirs[0]
            self.assertRegex(batch_dir.name, r"^cli_\d{8}_\d{6}_[0-9a-f]{8}$")
            report_files = sorted(batch_dir.glob("eval_report_*.md"))
            self.assertEqual([path.name for path in report_files], [f"eval_report_{batch_dir.name}.md"])
            run_dirs = sorted(
                path for path in batch_dir.iterdir()
                if path.is_dir() and not path.name.endswith("_trace")
            )
            self.assertEqual(len(run_dirs), 2)
            self.assertEqual(len({path.name for path in run_dirs}), 2)
            for run_dir in run_dirs:
                self.assertTrue(run_dir.is_relative_to(batch_dir))
                self.assertRegex(run_dir.name, r"^cli_Material_002_\d{8}_\d{6}_[0-9a-f]{8}$")
                self.assertTrue((run_dir / "INSTRUCTIONS.md").exists())
                self.assertTrue((run_dir / "_agent_output.jsonl").exists())
                self.assertTrue((run_dir / "report" / "report.md").exists())
                self.assertFalse((run_dir / "_researchharness_trace").exists())
                meta = json.loads((run_dir / "_meta.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["status"], "completed")
                self.assertEqual(meta["agent_name"], "ResearchHarness (fake-model)")
                self.assertEqual(meta["workspace"], str(run_dir))

            trace_dirs = sorted(path for path in batch_dir.iterdir() if path.name.endswith("_trace"))
            self.assertEqual(len(trace_dirs), 2)
            self.assertEqual({path.name for path in trace_dirs}, {f"{path.name}_trace" for path in run_dirs})

            with patch.object(utils, "WORKSPACES_DIR", workspace_root):
                self.assertEqual(utils.list_runs("Material_002"), [])
                self.assertEqual(utils.get_run_workspace(run_dirs[0].name), run_dirs[0])

            report_text = report_files[0].read_text(encoding="utf-8")
            self.assertIn(f"Batch ID: `{batch_dir.name}`", report_text)
            self.assertIn("Agent model extra_body: `{}`", report_text)
            self.assertIn("## Run Directories", report_text)
            for run_dir in run_dirs:
                self.assertIn(run_dir.name, report_text)
                self.assertIn(f"{run_dir.name}_trace", report_text)
            self.assertIn("## Per-Run Results", report_text)
            self.assertIn("## Per-Task Summary", report_text)
            self.assertIn("## Runtime Summary", report_text)
            self.assertIn("| metric | value |", report_text)
            self.assertIn("| max_concurrent_runs | 2 |", report_text)
            self.assertIn("| wall_clock_seconds |", report_text)
            self.assertIn("| mean_run_duration_seconds |", report_text)
            self.assertIn("| task_id | repeat | run_id | status | score |", report_text)
            self.assertIn("| task_id | runs | completed_runs | skipped_runs | scored_runs | mean |", report_text)
            self.assertIn("mean_duration_seconds", report_text)
            self.assertNotIn("```text", report_text)
            self.assertNotIn("```json", report_text)

            output_text = stdout.getvalue()
            self.assertIn("Per-run results:", output_text)
            self.assertIn("Per-task summary:", output_text)
            self.assertIn("Runtime summary:", output_text)
            self.assertIn("Overall summary:", output_text)
            self.assertIn(f"Evaluation report: {report_files[0]}", output_text)

    def test_get_run_workspace_ignores_invalid_cli_name_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / "workspaces"
            run_id = "cli_Information_002_20260608_050636_9bf734d3"

            bad_without_meta = workspace_root / "cli_runs" / "cli_bad_a" / run_id
            bad_without_meta.mkdir(parents=True)
            (bad_without_meta / "outputs").mkdir()

            bad_wrong_meta = workspace_root / "cli_runs" / "cli_bad_b" / run_id
            bad_wrong_meta.mkdir(parents=True)
            (bad_wrong_meta / "_meta.json").write_text(
                json.dumps({"run_id": "different_run_id", "task_id": "Information_002"}),
                encoding="utf-8",
            )

            good = workspace_root / "cli_runs" / "cli_good" / run_id
            good.mkdir(parents=True)
            (good / "_meta.json").write_text(
                json.dumps({"run_id": run_id, "task_id": "Information_002"}),
                encoding="utf-8",
            )

            with patch.object(utils, "WORKSPACES_DIR", workspace_root):
                self.assertEqual(utils.get_run_workspace(run_id), good)
                self.assertIsNone(utils.get_run_workspace(f"../{run_id}"))

    def test_cli_scoring_uses_exact_workspace_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exact_workspace"
            workspace.mkdir()
            scorer = cli_eval.ScorerConfig(
                enabled=True,
                model="fake-judge",
                api_base="http://example.invalid/v1",
                api_key="fake-key",
                model_source="test",
                api_key_source="test",
                api_base_source="test",
            )
            captured = {}

            def fake_score_workspace(path):
                captured["workspace"] = Path(path)
                return {"total_score": 42.0}

            from evaluation import score as score_module

            with patch.object(score_module, "score_workspace", side_effect=fake_score_workspace):
                score_value, score_data, score_error = cli_eval._score_completed_run(workspace, scorer)

            self.assertEqual(captured["workspace"], workspace)
            self.assertEqual(score_value, 42.0)
            self.assertEqual(score_data, {"total_score": 42.0})
            self.assertEqual(score_error, "")

    def test_active_runs_are_marked_failed_on_interrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runner = cli_eval.TaskRunner(
                "Material_002",
                agent_cmd="ResearchHarness Python API",
                agent_name="ResearchHarness (fake-model)",
            )
            runner.run_id = "cli_Material_002_20260606_000000_deadbeef"
            runner.workspace = tmp_path / runner.run_id
            runner.meta_path = runner.workspace / "_meta.json"
            runner.output_path = runner.workspace / "_agent_output.jsonl"
            runner.instructions_path = runner.workspace / "INSTRUCTIONS.md"
            runner.workspace.mkdir(parents=True)
            runner._write_meta("running")

            reason = "CLI evaluation interrupted by SIGTERM."
            cli_eval._register_active_run(
                runner,
                start_time=time.time() - 5,
                model_name="fake-model",
                config_name="interrupt_smoke",
                repeat_index=1,
            )
            cli_eval._mark_active_runs_interrupted(reason, 143)

            meta = json.loads(runner.meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["status"], "failed")
            self.assertEqual(meta["exit_code"], 143)
            self.assertEqual(meta["termination"], "interrupted")
            self.assertEqual(meta["error"], reason)
            self.assertEqual(meta["model"], "fake-model")
            self.assertEqual(meta["evaluation_config"], "interrupt_smoke")
            self.assertEqual(meta["repeat_index"], 1)
            self.assertGreaterEqual(meta["duration_seconds"], 0)
            self.assertIn(reason, runner.output_path.read_text(encoding="utf-8"))
            with cli_eval._ACTIVE_RUNS_LOCK:
                self.assertEqual(cli_eval._ACTIVE_RUNS, {})
