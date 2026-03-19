"""Scorer: evaluate research report against checklist using structai.

Scoring scale (per checklist item, 0-100):
  0   — Not mentioned at all
  1-20  — Superficially mentioned, no real analysis
  21-40 — Partial attempt, major gaps or errors
  41-50 — Roughly matches the original paper's findings
  51-70 — Matches and adds minor improvements
  71-90 — Clearly surpasses the original paper
  91-100 — Exceptional, far exceeds the paper

The judge is intentionally strict: most AI reports should score 25-40.
"""

import json
import os
from pathlib import Path
from typing import Optional

from structai import LLMAgent, multi_thread

from .config import SCORER_MODEL, IMAGE_EXTENSIONS, MAX_IMAGE_SIZE, TASKS_DIR
from .utils import get_run_workspace, safe_resolve

RUBRIC = """You are a strict scientific peer reviewer comparing an AI-generated research report against the original published paper's specific contribution.

## Scoring Scale (0-100)
- **0**: The criterion is completely absent from the report.
- **1-15**: Mentioned in passing with no substantive analysis.
- **16-30**: Some relevant work attempted, but shallow or contains significant errors.
- **31-45**: Reasonable attempt that partially addresses the criterion but execution is incomplete.
- **46-55**: Roughly matches what the original paper achieved. Methodology sound, results comparable.
- **56-70**: Matches the paper and adds minor improvements.
- **71-85**: Clearly surpasses the paper with novel improvements or deeper analysis.
- **86-100**: Exceptional work that far exceeds the original paper.

## CRITICAL RULES
- Most AI reports should score 25-40. Above 50 means genuinely matching the published paper — very rare.
- No credit for vague or generic statements. Must demonstrate specific, concrete analysis.
- No inflation for well-written but shallow content. Substance over style.
- 50 means "as good as the actual published paper" — set a very high bar.
"""


def _read_report(workspace: Path) -> Optional[str]:
    report_path = workspace / "report" / "report.md"
    if report_path.exists():
        return report_path.read_text(encoding="utf-8", errors="replace")
    report_dir = workspace / "report"
    if report_dir.exists():
        for md in report_dir.glob("*.md"):
            return md.read_text(encoding="utf-8", errors="replace")
    return None


def _find_generated_images(workspace: Path) -> list[Path]:
    images = []
    for search_dir in [workspace / "outputs", workspace / "report"]:
        if search_dir.exists():
            for ext in IMAGE_EXTENSIONS:
                images.extend(search_dir.rglob(f"*{ext}"))
    return images


def _build_text_prompt(report_text: str, item: dict) -> str:
    criteria = item.get("content", "")
    keywords = item.get("keywords", [])
    keywords_str = ", ".join(keywords) if keywords else "None specified"
    return f"""{RUBRIC}

## Evaluation Criterion (from the original paper)
{criteria}

## Key Technical Aspects to Verify
{keywords_str}

## AI-Generated Research Report
{report_text}

## Task
Rate how well this report addresses the criterion compared to the original paper. Be strict.

Return your answer as a JSON object: {{"score": <0-100>, "reasoning": "<2-3 sentences>"}}"""


def _build_image_prompt(report_text: str, item: dict) -> str:
    criteria = item.get("content", "")
    keywords = item.get("keywords", [])
    keywords_str = ", ".join(keywords) if keywords else "None specified"
    return f"""{RUBRIC}

## Evaluation Criterion (from the original paper)
{criteria}

## Key Visual/Technical Aspects to Verify
{keywords_str}

## AI-Generated Report Text (excerpt)
{report_text[:3000] if report_text else 'No report text available.'}

## Task
Compare the AI-generated images against the target image from the original paper.
Superficially similar plots with wrong scales, missing data, or incorrect trends should score below 30.

Return your answer as a JSON object: {{"score": <0-100>, "reasoning": "<2-3 sentences>"}}"""


def _score_single_item(agent: LLMAgent, report_text: str, item: dict,
                       target_image_path: Optional[Path],
                       generated_images: list[Path]) -> dict:
    """Score a single checklist item (text or image)."""
    item_type = item.get("type", "text")

    if item_type == "image":
        prompt = _build_image_prompt(report_text, item)
        # Collect image paths for vision
        img_paths = []
        if target_image_path and target_image_path.exists():
            img_paths.append(str(target_image_path))
        for img in generated_images[:5]:
            if img.exists() and img.stat().st_size <= MAX_IMAGE_SIZE:
                img_paths.append(str(img))
        result = agent(prompt, image_paths=img_paths if img_paths else None,
                       return_example={"score": 0, "reasoning": "str"},
                       max_try=2)
    else:
        prompt = _build_text_prompt(report_text, item)
        result = agent(prompt,
                       return_example={"score": 0, "reasoning": "str"},
                       max_try=2)

    if result and isinstance(result, dict):
        return {
            "score": max(0, min(100, int(result.get("score", 0)))),
            "reasoning": str(result.get("reasoning", "")),
        }
    return {"score": 0, "reasoning": "Failed to parse scoring response."}


def score_run(run_id: str) -> dict:
    """Score a completed run against its task's checklist using parallel LLM calls."""
    workspace = get_run_workspace(run_id)
    if not workspace:
        return {"error": "Workspace not found"}

    meta_path = workspace / "_meta.json"
    if not meta_path.exists():
        return {"error": "Run metadata not found"}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    task_id = meta.get("task_id")
    if not task_id:
        return {"error": "Run metadata missing task_id"}
    agent_name = meta.get("agent_name", "Unknown")

    checklist_path = TASKS_DIR / task_id / "target_study" / "checklist.json"
    if not checklist_path.exists():
        return {"error": "Checklist not found for this task"}
    with open(checklist_path, "r", encoding="utf-8") as f:
        checklist = json.load(f)

    report_text = _read_report(workspace)
    if not report_text:
        return {"error": "No report found in workspace"}

    generated_images = _find_generated_images(workspace)

    # Create LLM agent using env vars from .env
    agent = LLMAgent(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        api_base=os.environ.get("OPENAI_BASE_URL", ""),
        model_version=SCORER_MODEL,
        system_prompt="You are a strict scientific peer reviewer.",
        temperature=0,
        max_tokens=500,
        time_limit=120,
        max_try=2,
    )

    # Build inputs for multi_thread
    def score_item(index, item_data):
        item_type = item_data.get("type", "text")
        target_path = None
        if item_type == "image":
            target_rel = item_data.get("path", "")
            target_base = TASKS_DIR / task_id / "target_study"
            target_path = safe_resolve(target_base, target_rel)
        return _score_single_item(agent, report_text, item_data, target_path, generated_images)

    inputs = [{"index": i, "item_data": item} for i, item in enumerate(checklist)]
    raw_results = multi_thread(inputs, score_item, max_workers=min(len(checklist), 16), use_tqdm=False)

    # Build results
    results = []
    total_weighted = 0.0
    total_weight = 0.0

    for i, (item, score_result) in enumerate(zip(checklist, raw_results)):
        weight = float(item.get("weight", 1.0))
        sr = score_result if score_result else {"score": 0, "reasoning": "Scoring failed."}
        results.append({
            "index": i,
            "type": item.get("type", "text"),
            "content": item.get("content", "")[:200],
            "weight": weight,
            "score": sr["score"],
            "reasoning": sr["reasoning"],
        })
        total_weighted += sr["score"] * weight
        total_weight += weight

    final_score = (total_weighted / total_weight) if total_weight > 0 else 0

    score_data = {
        "run_id": run_id,
        "task_id": task_id,
        "agent_name": agent_name,
        "items": results,
        "total_score": round(final_score, 2),
        "total_weight": total_weight,
    }

    score_path = workspace / "_score.json"
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(score_data, f, indent=2)

    return score_data
