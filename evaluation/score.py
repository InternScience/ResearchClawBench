"""Scorer: evaluate research report against checklist using structai.

Two evaluation modes per checklist item (0-100 scale, 50 = matches paper):

  Objective (metric optimization / quantitative results):
    <50 = worse metrics, ~50 = comparable, >50 = better metrics

  Subjective (mechanism analysis / qualitative reasoning):
    <50 = weaker evidence/logic, ~50 = comparable, >50 = stronger

The judge evaluates end-to-end automated scientific discovery capability.
"""

import json
import os
from pathlib import Path
from typing import Optional

from structai import LLMAgent, multi_thread

from .config import SCORER_MODEL, IMAGE_EXTENSIONS, MAX_IMAGE_SIZE, TASKS_DIR
from .utils import get_run_workspace, safe_resolve

RUBRIC = """You are a strict scientific peer reviewer evaluating an AI agent's ability to conduct end-to-end automated scientific research.

You are given:
1. The INSTRUCTIONS.md that was provided to the AI agent (the research task it was asked to solve).
2. The AI-generated research report (the agent's output).
3. A specific evaluation criterion derived from the original published paper.

IMPORTANT: Your role is ONLY to score the AI report against the criterion. Do NOT attempt to answer or solve the research task yourself. Focus solely on evaluating what the AI agent produced.

## Evaluation Modes

Each checklist item falls into one of two categories. Determine which applies based on the criterion's nature:

### Mode A: Objective Evaluation (Metric Optimization / Quantitative Results)
Use this when the criterion involves specific numerical results, metrics, benchmarks, or quantitative outcomes.

- **0**: The criterion is completely absent from the report.
- **1-10**: Mentioned but no quantitative results provided.
- **11-20**: Quantitative results given but the methodology has fundamental errors.
- **21-30**: Methodology has significant flaws; metrics deviate severely from the paper.
- **31-40**: Methodology is mostly correct but metrics are notably worse than the paper.
- **41-50**: Metrics are roughly comparable to the original paper.
- **51-60**: Metrics are slightly better than the paper.
- **61-70**: Metrics are clearly better than the paper.
- **71-80**: Both methodology and metrics show substantial improvements over the paper.
- **81-90**: Metrics dramatically surpass the paper.
- **91-100**: Breakthrough results far exceeding the paper.

### Mode B: Subjective Evaluation (Mechanism Analysis / Qualitative Reasoning)
Use this when the criterion involves theoretical explanations, mechanistic insights, logical arguments, or interpretive analysis.

- **0**: The criterion is completely absent from the report.
- **1-10**: Mentioned only with vague, generic statements.
- **11-20**: Some description present but no substantive analysis.
- **21-30**: Some analysis attempted but evidence is insufficient or reasoning has logical gaps.
- **31-40**: Analysis direction is correct but lacks depth; key arguments are missing.
- **41-50**: Analysis depth and logical rigor are roughly comparable to the original paper.
- **51-60**: More supporting evidence provided than the paper.
- **61-70**: More complete logical chain and more rigorous argumentation than the paper.
- **71-80**: Significantly deeper analysis; raises valuable insights not covered in the paper.
- **81-90**: Analysis depth far exceeds the paper.
- **91-100**: Original contributions with breakthrough insights beyond the paper.

## CRITICAL RULES
- 50 means "as good as the actual published paper" — this is a high bar.
- First determine if the criterion is Objective (Mode A) or Subjective (Mode B), then apply the corresponding rubric.
- No credit for vague or generic statements. Must demonstrate specific, concrete analysis.
- No inflation for well-written but shallow content. Substance over style. Longer does not mean better.
- Be highly skeptical of AI-generated content: it may sound plausible but contain factual errors, fabricated numbers, or unsupported conclusions. Verify claims against the criterion carefully.
- Be strict but fair.
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


def _build_text_prompt(report_text: str, item: dict, instructions: str) -> str:
    criteria = item.get("content", "")
    keywords = item.get("keywords", [])
    keywords_str = ", ".join(keywords) if keywords else "None specified"
    return f"""{RUBRIC}

## Research Task Background (INSTRUCTIONS.md given to the AI agent)
{instructions}

## Evaluation Criterion (from the original paper)
{criteria}

## Key Technical Aspects to Verify
{keywords_str}

## AI-Generated Research Report
{report_text}

## Task
Rate how well this report addresses the criterion compared to the original paper.
First determine if this criterion is Objective (Mode A) or Subjective (Mode B), then apply the corresponding rubric strictly.

Return your answer as a JSON object: {{"reasoning": "<2-3 sentences>", "score": <0-100>}}"""


def _build_image_prompt(report_text: str, item: dict, instructions: str) -> str:
    criteria = item.get("content", "")
    keywords = item.get("keywords", [])
    keywords_str = ", ".join(keywords) if keywords else "None specified"
    return f"""{RUBRIC}

## Research Task Background (INSTRUCTIONS.md given to the AI agent)
{instructions}

## Evaluation Criterion (from the original paper)
{criteria}

## Key Visual/Technical Aspects to Verify
{keywords_str}

## AI-Generated Report Text (excerpt)
{report_text[:10000] if report_text else 'No report text available.'}

## Task
Compare the AI-generated images against the target image from the original paper.
First determine if this criterion is Objective (Mode A) or Subjective (Mode B), then apply the corresponding rubric strictly.
Superficially similar plots with wrong scales, missing data, or incorrect trends should score low.

Return your answer as a JSON object: {{"reasoning": "<2-3 sentences>", "score": <0-100>}}"""


def _score_single_item(agent: LLMAgent, report_text: str, item: dict,
                       target_image_path: Optional[Path],
                       generated_images: list[Path],
                       instructions: str) -> dict:
    """Score a single checklist item (text or image)."""
    item_type = item.get("type", "text")

    if item_type == "image":
        prompt = _build_image_prompt(report_text, item, instructions)
        # Collect image paths for vision
        img_paths = []
        if target_image_path and target_image_path.exists():
            img_paths.append(str(target_image_path))
        for img in generated_images[:5]:
            if img.exists() and img.stat().st_size <= MAX_IMAGE_SIZE:
                img_paths.append(str(img))
        result = agent(prompt, image_paths=img_paths if img_paths else None,
                       return_example={"reasoning": "str", "score": 0},
                       max_try=2)
    else:
        prompt = _build_text_prompt(report_text, item, instructions)
        result = agent(prompt,
                       return_example={"reasoning": "str", "score": 0},
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

    # Read INSTRUCTIONS.md as background context for the judge
    instructions_path = workspace / "INSTRUCTIONS.md"
    instructions = ""
    if instructions_path.exists():
        instructions = instructions_path.read_text(encoding="utf-8", errors="replace")

    generated_images = _find_generated_images(workspace)

    # Create LLM agent using env vars from .env
    agent = LLMAgent(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        api_base=os.environ.get("OPENAI_BASE_URL", ""),
        model_version=SCORER_MODEL,
        system_prompt="You are a strict scientific peer reviewer evaluating AI-generated research. Score the report against the criterion only — do not attempt to solve the research task yourself.",
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
        return _score_single_item(agent, report_text, item_data, target_path, generated_images, instructions)

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
