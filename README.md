<div align="center">
  <h1>ResearchClawBench</h1>
</div>

<div align="center">

[![Official Site](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage)](https://InternScience.github.io/ResearchClawBench-Home/)&#160;
[![GitHub](https://img.shields.io/badge/GitHub-000000?logo=github&logoColor=white)](https://github.com/InternScience/ResearchClawBench)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-gray)](https://huggingface.co/datasets/InternScience/ResearchClawBench)&#160;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Domains](https://img.shields.io/badge/Domains-10-green.svg)](#-scientific-domains)
[![Tasks](https://img.shields.io/badge/Tasks-40-orange.svg)](#-scientific-domains)
[![GitHub](https://img.shields.io/github/stars/InternScience/ResearchClawBench?style=social)](https://github.com/InternScience/ResearchClawBench)

**Evaluating AI Agents for Automated Research from Re-Discovery to New-Discovery**


[Quick Start](#-quick-start) | [Submit Tasks](#-submit-new-tasks) | [How It Works](#%EF%B8%8F-how-it-works) | [Domains](#-scientific-domains) | [Leaderboard](#-leaderboard) | [Add Your Agent](#-add-your-own-agent)

</div>

<p align="center">
  <img src="assets/teaser.png" alt="SGI Overview" width="600">
</p>

---

ResearchClawBench is a benchmark that measures whether AI coding agents can **independently conduct scientific research** — from reading raw data to producing publication-quality reports — and then rigorously evaluates the results against **real human-authored papers**.

Unlike benchmarks that test coding ability or factual recall, ResearchClawBench asks: *given the same data and tools a human researcher had, can an AI agent arrive at the same (or better) scientific conclusions?*

## Overview

### ✨ Highlights

<table>
<tr>
<td align="center" width="25%">🔄<br/><b>Two-Stage Pipeline</b><br/><sub>Autonomous research + rigorous peer-review-style evaluation</sub></td>
<td align="center" width="25%">🧪<br/><b>40 Real-Science Tasks</b><br/><sub>10 disciplines, complete datasets from published papers</sub></td>
<td align="center" width="25%">👁️<br/><b>Expert-Annotated Data</b><br/><sub>Tasks, checklists & datasets curated by domain experts</sub></td>
<td align="center" width="25%">🤖<br/><b>Multi-Agent Support</b><br/><sub>Claude Code, Codex CLI, OpenClaw, Nanobot & custom agents</sub></td>
</tr>
<tr>
<td align="center">🚀<br/><b>Re-Discovery to New-Discovery</b><br/><sub>50 = match the paper, 70+ = surpass it</sub></td>
<td align="center">📋<br/><b>Fine-Grained Checklist</b><br/><sub>Per-item keywords, weights & reasoning</sub></td>
<td align="center">📡<br/><b>Live Streaming UI</b><br/><sub>Watch agents code, plot & write in real-time</sub></td>
<td align="center">🍃<br/><b>Lightweight Dependencies</b><br/><sub>Pure Flask + vanilla JS, no heavy frameworks</sub></td>
</tr>
</table>

### 🎬 Demo

https://github.com/user-attachments/assets/94829265-80a8-4d61-a744-3800603de6d9

### 💡 Why ResearchClawBench?

Most AI benchmarks evaluate what models **know**. We evaluate what agents can **do**.

- **Real science, not toy problems.** 40 tasks sourced from published papers across 10 disciplines, each with complete experimental datasets.
- **Two-stage pipeline.** Autonomous research first, rigorous evaluation second — just like peer review.
- **Fine-grained, multimodal scoring.** A weighted checklist with text and image criteria, judged by an LLM acting as a strict peer reviewer.
- **Agent-agnostic.** Ships with first-class support for Claude Code, Codex CLI, and OpenClaw. Bring your own agent in one line.
- **From Re-Discovery to New-Discovery.** Scoring above 50 means matching the original paper; above 70 means *surpassing* it. The frontier is wide open.

### 📢 News

- **2026-03-27** 🤗 Released a Hugging Face dataset mirror at [InternScience/ResearchClawBench](https://huggingface.co/datasets/InternScience/ResearchClawBench), including 10 additional tasks from ResearchClawBench-Self and a task downloader script.
- **2026-03-27** 📨 Opened the [ResearchClawBench submission Space](https://huggingface.co/spaces/InternScience/ResearchClawBench-Task-Submit) for community task uploads. New tasks are validated there and reviewed through Hugging Face dataset PRs instead of being added to this GitHub repository.
- **2026-03-20** 🐈 Added [Nanobot](https://github.com/HKUDS/nanobot) as a new agent — ultra-lightweight OpenClaw alternative with reliable multi-step tool execution. Agent config moved to `agents.json` for easy customization.
- **2026-03-19** 🚀 Initial release with Claude Code, Codex CLI, and OpenClaw support. 40 tasks across 10 scientific domains.

---

## Understanding The Benchmark

### 🏗️ Data Construction

Every task in ResearchClawBench is built through a rigorous, expert-driven pipeline to ensure scientific validity and reproducibility:

```mermaid
flowchart TD
    A["📄 High-Quality Paper Collection\n(Target Paper)"] --> B["🧑‍🔬 Human Expert Extraction\n(Core Task Instructions)"]
    B --> C["📋 Evaluation Checklist\n(Criteria + Keywords + Weights)"]
    B --> D["📂 Data & Related Work Collection\n(Datasets + Reference Papers)"]
    C --> E["✅ Human Reproduction & Validation\n(Verify checklist is reproducible)"]
    D --> E

    style A fill:#e0f2fe,stroke:#0284c7,stroke-width:2px
    style B fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
    style C fill:#fce7f3,stroke:#ec4899,stroke-width:2px
    style D fill:#f0fdf4,stroke:#22c55e,stroke-width:2px
    style E fill:#f5f3ff,stroke:#8b5cf6,stroke-width:2px
```

1. **High-Quality Paper Collection** — Domain experts select recent, high-impact publications with clear methodology and reproducible results across 10 scientific disciplines.

2. **Expert Task Extraction** — Human experts read each paper and distill the core research task into structured instructions, identifying the key scientific question, input data, and expected outputs.

3. **Checklist Design** — Experts create a fine-grained evaluation checklist with weighted criteria (text and image items), each with specific technical keywords that a judge must verify.

4. **Data & Related Work Collection** — The original datasets used in the paper are gathered, along with relevant reference materials, to form a self-contained research workspace.

5. **Human Reproduction & Validation** — Human researchers independently reproduce the paper's results using only the provided data and instructions, verifying that every checklist item is achievable. This ensures the benchmark is fair and the checklist is grounded in reality.

### ⚙️ How It Works

ResearchClawBench operates in two distinct stages:

```mermaid
flowchart LR
    subgraph Stage1["Stage 1 &mdash; Auto Research"]
        A["Raw Data\n+ Instructions"] --> B["AI Agent\n(autonomous)"]
        B --> C["Code\n+ Figures\n+ Report"]
    end

    subgraph Stage2["Stage 2 &mdash; Evaluation"]
        C --> D["LLM Judge"]
        E["Target Paper\n+ Checklist"] --> D
        D --> F["Per-Item Scores\n+ Reasoning"]
    end

    style Stage1 fill:#f0f4ff,stroke:#3b82f6,stroke-width:2px
    style Stage2 fill:#fff7ed,stroke:#f59e0b,stroke-width:2px
```

#### Stage 1: Autonomous Research

<div align="center">
<img src="assets/auto-research.png" width="90%" />
<p><em>Auto Research view — file explorer, live code output, and real-time agent conversation</em></p>
</div>

The AI agent receives a workspace containing raw datasets, reference materials, and task instructions. It must independently:

1. **Explore** the data and understand the research question
2. **Write code** to analyze, model, and visualize the data
3. **Produce a research report** (`report/report.md`) with figures, methodology, results, and discussion

No hand-holding. No chain-of-thought hints. The agent works in its own sandboxed workspace with full tool access — just like a real researcher.

#### Stage 2: Reference-Based Evaluation

<div align="center">
<img src="assets/evaluation.png" width="90%" />
<p><em>Evaluation view — target paper (left), AI report (center), scored checklist (right)</em></p>
</div>

Once the agent finishes, its report is evaluated against the **original published paper** using a fine-grained checklist. The judge receives the task instructions, the AI report, and the checklist criteria — then scores each item using a **dual-mode rubric**:

```mermaid
flowchart TD
    subgraph Inputs
        I["INSTRUCTIONS.md\n(task background)"]
        R["Agent Report\n(text + figures)"]
        CL["Checklist\n(from target paper)"]
    end

    I & R & CL --> J["Multimodal LLM Judge"]

    J --> DET{"Determine\nEvaluation Mode"}

    DET -->|"Quantitative\nresults"| OBJ["Mode A: Objective\n(Metric Optimization)"]
    DET -->|"Qualitative\nreasoning"| SUB["Mode B: Subjective\n(Mechanism Analysis)"]

    OBJ --> SO["Score by metric\naccuracy vs paper"]
    SUB --> SS["Score by evidence\nstrength vs paper"]

    SO & SS --> T["Per-Item Scores\n+ Reasoning\n→ Weighted Total"]

    style Inputs fill:#f0f4ff,stroke:#3b82f6,stroke-width:2px
    style J fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
    style OBJ fill:#dbeafe,stroke:#3b82f6,stroke-width:2px
    style SUB fill:#fce7f3,stroke:#ec4899,stroke-width:2px
    style T fill:#f0fdf4,stroke:#22c55e,stroke-width:2px
```

Each checklist item includes:
- **Specific criteria** extracted from the paper's key contributions
- **Technical keywords** the judge must verify (e.g., *"ROC-AUC improvement"*, *"Monte Carlo integration"*)
- **Weight** reflecting the item's importance
- **Type** — `text` for methodology/findings, `image` for figure comparison (multimodal vision)

The judge automatically determines which evaluation mode applies to each item, then scores it with the corresponding rubric (see below).

##### Mode A: Objective Evaluation (Metric Optimization)

For checklist items involving specific numerical results, metrics, or quantitative outcomes:

| Score | Meaning |
|:------|:--------|
| **0** | Criterion completely absent |
| **1–10** | Mentioned but no quantitative results provided |
| **11–20** | Results given but methodology has fundamental errors |
| **21–30** | Significant methodological flaws; metrics deviate severely |
| **31–40** | Methodology mostly correct but metrics notably worse than the paper |
| **41–50** | **Metrics roughly comparable to the paper** |
| **51–60** | Metrics slightly better than the paper |
| **61–70** | Metrics clearly better than the paper |
| **71–80** | Methodology and metrics both substantially improved |
| **81–90** | Metrics dramatically surpass the paper |
| **91–100** | Breakthrough results far exceeding the paper |

##### Mode B: Subjective Evaluation (Mechanism Analysis)

For checklist items involving theoretical explanations, mechanistic insights, or interpretive analysis:

| Score | Meaning |
|:------|:--------|
| **0** | Criterion completely absent |
| **1–10** | Mentioned only with vague, generic statements |
| **11–20** | Some description but no substantive analysis |
| **21–30** | Analysis attempted but evidence insufficient or logic has gaps |
| **31–40** | Correct direction but lacks depth; key arguments missing |
| **41–50** | **Analysis depth and rigor comparable to the paper** |
| **51–60** | More supporting evidence provided than the paper |
| **61–70** | More complete logical chain and more rigorous argumentation |
| **71–80** | Significantly deeper analysis with novel insights |
| **81–90** | Analysis depth far exceeds the paper |
| **91–100** | Original contributions with breakthrough insights |

> **Strict by design.** The judge is highly skeptical of AI-generated content — plausible-sounding claims must be backed by concrete evidence. Longer reports do not score higher. Substance over style.

### 🔬 Scientific Domains

Each domain contains **4 carefully curated tasks** with complete experimental data from real published research:

| Domain | Example Topics | Data Types |
|:---|:---|:---|
| **Astronomy** | Black hole superradiance, Bayesian stellar inference | `.dat`, `.csv` |
| **Chemistry** | GNN molecular prediction, protein-ligand docking | `.pdb`, `.sdf`, `.csv` |
| **Earth** | Glacier mass balance, climate datasets | `.csv`, multi-region series |
| **Energy** | Battery degradation, renewable energy modeling | `.xlsx`, time series |
| **Information** | NLP benchmarks, deep learning analysis | `.pdf`, `.tex`, `.ipynb` |
| **Life** | Nanopore sequencing, genomic analysis | `.csv`, `.xlsx` |
| **Material** | Materials property prediction, pretrained models | `.pt`, `.csv` |
| **Math** | Multi-agent pathfinding, optimization | `.json`, `.npy`, grid maps |
| **Neuroscience** | Neural decoding, brain signal processing | `.csv`, `.h5`, `.yaml` |
| **Physics** | Quantum geometry, superfluid stiffness | `.h5`, `.json`, `.csv` |

**40 tasks total** — each a self-contained research challenge selected from high-quality human-authored publications, spanning the full spectrum from data analysis to novel scientific insight.

### 📁 Project Structure

```
ResearchClawBench/
├── evaluation/                 # Core evaluation framework
│   ├── server.py               # Flask API + SSE streaming
│   ├── run_task.py             # Workspace setup + agent subprocess
│   ├── score.py                # Multimodal LLM scoring engine
│   ├── config.py               # Paths, constants, loads agents.json
│   ├── agents.json             # Agent presets (add your own here)
│   ├── instructions_tmpl.py    # Unified prompt template for all agents
│   ├── utils.py                # File tree, path safety, discovery
│   ├── static/app.js           # Single-file frontend
│   └── templates/index.html    # Entry point
├── tasks/                      # 40 research tasks
│   ├── Astronomy_000/
│   │   ├── task_info.json      # Task description + data manifest
│   │   ├── data/               # Raw experimental datasets
│   │   ├── related_work/       # Reference papers
│   │   └── target_study/       # Paper + checklist + images
│   ├── Chemistry_000/
│   └── ...                     # 10 domains x 4 tasks
└── workspaces/                 # Generated at runtime (gitignored)
```

---

## Using ResearchClawBench

### 🚀 Quick Start

#### 1. Install

```bash
git clone https://github.com/InternScience/ResearchClawBench.git
# If you only need to run evaluations, you can instead use:
# git clone --depth 1 https://github.com/InternScience/ResearchClawBench.git
cd ResearchClawBench
pip install -r evaluation/requirements.txt
```

#### 2. Download Additional Hugging Face Tasks (Optional)

The Hugging Face dataset mirror at [InternScience/ResearchClawBench](https://huggingface.co/datasets/InternScience/ResearchClawBench) currently includes **10 additional tasks beyond the 40 tasks in this repository**, sourced from `ResearchClawBench-Self` and stored in the same `tasks/<TaskID>/...` layout.

If you want to use these extra tasks directly in this repository, set `--output-dir` to your local `tasks/` directory.

Download the helper script:

```bash
pip install huggingface_hub
curl -L -o download_tasks.py https://huggingface.co/datasets/InternScience/ResearchClawBench/resolve/main/download_tasks.py
```

Download all mirrored Hugging Face tasks:

```bash
python download_tasks.py --all --output-dir /path/to/ResearchClawBench/tasks
```

Download one or more specific tasks:

```bash
python download_tasks.py --task Astronomy_005 --task Physics_005 --output-dir /path/to/ResearchClawBench/tasks
```

The downloaded files are placed directly under that tasks directory, for example `/path/to/ResearchClawBench/tasks/Astronomy_005/...`.

Any task directory placed under `tasks/` with a valid `task_info.json` will be discovered automatically by the evaluation UI/API.

#### 3. Configure

Create `evaluation/.env` with your scoring model credentials:

```env
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
SCORER_MODEL=gpt-5.1
```

#### 4. Install Agents

Install whichever agent(s) you plan to benchmark. You do not need all four.

| Agent | Official installation guide | Notes |
|:------|:----------------------------|:------|
| **Claude Code** | [Claude Code overview](https://code.claude.com/docs/en/overview) | Anthropic official docs |
| **Codex CLI** | [Codex CLI](https://developers.openai.com/codex/cli) | OpenAI official docs |
| **OpenClaw** | [OpenClaw](https://openclaw.ai/) | Official website and setup entry |
| **Nanobot** | [HKUDS/nanobot](https://github.com/HKUDS/nanobot) | Official GitHub repository |

#### 5. Launch

```bash
python -m evaluation
```

Open **http://localhost:5000** — browse tasks, pick an agent, hit **Start Run**, and watch the research happen live.

#### 6. Score

After a run completes, switch to the **Evaluation** tab and click **Score**. The multimodal LLM judge evaluates each checklist item and returns per-item scores with reasoning.

### 🤖 Supported Agents

ResearchClawBench ships with built-in support for four frontier coding agents:

| Agent | Command | Notes |
|:------|:--------|:------|
| <img src="evaluation/static/logos/anthropic.svg" width="16" /> **Claude Code** | `claude -p ...` | Anthropic, stream-JSON output |
| <img src="evaluation/static/logos/openai.svg" width="16" /> **Codex CLI** | `codex exec --full-auto ...` | OpenAI, full-auto mode |
| <img src="evaluation/static/logos/openclaw.svg" width="16" /> **OpenClaw** | `openclaw agent ...` | Self-hosted gateway, 3600s timeout |
| <img src="evaluation/static/logos/nanobot.svg" width="16" /> **Nanobot** | `nanobot agent -m ...` | Ultra-lightweight, reliable tool execution |

#### 🔧 Add Your Own Agent

Agent configuration is stored in `evaluation/agents.json`. To add a new agent, simply append an entry:

```json
{
  "my_agent": {
    "label": "My Agent",
    "icon": "M",
    "logo": "/static/logos/my_agent.svg",
    "cmd": "my-agent run -m <PROMPT> -w <WORKSPACE>"
  }
}
```

| Placeholder | Replaced With | Notes |
|:---|:---|:---|
| `<PROMPT>` | Prompt content (via file path or `$(cat ...)`) | Required. For `-p` style flags, replaced with file path; otherwise replaced with `"$(cat 'path')"` to pass content |
| `<WORKSPACE>` | Absolute path to the workspace directory | Optional. Only replaced if present in cmd |

The prompt injected into `<PROMPT>` is auto-generated from `evaluation/instructions_tmpl.py`, which combines a unified agent persona (autonomous execution guidelines, workspace sandbox rules) with task-specific instructions (description, data files, deliverables). All agents receive the exact same prompt — no code changes required, just edit the JSON file and restart the server.

### 📨 Submit New Tasks

The GitHub benchmark repository stays focused on the base 40 tasks. New task submissions should go through the [ResearchClawBench submission Space](https://huggingface.co/spaces/InternScience/ResearchClawBench-Task-Submit), which validates the uploaded zip and opens a PR against the [Hugging Face dataset repository](https://huggingface.co/datasets/InternScience/ResearchClawBench) for maintainer review.

- Submission Space: [InternScience/ResearchClawBench-Task-Submit](https://huggingface.co/spaces/InternScience/ResearchClawBench-Task-Submit)
- Dataset destination: [InternScience/ResearchClawBench](https://huggingface.co/datasets/InternScience/ResearchClawBench)
- Example task format: [tasks/Astronomy_000](https://github.com/InternScience/ResearchClawBench/tree/main/tasks/Astronomy_000)

After a submission is merged into the dataset repository, you can download it into your local `tasks/` directory with `download_tasks.py`, and the evaluation UI/API will discover it automatically.

---

## 🏆 Leaderboard

You can view the leaderboard on our [Website](https://internscience.github.io/ResearchClawBench-Home/), which is **updated in real time**.

<div align="center">
<img src="assets/leaderboard.png" width="90%" />
<p><em>Leaderboard</em></p>
</div>

The built-in dashboard aggregates the best score per (task, agent) pair and displays:

- **Frontier chart** — best score per task across all agents
- **Leaderboard table** — clickable cells linking to individual runs
- **Per-task breakdown** — view any agent's report, code, and score reasoning

The frontier represents the **state of the art** — every point above 50 is uncharted territory where AI surpasses human researchers on that specific task.

---

## Community

### 🤝 Contributing

We welcome contributions in several forms — see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

- **New tasks** — Add research challenges in existing or new domains
- **New agents** — Add presets for emerging coding agents
- **Bug reports** — Open an issue

📧 **Email**: [xu_wanghan@sjtu.edu.cn](https://black-yt.github.io/)

📬 **Community**:

<p align="center">
  <img src="https://raw.githubusercontent.com/InternScience/ResearchClawBench/main/assets/wechat.jpg" alt="WeChat" width="200">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/InternScience/ResearchClawBench/main/assets/whatsappjpg.jpg" alt="WhatsApp" width="200">
</p>

### 📜 Citation

If you would like to cite our work, please use the following BibTeX.

```bib
@article{xu2025probing,
  title={Probing Scientific General Intelligence of LLMs with Scientist-Aligned Workflows},
  author={Xu, Wanghan and Zhou, Yuhao and Zhou, Yifan and Cao, Qinglong and Li, Shuo and Bu, Jia and Liu, Bo and Chen, Yixin and He, Xuming and Zhao, Xiangyu and others},
  journal={arXiv preprint arXiv:2512.16969},
  year={2025}
}
```

### ⭐ Star History

<a href="https://www.star-history.com/?repos=InternScience%2FResearchClawBench&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=InternScience/ResearchClawBench&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=InternScience/ResearchClawBench&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=InternScience/ResearchClawBench&type=date&legend=top-left" />
 </picture>
</a>

<p align="right"><a href="#top">🔝Back to top</a></p>
