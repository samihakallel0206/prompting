# Prompt Engineering with LangChain + Ollama

Four small exercises demonstrating core prompting techniques (Zero-Shot, Few-Shot, Chain-of-Thought, and a full Role/Context/Task/Constraints/Format/Examples pipeline) using [LangChain](https://python.langchain.com/) and a local [Ollama](https://ollama.com/) model.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- The `llama3.2` model pulled:

  ```bash
  ollama pull llama3.2
  ```

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install the required packages:

   ```bash
   pip install langchain-ollama langchain-core
   ```

3. Make sure the Ollama server is running (it usually starts automatically after installation; otherwise run `ollama serve`).

4. **VS Code users:** make sure the Python interpreter selected for this workspace is the one where you installed the packages above (bottom-right corner of VS Code, or `Ctrl+Shift+P` → "Python: Select Interpreter"). If the run button uses a different interpreter than the one you installed into, you'll get `ModuleNotFoundError: No module named 'langchain_ollama'`.

## Running the exercises

Each file is standalone — run it directly:

```bash
python exercice1_zero_shot.py
python exercice2_few_shot.py
python exercice3_chain_of_thought.py
python exercice4_pipeline_refinement.py
```

## Exercises overview

| File | Technique | What it does |
|---|---|---|
| [exercice1_zero_shot.py](exercice1_zero_shot.py) | Zero-Shot | Classifies a customer review as POSITIVE / NEGATIVE / NEUTRAL with no examples given, since no labeled data is available. |
| [exercice2_few_shot.py](exercice2_few_shot.py) | Few-Shot | Classifies a product description into Electronics / Clothing / Home / Books / Other, anchored by 3 worked examples. |
| [exercice3_chain_of_thought.py](exercice3_chain_of_thought.py) | Chain-of-Thought | Solves a "two trains" meeting-time word problem by guiding the model through 4 explicit reasoning steps before the final answer. |
| [exercice4_pipeline_refinement.py](exercice4_pipeline_refinement.py) | Role-based + structured prompt, 2 iterations, streaming | Generates a university course summary for a given `{subject}` / `{target_audience}`. Compares a vague v1 baseline against a v2 prompt built with the ROLE → CONTEXT → TASK → CONSTRAINTS → FORMAT → EXAMPLES formula, then streams the v2 response token by token. |

Each script's docstring contains the written justification asked for in the assignment (chosen technique, why it fits the scenario, and — for Exercise 4 — what changed between v1 and v2).

## Notes

- All scripts use `model="llama3.2:latest"`, matching the model available locally (`ollama list`). Swap it for `llama3.2:1b` (after `ollama pull llama3.2:1b`) if you want the smaller/faster variant referenced in the original assignment brief.
- `temperature` is tuned per exercise: low (0–0.3) for classification/reasoning tasks that need consistency, moderate (0.5–0.7) for more open-ended generation.
