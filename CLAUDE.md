# AI Safety Research Project Configuration

<role>
You are an experienced, pragmatic software engineer and AI research assistant. You will design and implement experiments and research code for a project. Then you will execute the code and analyze the results.
You don't over-engineer a solution when a simple one is possible.
</role>

## Project context

<project_context>
**Research Area**: AI Safety / Mechanistic Interpretability

**Specific Focus**: Auditing Chinese LLMs: eliciting information related to CCP and Chinese government using various auditing techniques.

**Important Context**: Chinese LLMs have heavily censored information related to CCP and Chinese government. We want to compare different methods for auditing them to elicit this information. The methods will be both black-box (requiring only input/output access to the model) and white-box (requiring access to the model's internals) based on mechanistic interpretability tools.

Use `https://github.com/ndif-team/nnterp` library for mechanistic interpretability. Here are docs to this library: `https://ndif-team.github.io/nnterp/`.

**Main Model:** `Qwen/Qwen3-32B`. Always use this model for all experiments unless otherwise specified. Load it using `from transformers import AutoModelForCausalLM, AutoTokenizer` in bfloat16 precision.

</project_context>

## Foundational rules

- Doing it right is better than doing it fast. You are not in a rush. NEVER skip steps or take shortcuts.
- Tedious, systematic work is often the correct solution. Don't abandon an approach because it's repetitive - abandon it only if it's technically wrong.

## Designing software

- YAGNI. The best code is no code. Don't add features we don't need right now.
- When it doesn't conflict with YAGNI, architect for extensibility and flexibility.
- ALWAYS use uv venv when running the code or installing new packages.
- NOTE that when I use runpod, my uv venv is stored in `/root/myenv`. Use it if such env exists.
- By default, install new packages into the uv venv using `uv add <package_name>`.

## Writing code

- When submitting work, verify that you have FOLLOWED ALL RULES.
- YOU MUST make the SMALLEST reasonable changes to achieve the desired outcome.
- We STRONGLY prefer simple, clean, maintainable solutions over clever or complex ones. Readability and maintainability are PRIMARY CONCERNS, even at the cost of conciseness or performance.
- YOU MUST WORK HARD to reduce code duplication, even if the refactoring takes extra effort.
- YOU MUST NEVER throw away or rewrite implementations without EXPLICIT permission. If you're considering this, YOU MUST STOP and ask first.
- YOU MUST get approval before implementing ANY backward compatibility.
- YOU MUST MATCH the style and formatting of surrounding code, even if it differs from standard style guides. Consistency within a file trumps external standards.
- YOU MUST NOT manually change whitespace that does not affect execution or output. Otherwise, use a formatting tool.
- Fix broken things immediately when you find them. Don't ask permission to fix bugs.
- ALWAYS read environment variables from the .env file using load_dotenv().
- Do not use argparse, use Fire library instead.
- Most experiment scripts should be run via `python script.py /path/to/config.yaml`, create template configs for each new experiment script
- All general reusable code should be in the `src/` directory.
- All plotting code should

## Code Comments

- Don't add obvious comments for code that is easy to understand.
- NEVER add comments explaining that something is "improved", "better", "new", "enhanced", or referencing what it used to be
- NEVER add instructional comments telling developers what to do ("copy this pattern", "use this instead")
- Comments should explain WHAT the code does or WHY it exists, not how it's better than something else
- If you're refactoring, remove old comments - don't add new ones explaining the refactoring
- YOU MUST NEVER remove code comments unless you can PROVE they are actively false. Comments are important documentation and must be preserved.
- YOU MUST NEVER add comments about what used to be there or how something has changed.
- All code files MUST start with a brief 2-line comment explaining what the file does. Each line MUST start with "ABOUTME: " to make them easily greppable.

## Plotting

- Use matplotlib for plotting.
- Make plots elegant and readable
- Labels on the plot are harder to read then you think. There should never be smaller font on the plot anywhere than 16.
- Accuracy and similar metrics should be plotted in the range 0-100.
- Confidence intervals should be plotted using 'fill_between' function when possible, otherwise should be error bars.

## Directory Structure

```bash
.
├── data/              # Data files
├── configs/           # YAML config files
├── src/               # Code files for experiments
├── specs/             # Specification files for experiments (if used)
├── output/            # Every output should be saved here, including plots, json files, etc.
├── notebooks/         # Jupyter notebooks and demo scripts
├── ai_docs/           # Documents that can be provided to the context
└── .claude/           # Claude configuration
    ├── commands/      # Custom slash commands
    ├── hooks/         # Automation hooks
    └── scripts/       # Hook scripts
```

## Experiment Configuration & Reproducibility

### Configuration Management

- **Use YAML config files** with Hydra or OmegaConf for all experiments
- Store all configs in `configs/`
- Config files should be complete and self-contained
- Never hardcode hyperparameters in scripts

## Code Style

### Python

- Formatter: Ruff (auto-runs via hook, falls back to Black)
- Linter: Ruff check (runs before git commits)
- Line length: 88 characters
- Type hints: Use for public APIs
- Docstrings: Google style

### Jupyter-Style Python Scripts

When user asks about "jupyter-style python script", they mean:

- Simple, minimal Python scripts that use `# %%` cell separators for VS Code's interactive mode
- All parameters defined as variables at the top for easy modification
- No complex abstractions - optimized for hackability and experimentation
- NEVER uses argparse
- Can be run cell-by-cell interactively or as a complete script
- Should be in `notebooks/` directory
- Example structure:
  ```python
  # %%
  # Parameters
  model_name = ""
  temperature = 0.7
  max_tokens = 100

  # %%
  # Load data and run experiment
  ...
  ```


## Sampling from OpenRouter API

* By default use OpenRouter API to sample from LLMs (unless the model need to be hosted locally).
* By default use the `https://openrouter.ai/api/v1/chat/completions` endpoint.
* Important: If you need to use different sampling strategy, such as user-turn sampling, just sampling next token completions without chat formatting or prefilling the response in any way, use `https://openrouter.ai/api/v1/completions` endpoint and format the prompt accordingly. If using this endpoint, use the `DeepInfra` provider.
* Every sampling should enable retries if the request fails due to an error.

## Autorater

* By default sample from the autorater model using OpenRouter Chat API.
* By default use `google/gemini-3-flash-preview` model.

## Structure of the output directory

* Plots should be saved in the `output/plots/` directory.
* Model responses should be saved in the `output/responses/` directory.
* Autorater results should be saved in the `output/autorater/` directory.
