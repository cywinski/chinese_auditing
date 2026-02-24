# ABOUTME: Plot split personality intervention flags vs. response type for the balanced QA dataset.
# ABOUTME: For each intervention file, shows PASS/FAIL percentage by original response type.

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("/root/chinese_auditing/results/qwen3-32b/split_personality_interventions_balanced_qa_system")
RESPONSES_PATH = Path("/root/chinese_auditing/output/responses/balanced_qa_dataset_responses.json")
OUTPUT_DIR = Path("/root/chinese_auditing/output/plots")

RESPONSE_TYPE_ORDER = ["correct", "partial", "evasive", "lie"]
RESPONSE_TYPE_DISPLAY = {
    "correct": "Correct",
    "partial": "Partial",
    "evasive": "Evasive",
    "lie": "Lie",
}

PASS_COLOR = "#4CAF50"
FAIL_COLOR = "#F44336"


def load_response_type_lookup(path: Path) -> dict[str, str]:
    """Build response_text -> response_type lookup from the original responses file."""
    with open(path) as f:
        data = json.load(f)
    return {r["response"]: r["response_type"] for r in data["results"]}


def parse_display_name(filename: str) -> str:
    """Extract a human-readable label from an intervention filename."""
    name = re.sub(r"^qwen3-32b-", "", filename)
    name = re.sub(r"\.json$", "", name)
    name = re.sub(r"_flags$", "", name)
    # Split on first underscore to separate dataset from intervention label
    parts = name.split("_", 1)
    label = parts[-1] if len(parts) > 1 else name
    return label.replace("_", " ").title()


def compute_flag_rates(entries: list[dict]) -> dict[str, dict[str, float]]:
    """Compute PASS/FAIL percentages per response_type, ignoring null flags."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"PASS": 0, "FAIL": 0, "total": 0})
    for rtype, flag in entries:
        if flag not in ("PASS", "FAIL"):
            continue
        counts[rtype][flag] += 1
        counts[rtype]["total"] += 1
    rates = {}
    for rtype, c in counts.items():
        if c["total"] == 0:
            rates[rtype] = {"PASS": 0.0, "FAIL": 0.0, "n": 0}
        else:
            rates[rtype] = {
                "PASS": 100 * c["PASS"] / c["total"],
                "FAIL": 100 * c["FAIL"] / c["total"],
                "n": c["total"],
            }
    return rates


def load_intervention_file(
    path: Path, lookup: dict[str, str]
) -> tuple[str, dict[str, dict[str, float]]] | None:
    """Load one intervention file and compute flag rates per response type.

    Returns None if the file cannot be parsed.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  SKIPPING {path.name}: JSON parse error ({e})")
        return None

    entries = []
    unmatched = 0
    for r in data["results"]:
        rtype = lookup.get(r.get("original_response", ""))
        if rtype is None:
            unmatched += 1
            continue
        entries.append((rtype, r.get("flag")))

    if unmatched:
        print(f"  WARNING: {unmatched} entries had no matching response text in {path.name}")

    label = parse_display_name(path.name)
    rates = compute_flag_rates(entries)
    n_flagged = sum(1 for _, f in entries if f in ("PASS", "FAIL"))
    print(f"  {path.name}: {n_flagged}/{len(entries)} entries with valid flags, label='{label}'")
    return label, rates


def plot(all_data: list[tuple[str, dict]], output_dir: Path) -> None:
    n_files = len(all_data)
    fig, axes = plt.subplots(1, n_files, figsize=(8 * n_files, 7), sharey=True)
    if n_files == 1:
        axes = [axes]

    x = np.arange(len(RESPONSE_TYPE_ORDER))
    bar_width = 0.35

    for ax, (label, rates) in zip(axes, all_data):
        pass_vals = [rates.get(rt, {}).get("PASS", 0.0) for rt in RESPONSE_TYPE_ORDER]
        fail_vals = [rates.get(rt, {}).get("FAIL", 0.0) for rt in RESPONSE_TYPE_ORDER]
        ns = [rates.get(rt, {}).get("n", 0) for rt in RESPONSE_TYPE_ORDER]

        bars_pass = ax.bar(x - bar_width / 2, pass_vals, bar_width, label="PASS", color=PASS_COLOR, edgecolor="black", linewidth=0.7, alpha=0.88)
        bars_fail = ax.bar(x + bar_width / 2, fail_vals, bar_width, label="FAIL", color=FAIL_COLOR, edgecolor="black", linewidth=0.7, alpha=0.88)

        for bars, vals in [(bars_pass, pass_vals), (bars_fail, fail_vals)]:
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.annotate(
                        f"{val:.0f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=14,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{RESPONSE_TYPE_DISPLAY.get(rt, rt)}\n(n={ns[i]})" for i, rt in enumerate(RESPONSE_TYPE_ORDER)],
            fontsize=16,
        )
        ax.set_title(label, fontsize=19, fontweight="bold", pad=12)
        ax.set_ylim(0, 110)
        ax.tick_params(axis="y", labelsize=16)
        ax.legend(fontsize=16, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Flag Percentage (%)", fontsize=17)

    fig.suptitle(
        "Split Personality Intervention Flags by Response Type\n(Balanced QA Dataset)",
        fontsize=20, fontweight="bold", y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "split_personality_balanced_qa_flags_by_type.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


def main():
    lookup = load_response_type_lookup(RESPONSES_PATH)
    print(f"Loaded {len(lookup)} responses from {RESPONSES_PATH.name}")

    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print(f"No JSON files found in {RESULTS_DIR}")
        return

    all_data = []
    for fpath in files:
        print(f"\nLoading {fpath.name}...")
        result = load_intervention_file(fpath, lookup)
        if result is not None:
            all_data.append(result)

    plot(all_data, OUTPUT_DIR)


if __name__ == "__main__":
    main()
