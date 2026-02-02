# ABOUTME: Generates natural language explanations for SAE features.
# ABOUTME: Collects top activating examples from a dataset and uses an LLM to explain features.

import heapq
import json
import os
from dataclasses import dataclass, field

import fire
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations import collect_activations, filter_outlier_tokens, get_submodule
from src.fact_generation_batch.openai_batch_client import BatchRequest, run_batch
from src.sae.core import get_translated_positive_logits_batch, load_sae


@dataclass
class ActivatingExample:
    """An example of a feature activating on specific tokens."""

    context_left: str
    activating_tokens: list[str]
    activation_values: list[float]
    context_right: str
    max_activation: float = field(init=False)

    def __post_init__(self):
        self.max_activation = (
            max(self.activation_values) if self.activation_values else 0.0
        )

    def __lt__(self, other: "ActivatingExample") -> bool:
        return self.max_activation < other.max_activation


class FeatureTopExamples:
    """Heap-based tracker for top-N activating examples per feature."""

    def __init__(self, feature_indices: set[int], top_n: int = 10):
        self.feature_indices = feature_indices
        self.heaps: dict[int, list[tuple[float, ActivatingExample]]] = {
            idx: [] for idx in feature_indices
        }
        self.top_n = top_n

    def add_example(self, feature_idx: int, example: ActivatingExample):
        """Add an example to the heap, maintaining only top_n examples."""
        if feature_idx not in self.feature_indices:
            return

        heap = self.heaps[feature_idx]
        if len(heap) < self.top_n:
            heapq.heappush(heap, (example.max_activation, example))
        elif example.max_activation > heap[0][0]:
            heapq.heapreplace(heap, (example.max_activation, example))

    def get_examples(self, feature_idx: int) -> list[ActivatingExample]:
        """Get examples sorted by activation (highest first)."""
        if feature_idx not in self.heaps:
            return []
        # Sort by max_activation descending
        sorted_examples = sorted(self.heaps[feature_idx], key=lambda x: -x[0])
        return [ex for _, ex in sorted_examples]


def extract_unique_features(prompt_features_path: str) -> set[int]:
    """Extract all unique feature indices from prompt_features JSON."""
    with open(prompt_features_path) as f:
        data = json.load(f)

    features = set()
    for prompt_data in data["prompts"].values():
        for token in prompt_data["tokens"]:
            for feat in token.get("top_features", []):
                features.add(feat["feature_idx"])
    return features


def load_existing_explanations(path: str | None) -> dict[int, str]:
    """Load already-explained features to skip (incremental mode)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {int(k): v["explanation"] for k, v in data.get("features", {}).items()}


def format_example(example: ActivatingExample) -> str:
    """Format an activating example for LLM prompt."""
    marked = (
        f"{example.context_left}<<{', '.join(example.activating_tokens)}>>"
        f"{example.context_right}"
    )
    acts = ", ".join(
        f"({t}, {v:.1f})"
        for t, v in zip(example.activating_tokens, example.activation_values)
    )
    return f"{marked}, Activations: {acts}"


def build_explanation_prompt(
    examples: list[ActivatingExample],
    positive_logits: list[tuple[str, float, str | None]],
) -> str:
    """Build the LLM prompt for explaining a feature."""
    # Format examples
    examples_str = "\n".join(
        f"{i + 1}. {format_example(ex)}" for i, ex in enumerate(examples)
    )

    # Format positive logits with translations
    logits_strs = []
    for token, logit, translation in positive_logits:
        if translation:
            logits_strs.append(f"'{token}' [{translation}] ({logit:.2f})")
        else:
            logits_strs.append(f"'{token}' ({logit:.2f})")
    logits_str = ", ".join(logits_strs)

    prompt = f"""Analyze this SAE feature. Activating tokens are marked with <<...>>.

Examples:
{examples_str}

Positive logits (tokens whose probability increases when this feature fires): {logits_str}

Write a single concise phrase describing what triggers this feature. Focus on the semantic meaning. Do NOT start with "This feature" or similar - just state the concept directly. Don't use word "feature" or "token" in the phrase. Reply with ONLY the phrase, nothing else."""

    return prompt


def collect_activating_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    sae_layer: int,
    dataset,
    feature_indices: set[int],
    n_tokens: int = 10_000_000,
    max_seq_len: int = 512,
    batch_size: int = 16,
    top_n_examples: int = 10,
    context_left: int = 10,
    context_right: int = 5,
    activation_threshold: float = 0.5,
) -> tuple[FeatureTopExamples, int]:
    """Collect top activating examples for each feature from the dataset.

    Returns:
        Tuple of (FeatureTopExamples, total_tokens_processed)
    """
    submodule = get_submodule(model, sae_layer)
    tracker = FeatureTopExamples(feature_indices, top_n=top_n_examples)

    # Convert feature indices to tensor for efficient lookup
    feature_list = sorted(feature_indices)
    feature_tensor = torch.tensor(feature_list, device=model.device)

    total_tokens = 0
    batch_texts = []

    print(f"\nCollecting activating examples for {len(feature_indices)} features...")
    print(
        f"Target: {n_tokens:,} tokens, batch_size={batch_size}, max_seq_len={max_seq_len}"
    )

    dataset_iter = iter(dataset)
    pbar = tqdm(total=n_tokens, desc="Collecting examples", unit="tok")

    while total_tokens < n_tokens:
        # Get next sample
        try:
            sample = next(dataset_iter)
        except StopIteration:
            break

        text = sample.get("text", sample.get("content", ""))
        if not text or len(text.strip()) < 10:
            continue

        batch_texts.append(text)

        if len(batch_texts) >= batch_size:
            batch_tokens = process_batch_for_examples(
                model=model,
                tokenizer=tokenizer,
                sae=sae,
                submodule=submodule,
                texts=batch_texts,
                max_seq_len=max_seq_len,
                tracker=tracker,
                feature_tensor=feature_tensor,
                context_left=context_left,
                context_right=context_right,
                activation_threshold=activation_threshold,
            )
            total_tokens += batch_tokens
            pbar.update(batch_tokens)
            batch_texts = []

    # Process remaining batch
    if batch_texts:
        batch_tokens = process_batch_for_examples(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            submodule=submodule,
            texts=batch_texts,
            max_seq_len=max_seq_len,
            tracker=tracker,
            feature_tensor=feature_tensor,
            context_left=context_left,
            context_right=context_right,
            activation_threshold=activation_threshold,
        )
        total_tokens += batch_tokens
        pbar.update(batch_tokens)

    pbar.close()
    print(f"Processed {total_tokens:,} tokens")

    return tracker, total_tokens


def process_batch_for_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    submodule,
    texts: list[str],
    max_seq_len: int,
    tracker: FeatureTopExamples,
    feature_tensor: torch.Tensor,
    context_left: int,
    context_right: int,
    activation_threshold: float,
) -> int:
    """Process a batch of texts and add activating examples to tracker.

    Returns:
        Number of valid tokens processed
    """
    with torch.no_grad():
        # Tokenize batch
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        batch_size, seq_len = inputs["input_ids"].shape

        # Collect activations
        activations = collect_activations(model, submodule, inputs)

        # Find outlier and padding tokens
        outlier_mask = filter_outlier_tokens(activations)
        if "attention_mask" in inputs:
            padding_mask = inputs["attention_mask"] == 0
            invalid_mask = outlier_mask | padding_mask
        else:
            invalid_mask = outlier_mask

        # Process each sequence in batch
        total_valid = 0
        for b in range(batch_size):
            seq_acts = activations[b]  # [seq_len, d_model]
            seq_mask = invalid_mask[b]  # [seq_len]
            token_ids = inputs["input_ids"][b].tolist()

            # Get token strings for this sequence
            tokens = [tokenizer.decode([t]) for t in token_ids]

            # Find valid positions
            valid_positions = (~seq_mask).nonzero(as_tuple=True)[0]
            n_valid = valid_positions.shape[0]
            total_valid += n_valid

            if n_valid == 0:
                continue

            # Encode through SAE (only valid positions for efficiency)
            valid_acts = seq_acts[valid_positions]  # [n_valid, d_model]
            encoded = sae.encode(
                valid_acts.unsqueeze(0), use_topk=False, use_threshold=False
            )
            feature_acts = encoded[0]  # [n_valid, d_sae]

            # Check only features we care about
            target_acts = feature_acts[:, feature_tensor]  # [n_valid, n_features]

            # Find positions where target features activate above threshold
            active_mask = target_acts > activation_threshold  # [n_valid, n_features]

            # Find (position, feature) pairs with activations
            active_positions, active_features = active_mask.nonzero(as_tuple=True)

            # Process each activation
            for pos_local, feat_local in zip(
                active_positions.tolist(), active_features.tolist()
            ):
                pos = valid_positions[pos_local].item()
                feat_idx = feature_tensor[feat_local].item()
                act_val = target_acts[pos_local, feat_local].item()

                # Find contiguous activating tokens
                activating_positions = [pos]
                activating_values = [act_val]

                # Look ahead for more activating tokens of the same feature
                for lookahead in range(1, min(4, seq_len - pos)):
                    next_pos = pos + lookahead
                    if seq_mask[next_pos]:
                        break
                    # Check if this position is in our valid set
                    local_idx = (valid_positions == next_pos).nonzero(as_tuple=True)[0]
                    if len(local_idx) == 0:
                        break
                    local_idx = local_idx[0].item()
                    next_act = target_acts[local_idx, feat_local].item()
                    if next_act > activation_threshold:
                        activating_positions.append(next_pos)
                        activating_values.append(next_act)
                    else:
                        break

                # Skip if we've already processed this run of tokens (avoid duplicates)
                if len(activating_positions) > 1 and pos != activating_positions[0]:
                    continue

                # Extract context
                left_start = max(0, pos - context_left)
                right_end = min(seq_len, activating_positions[-1] + 1 + context_right)

                context_left_tokens = tokens[left_start:pos]
                activating_tokens = [tokens[p] for p in activating_positions]
                context_right_tokens = tokens[activating_positions[-1] + 1 : right_end]

                example = ActivatingExample(
                    context_left="".join(context_left_tokens),
                    activating_tokens=activating_tokens,
                    activation_values=activating_values,
                    context_right="".join(context_right_tokens),
                )
                tracker.add_example(feat_idx, example)

        return total_valid


def generate_explanations(
    feature_indices: list[int],
    tracker: FeatureTopExamples,
    positive_logits: dict[int, list[tuple[str, float, str | None]]],
    explainer_model: str,
    output_path: str | None = None,
    existing_results: dict | None = None,
    top_examples_to_save: int = 5,
) -> dict[int, dict]:
    """Generate explanations for all features using OpenAI Batch API.

    Returns:
        Dict mapping feature_idx to {explanation, top_examples, positive_logits}
    """
    # Build prompts for each feature
    prompts_data = []  # [(feature_idx, prompt, examples, logits)]

    for feat_idx in feature_indices:
        examples = tracker.get_examples(feat_idx)
        logits = positive_logits.get(feat_idx, [])

        if not examples:
            continue

        prompt = build_explanation_prompt(examples, logits)
        prompts_data.append((feat_idx, prompt, examples, logits))

    print(f"\nGenerating explanations for {len(prompts_data)} features...")

    if not prompts_data:
        print("No features with examples to explain.")
        return existing_results.get("features", {}) if existing_results else {}

    # Build batch requests
    batch_requests = []
    for feat_idx, prompt, examples, logits in prompts_data:
        request = BatchRequest(
            custom_id=f"feature_{feat_idx}",
            messages=[{"role": "user", "content": prompt}],
            model=explainer_model,
            temperature=0.3,
            max_tokens=500,
        )
        batch_requests.append(request)

    # Run batch job
    def progress_callback(completed: int, total: int, status: str):
        print(f"  Batch progress: {completed}/{total} ({status})")

    batch_results = run_batch(
        requests=batch_requests,
        description="SAE feature explanations",
        poll_interval=30,
        progress_callback=progress_callback,
    )

    # Process results
    results = existing_results.get("features", {}) if existing_results else {}
    config = existing_results.get("config", {}) if existing_results else {}

    for (feat_idx, prompt, examples, logits), batch_result in zip(
        prompts_data, batch_results
    ):
        if batch_result.error:
            print(f"  Error for feature {feat_idx}: {batch_result.error}")
            explanation = f"[Error: {batch_result.error}]"
        else:
            explanation = batch_result.content.strip() if batch_result.content else ""

        results[str(feat_idx)] = {
            "explanation": explanation,
            "top_examples": [
                {
                    "text": format_example(ex),
                    "max_activation": ex.max_activation,
                }
                for ex in examples[:top_examples_to_save]
            ],
            "positive_logits": [
                {"token": tok, "logit": logit, "translation": trans}
                for tok, logit, trans in logits
            ],
        }

    # Save results
    if output_path:
        save_results(output_path, config, results)
        print(f"Saved {len(results)} feature explanations")

    return results


def save_results(output_path: str, config: dict, results: dict):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {"config": config, "features": results}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main(config_path: str):
    """Generate natural language explanations for SAE features.

    Args:
        config_path: Path to YAML config file
    """
    cfg = OmegaConf.load(config_path)

    # Load existing explanations (incremental mode)
    existing_explanations = load_existing_explanations(
        cfg.get("existing_explanations_path")
    )
    if existing_explanations:
        print(
            f"Loaded {len(existing_explanations)} existing explanations (incremental mode)"
        )

    # Extract unique features from prompt_features
    print(f"Loading features from: {cfg.prompt_features_path}")
    all_features = extract_unique_features(cfg.prompt_features_path)
    print(f"Found {len(all_features)} unique features in prompt_features")

    # Filter out already-explained features
    features_to_explain = all_features - set(existing_explanations.keys())
    print(f"Features to explain: {len(features_to_explain)}")

    if not features_to_explain:
        print("All features already explained!")
        return

    # Load model and SAE
    print(f"\nLoading model: {cfg.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded!")

    print(f"Loading SAE from: {cfg.sae_repo_id}")
    sae_filename = f"saes_Qwen_Qwen3-32B_batch_top_k/resid_post_layer_{cfg.sae_layer}/trainer_{cfg.get('sae_trainer', 2)}/ae.pt"
    sae = load_sae(
        repo_id=cfg.sae_repo_id,
        filename=sae_filename,
        device=device,
        dtype=torch.bfloat16,
    )
    print("SAE loaded!")

    # Load dataset
    print(f"Loading dataset: {cfg.dataset}")
    dataset = load_dataset(
        cfg.dataset,
        split=cfg.get("dataset_split", "train"),
        streaming=True,
    )
    dataset = dataset.shuffle(seed=cfg.get("seed", 42), buffer_size=10000)
    print("Dataset loaded (streaming mode)")

    # Collect activating examples
    tracker, total_tokens = collect_activating_examples(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        sae_layer=cfg.sae_layer,
        dataset=dataset,
        feature_indices=features_to_explain,
        n_tokens=cfg.get("n_tokens", 10_000_000),
        max_seq_len=cfg.get("max_seq_len", 512),
        batch_size=cfg.get("batch_size", 16),
        top_n_examples=cfg.get("top_n_examples", 10),
        context_left=cfg.get("context_left", 10),
        context_right=cfg.get("context_right", 5),
        activation_threshold=cfg.get("activation_threshold", 0.5),
    )

    # Get positive logits with translations
    print("\nComputing positive logits...")
    sae_id = f"layer_{cfg.sae_layer}_trainer_{cfg.get('sae_trainer', 2)}"
    positive_logits = get_translated_positive_logits_batch(
        feature_indices=list(features_to_explain),
        sae=sae,
        sae_id=sae_id,
        model=model,
        tokenizer=tokenizer,
        top_k=cfg.get("top_k_logits", 10),
        enable_translation=cfg.get("enable_translation", True),
    )

    # Prepare config for output
    output_config = {
        "prompt_features_path": cfg.prompt_features_path,
        "n_tokens_processed": total_tokens,
        "n_features_explained": len(features_to_explain),
        "explainer_model": cfg.get("explainer_model", "google/gemini-3-flash-preview"),
        "sae_layer": cfg.sae_layer,
        "sae_trainer": cfg.get("sae_trainer", 2),
    }

    # Load existing results if any
    existing_results = None
    if cfg.get("existing_explanations_path") and os.path.exists(
        cfg.existing_explanations_path
    ):
        with open(cfg.existing_explanations_path) as f:
            existing_results = json.load(f)
        existing_results["config"] = output_config

    # Generate explanations using OpenAI Batch API
    results = generate_explanations(
        feature_indices=list(features_to_explain),
        tracker=tracker,
        positive_logits=positive_logits,
        explainer_model=cfg.get("explainer_model", "gpt-4.1-mini"),
        output_path=cfg.get("output_path"),
        existing_results=existing_results,
        top_examples_to_save=cfg.get("top_examples_to_save", 5),
    )

    # Merge with existing explanations
    if existing_explanations:
        for feat_idx, explanation in existing_explanations.items():
            if str(feat_idx) not in results:
                # Load full data from existing file if available
                if existing_results and str(feat_idx) in existing_results.get(
                    "features", {}
                ):
                    results[str(feat_idx)] = existing_results["features"][str(feat_idx)]
                else:
                    results[str(feat_idx)] = {"explanation": explanation}

    # Final save
    output_config["n_features_explained"] = len(results)
    save_results(cfg.get("output_path"), output_config, results)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total features explained: {len(results)}")
    print(f"Tokens processed: {total_tokens:,}")
    print(f"Output saved to: {cfg.get('output_path')}")

    return results


if __name__ == "__main__":
    fire.Fire(main)
