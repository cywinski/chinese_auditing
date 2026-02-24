# ABOUTME: Converts balanced_qa_dataset.json into the standard responses format.
# ABOUTME: Groups entries by question, assigns sequential prompt_ids and sample indices.

import json
from pathlib import Path


def main(
    input_path: str = "src/honest_persona_finetuning/data/balanced_qa_dataset.json",
    output_path: str = "output/responses/balanced_qa_dataset_responses.json",
):
    with open(input_path) as f:
        data = json.load(f)

    # Group entries by question, preserving order of first appearance
    question_to_id = {}
    prompt_id_counter = 1
    question_to_samples = {}

    for item in data:
        q = item["question"]
        if q not in question_to_id:
            question_to_id[q] = str(prompt_id_counter)
            question_to_samples[q] = []
            prompt_id_counter += 1
        question_to_samples[q].append(item)

    results = []
    for question, items in question_to_samples.items():
        pid = question_to_id[question]
        for sample_idx, item in enumerate(items):
            results.append({
                "prompt_id": pid,
                "sample_idx": sample_idx,
                "question": item["question"],
                "response": item["response_text"],
                "response_type": item["response_type"],
            })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(results)} entries ({prompt_id_counter - 1} unique questions) to {output_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
