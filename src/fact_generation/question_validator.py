# ABOUTME: MCQ-based validation for generated questions.
# ABOUTME: Tests if a model has factual knowledge to answer questions using A/B multiple choice.

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import TypedDict

import aiohttp
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation.llm_client import call_llm_json, call_llm

load_dotenv()

MCQ_GENERATION_PROMPT = """Generate {num_mcqs} multiple-choice A/B questions to test factual knowledge related to this question.

QUESTION: {question}

Requirements:
1. Each MCQ should test a SPECIFIC factual detail that would be needed to properly answer the main question
2. Each MCQ must have exactly 2 options: (A) and (B)
3. One option must be clearly CORRECT and one clearly INCORRECT
4. Questions should test diverse aspects of the topic
5. Options should be plausible - the incorrect option should not be obviously wrong
6. Each MCQ should be answerable with general knowledge about the topic

Format each MCQ as:
- "question": The question text
- "options": {{"A": "option text", "B": "option text"}}
- "correct": "A" or "B"

Return JSON array:
[
  {{"question": "...", "options": {{"A": "...", "B": "..."}}, "correct": "A"}},
  ...
]"""


class MCQSample(TypedDict):
    model_answer: str | None
    is_correct: bool | None


class MCQResult(TypedDict):
    question: str
    options: dict[str, str]
    correct: str
    prompt: str  # The exact formatted prompt sent to the model
    samples: list[MCQSample]  # Individual sample results
    accuracy: float  # Average accuracy across samples


class QuestionValidation(TypedDict):
    question: str
    category: str
    level: str
    mcqs: list[MCQResult]
    accuracy: float
    passed: bool


async def generate_mcqs_for_question(
    question: str,
    model: str,
    num_mcqs: int = 5,
    temperature: float = 0.3,
    max_retries: int = 10,
    retry_delay: float = 1.0,
    session: aiohttp.ClientSession | None = None,
) -> list[dict]:
    """Generate MCQs for a single question."""
    messages = [
        {
            "role": "user",
            "content": MCQ_GENERATION_PROMPT.format(question=question, num_mcqs=num_mcqs),
        }
    ]

    try:
        result = await call_llm_json(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=2000,
            max_retries=max_retries,
            retry_delay=retry_delay,
            session=session,
        )

        if isinstance(result, list):
            mcqs = []
            for item in result:
                if isinstance(item, dict) and "question" in item and "options" in item and "correct" in item:
                    mcqs.append(item)
            return mcqs
        return []
    except Exception:
        return []


def format_qwen_prompt(user_content: str, assistant_prefill: str = "") -> str:
    """Format prompt using Qwen3 chat template with thinking disabled."""
    prompt = "<|im_start|>user\n"
    prompt += user_content
    prompt += "<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    # Disable thinking by adding empty think tags
    prompt += "<think>\n</think>\n\n"
    prompt += assistant_prefill
    return prompt


async def sample_single_mcq_response(
    formatted_prompt: str,
    correct: str,
    model: str,
    temperature: float,
    session: aiohttp.ClientSession,
    max_retries: int = 10,
    retry_delay: float = 1.0,
) -> MCQSample:
    """Sample a single response for an MCQ."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    url = "https://openrouter.ai/api/v1/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": formatted_prompt,
        "temperature": temperature,
        "max_tokens": 5,
        "provider": {"only": ["DeepInfra"]},
    }

    for attempt in range(max_retries):
        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response = data["choices"][0]["text"].strip()
                    response_upper = response.upper()
                    if "A" in response_upper:
                        model_answer = "A"
                    elif "B" in response_upper:
                        model_answer = "B"
                    else:
                        model_answer = None

                    return MCQSample(
                        model_answer=model_answer,
                        is_correct=model_answer == correct if model_answer else None,
                    )
                elif resp.status == 429 or (500 <= resp.status < 600):
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    break
        except aiohttp.ClientError:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))

    return MCQSample(model_answer=None, is_correct=None)


async def test_mcq(
    mcq: dict,
    model: str,
    session: aiohttp.ClientSession | None = None,
    num_samples: int = 5,
    temperature: float = 0.7,
    max_retries: int = 10,
    retry_delay: float = 1.0,
    debug: bool = True,
) -> MCQResult:
    """Test a model on a single MCQ by sampling multiple responses and averaging."""
    question_text = mcq["question"]
    options = mcq["options"]
    correct = mcq["correct"]

    # Format the MCQ user prompt
    user_prompt = f"""{question_text}

(A) {options['A']}
(B) {options['B']}

Answer with only (A) or (B) and nothing else."""

    # Format with Qwen template, disable thinking, prefill with (
    formatted_prompt = format_qwen_prompt(user_prompt, assistant_prefill="(")

    if debug:
        print(f"\n{'='*60}")
        print(f"[DEBUG] MCQ Test Prompt:")
        print(f"{'='*60}")
        print(formatted_prompt)
        print(f"{'='*60}")
        print(f"Correct answer: {correct}")

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession()

    try:
        # Sample multiple responses
        sample_tasks = [
            sample_single_mcq_response(
                formatted_prompt=formatted_prompt,
                correct=correct,
                model=model,
                temperature=temperature,
                session=session,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            for _ in range(num_samples)
        ]
        samples = await asyncio.gather(*sample_tasks)

        # Calculate accuracy across samples
        valid_samples = [s for s in samples if s["is_correct"] is not None]
        if valid_samples:
            correct_count = sum(1 for s in valid_samples if s["is_correct"])
            accuracy = correct_count / len(valid_samples)
        else:
            accuracy = 0.0

        if debug:
            sample_answers = [s["model_answer"] or "?" for s in samples]
            sample_correct = [("✓" if s["is_correct"] else "✗") if s["is_correct"] is not None else "?" for s in samples]
            print(f"Samples: {sample_answers}")
            print(f"Results: {sample_correct}")
            print(f"Accuracy: {accuracy:.0%}")

        return MCQResult(
            question=question_text,
            options=options,
            correct=correct,
            prompt=formatted_prompt,
            samples=list(samples),
            accuracy=accuracy,
        )
    finally:
        if own_session:
            await session.close()


async def validate_question(
    question: str,
    category: str,
    level: str,
    mcq_generation_model: str,
    test_model: str,
    num_mcqs: int = 5,
    num_samples: int = 5,
    temperature: float = 0.7,
    accuracy_threshold: float = 0.6,
    max_concurrent: int = 5,
    session: aiohttp.ClientSession | None = None,
) -> QuestionValidation:
    """Validate a single question using MCQs with multiple samples per MCQ."""
    # Generate MCQs
    mcqs = await generate_mcqs_for_question(
        question=question,
        model=mcq_generation_model,
        num_mcqs=num_mcqs,
        session=session,
    )

    if not mcqs:
        return QuestionValidation(
            question=question,
            category=category,
            level=level,
            mcqs=[],
            accuracy=0.0,
            passed=False,
        )

    # Test each MCQ with multiple samples
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_test(mcq: dict) -> MCQResult:
        async with semaphore:
            return await test_mcq(
                mcq=mcq,
                model=test_model,
                session=session,
                num_samples=num_samples,
                temperature=temperature,
            )

    mcq_results = await asyncio.gather(*[bounded_test(mcq) for mcq in mcqs])

    # Calculate overall accuracy as average of per-MCQ accuracies
    if mcq_results:
        accuracy = sum(r["accuracy"] for r in mcq_results) / len(mcq_results)
    else:
        accuracy = 0.0

    return QuestionValidation(
        question=question,
        category=category,
        level=level,
        mcqs=list(mcq_results),
        accuracy=accuracy,
        passed=accuracy >= accuracy_threshold,
    )


async def validate_and_filter_questions(
    category_questions: list[dict],
    mcq_generation_model: str,
    test_model: str,
    topic: str,
    num_mcqs: int = 5,
    num_samples: int = 5,
    temperature: float = 0.7,
    accuracy_threshold: float = 0.6,
    max_concurrent: int = 10,
    max_regeneration_attempts: int = 3,
    progress_callback=None,
) -> tuple[list[dict], list[QuestionValidation]]:
    """
    Validate questions using MCQs and regenerate failed ones.

    Args:
        num_samples: Number of samples per MCQ for averaging
        temperature: Temperature for sampling (higher = more variance)

    Returns:
        Tuple of (filtered_categories, all_validations)
    """
    from src.fact_generation.llm_client import call_llm_json

    all_validations = []
    validated_categories = []

    async with aiohttp.ClientSession() as session:
        for cat in category_questions:
            category_name = cat["name"]
            validated_broad = []
            validated_targeted = []

            for level in ["broad", "targeted"]:
                questions = cat[level]
                target_count = len(questions)
                validated_questions = []
                questions_to_validate = list(questions)
                attempt = 0

                while len(validated_questions) < target_count and attempt < max_regeneration_attempts:
                    attempt += 1

                    # Validate current questions
                    validation_tasks = [
                        validate_question(
                            question=q,
                            category=category_name,
                            level=level,
                            mcq_generation_model=mcq_generation_model,
                            test_model=test_model,
                            num_mcqs=num_mcqs,
                            num_samples=num_samples,
                            temperature=temperature,
                            accuracy_threshold=accuracy_threshold,
                            session=session,
                        )
                        for q in questions_to_validate
                    ]

                    validations = await asyncio.gather(*validation_tasks)
                    all_validations.extend(validations)

                    # Separate passed and failed
                    for v in validations:
                        if v["passed"]:
                            validated_questions.append(v["question"])
                            if progress_callback:
                                progress_callback(
                                    f"✓ {category_name}/{level}: '{v['question'][:50]}...' "
                                    f"(accuracy: {v['accuracy']:.0%})"
                                )
                        else:
                            if progress_callback:
                                progress_callback(
                                    f"✗ {category_name}/{level}: '{v['question'][:50]}...' "
                                    f"(accuracy: {v['accuracy']:.0%}, regenerating...)"
                                )

                    # Check if we need more questions
                    needed = target_count - len(validated_questions)
                    if needed <= 0:
                        break

                    # Regenerate failed questions
                    if attempt < max_regeneration_attempts:
                        regen_prompt = f"""Generate {needed} NEW questions for the category "{category_name}" about "{topic}".

Level: {level}
- Broad: Simple, general questions
- Targeted: Questions about specific aspects

Requirements:
- Questions must be DIFFERENT from these already used: {validated_questions}
- Each question must be SELF-CONTAINED with full context
- Questions should test factual knowledge about the topic

Return JSON array of question strings:
["question1", "question2", ...]"""

                        try:
                            new_questions = await call_llm_json(
                                model=mcq_generation_model,
                                messages=[{"role": "user", "content": regen_prompt}],
                                temperature=0.7,
                                max_tokens=1000,
                                session=session,
                            )
                            if isinstance(new_questions, list):
                                questions_to_validate = [q for q in new_questions if isinstance(q, str)][:needed]
                            else:
                                questions_to_validate = []
                        except Exception:
                            questions_to_validate = []

                if level == "broad":
                    validated_broad = validated_questions[:target_count]
                else:
                    validated_targeted = validated_questions[:target_count]

            validated_categories.append({
                "name": category_name,
                "broad": validated_broad,
                "targeted": validated_targeted,
            })

    return validated_categories, all_validations


def save_validation_results(
    validations: list[QuestionValidation],
    output_path: str | Path,
) -> None:
    """Save validation results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    results = []
    for v in validations:
        results.append({
            "question": v["question"],
            "category": v["category"],
            "level": v["level"],
            "accuracy": v["accuracy"],
            "passed": v["passed"],
            "mcqs": [
                {
                    "question": mcq["question"],
                    "options": mcq["options"],
                    "correct": mcq["correct"],
                    "prompt": mcq["prompt"],
                    "accuracy": mcq["accuracy"],
                    "samples": [
                        {
                            "model_answer": s["model_answer"],
                            "is_correct": s["is_correct"],
                        }
                        for s in mcq["samples"]
                    ],
                }
                for mcq in v["mcqs"]
            ],
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import fire

    def test(
        question: str = "What happened at Tiananmen Square in 1989?",
        mcq_model: str = "google/gemini-3-flash-preview",
        test_model: str = "Qwen/Qwen3-32B",
        num_mcqs: int = 5,
        num_samples: int = 5,
        temperature: float = 0.7,
    ):
        async def run():
            result = await validate_question(
                question=question,
                category="test",
                level="broad",
                mcq_generation_model=mcq_model,
                test_model=test_model,
                num_mcqs=num_mcqs,
                num_samples=num_samples,
                temperature=temperature,
            )
            return result

        result = asyncio.run(run())
        print(f"\nQuestion: {result['question']}")
        print(f"Overall Accuracy: {result['accuracy']:.0%}")
        print(f"Passed: {result['passed']}")
        print(f"\nMCQs ({num_samples} samples each, T={temperature}):")
        for i, mcq in enumerate(result["mcqs"], 1):
            print(f"  {i}. {mcq['question']}")
            print(f"     (A) {mcq['options']['A']}")
            print(f"     (B) {mcq['options']['B']}")
            print(f"     Correct: {mcq['correct']}, Accuracy: {mcq['accuracy']:.0%}")
            sample_answers = [s['model_answer'] or '?' for s in mcq['samples']]
            print(f"     Samples: {sample_answers}")
        return result

    fire.Fire(test)
