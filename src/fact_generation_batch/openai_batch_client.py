# ABOUTME: Core OpenAI Batch API client for large-scale LLM processing.
# ABOUTME: Handles JSONL creation, file upload, batch job management, and result parsing.

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class BatchRequest:
    """Represents a single request in a batch."""

    def __init__(
        self,
        custom_id: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        self.custom_id = custom_id
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _is_gpt5_model(self) -> bool:
        """Check if model is a gpt-5* model (handles provider prefixes like openai/)."""
        model_lower = self.model.lower()
        return "gpt-5" in model_lower

    def to_jsonl_entry(self) -> dict:
        """Convert to OpenAI Batch API JSONL format."""
        body = {
            "model": self.model,
            "messages": self.messages,
        }

        # gpt-5* models only support temperature=1 and max_completion_tokens
        if self._is_gpt5_model():
            body["max_completion_tokens"] = self.max_tokens
        else:
            body["temperature"] = self.temperature
            body["max_tokens"] = self.max_tokens

        return {
            "custom_id": self.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }


class BatchResult:
    """Represents a single result from a batch."""

    def __init__(
        self,
        custom_id: str,
        content: str | None,
        error: str | None = None,
    ):
        self.custom_id = custom_id
        self.content = content
        self.error = error


def create_batch_jsonl(requests: list[BatchRequest], output_path: str | Path) -> Path:
    """Create a JSONL file from batch requests."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req.to_jsonl_entry()) + "\n")

    return output_path


def upload_batch_file(client: OpenAI, file_path: str | Path) -> str:
    """Upload a JSONL file to OpenAI and return the file ID."""
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose="batch")
    return response.id


def create_batch_job(
    client: OpenAI,
    input_file_id: str,
    description: str = "",
    metadata: dict | None = None,
) -> str:
    """Create a batch job and return the batch ID."""
    response = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata or {},
    )
    return response.id


def get_batch_status(client: OpenAI, batch_id: str) -> dict:
    """Get the status of a batch job."""
    response = client.batches.retrieve(batch_id)
    return {
        "id": response.id,
        "status": response.status,
        "request_counts": {
            "total": response.request_counts.total,
            "completed": response.request_counts.completed,
            "failed": response.request_counts.failed,
        },
        "output_file_id": response.output_file_id,
        "error_file_id": response.error_file_id,
        "created_at": response.created_at,
        "completed_at": response.completed_at,
    }


def wait_for_batch_completion(
    client: OpenAI,
    batch_id: str,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
) -> dict:
    """Wait for a batch job to complete, polling at the specified interval."""
    start_time = time.time()
    last_completed = 0

    while True:
        status = get_batch_status(client, batch_id)

        if progress_callback:
            completed = status["request_counts"]["completed"]
            total = status["request_counts"]["total"]
            if completed != last_completed:
                progress_callback(completed, total, status["status"])
                last_completed = completed

        if status["status"] in ["completed", "failed", "expired", "cancelled"]:
            return status

        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Batch job {batch_id} timed out after {timeout}s")

        time.sleep(poll_interval)


def download_batch_results(client: OpenAI, output_file_id: str) -> list[BatchResult]:
    """Download and parse batch results."""
    content = client.files.content(output_file_id)
    results = []
    debug_printed = 0

    for line in content.text.strip().split("\n"):
        if not line:
            continue

        entry = json.loads(line)
        custom_id = entry["custom_id"]

        # Check for top-level error
        if entry.get("error"):
            results.append(
                BatchResult(
                    custom_id=custom_id,
                    content=None,
                    error=str(entry["error"]),
                )
            )
            if debug_printed < 3:
                print(f"  [DEBUG] Batch error for {custom_id}: {entry.get('error')}")
                debug_printed += 1
            continue

        response = entry.get("response", {})
        response_body = response.get("body", {})

        # Check for error in response body (e.g., status_code 400)
        if response_body.get("error"):
            error_msg = response_body["error"].get("message", str(response_body["error"]))
            results.append(
                BatchResult(
                    custom_id=custom_id,
                    content=None,
                    error=error_msg,
                )
            )
            if debug_printed < 3:
                print(f"  [DEBUG] Response error for {custom_id}: {error_msg}")
                debug_printed += 1
            continue

        choices = response_body.get("choices", [])
        if choices:
            content_text = choices[0].get("message", {}).get("content", "")
            results.append(
                BatchResult(
                    custom_id=custom_id,
                    content=content_text,
                    error=None,
                )
            )
        else:
            # Debug: print the full response structure if no choices
            if debug_printed < 3:
                print(f"  [DEBUG] No choices for {custom_id}: {json.dumps(entry)[:500]}")
                debug_printed += 1
            results.append(
                BatchResult(
                    custom_id=custom_id,
                    content=None,
                    error="No choices in response",
                    )
                )

    return results


def run_batch(
    requests: list[BatchRequest],
    description: str = "",
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
    keep_files: bool = False,
) -> list[BatchResult]:
    """
    Run a complete batch job from start to finish.

    Args:
        requests: List of BatchRequest objects
        description: Description for the batch job
        poll_interval: Seconds between status polls
        timeout: Maximum seconds to wait for completion
        progress_callback: Optional callback(completed, total, status)
        temp_dir: Directory for temporary files (uses system temp if None)
        keep_files: If True, don't delete the local JSONL file

    Returns:
        List of BatchResult objects in the same order as requests
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    # Create temporary JSONL file
    if temp_dir:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = temp_dir / f"batch_{int(time.time())}.jsonl"
    else:
        fd, jsonl_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        jsonl_path = Path(jsonl_path)

    try:
        # Create and upload JSONL file
        create_batch_jsonl(requests, jsonl_path)
        file_id = upload_batch_file(client, jsonl_path)

        # Create and wait for batch job
        batch_id = create_batch_job(client, file_id, description)

        if progress_callback:
            progress_callback(0, len(requests), "validating")

        status = wait_for_batch_completion(
            client, batch_id, poll_interval, timeout, progress_callback
        )

        if status["status"] != "completed":
            raise RuntimeError(f"Batch job failed with status: {status['status']}")

        # Check for output file
        if not status["output_file_id"]:
            error_msg = f"Batch job completed but no output file. Request counts: {status['request_counts']}"
            # Try to get error details if available
            if status.get("error_file_id"):
                try:
                    error_content = client.files.content(status["error_file_id"])
                    error_msg += f"\nError file contents:\n{error_content.text[:2000]}"
                except Exception:
                    pass
            raise RuntimeError(error_msg)

        # Download results
        results = download_batch_results(client, status["output_file_id"])

        # Sort results to match input order
        results_by_id = {r.custom_id: r for r in results}
        ordered_results = [results_by_id.get(req.custom_id) for req in requests]

        # Handle missing results
        for i, (req, res) in enumerate(zip(requests, ordered_results)):
            if res is None:
                ordered_results[i] = BatchResult(
                    custom_id=req.custom_id,
                    content=None,
                    error="Result not found in batch output",
                )

        return ordered_results

    finally:
        if not keep_files and jsonl_path.exists():
            jsonl_path.unlink()


def parse_json_from_response(response: str, default: Any = None) -> Any:
    """Extract and parse JSON from LLM response text."""
    import re

    if not response or not response.strip():
        if default is not None:
            return default
        raise ValueError("Empty response")

    # Try to find JSON in code blocks first
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find JSON array or object directly
        json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", response)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = response.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        if default is not None:
            return default
        raise
