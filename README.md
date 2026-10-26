"""Small offline demo of LLM-style tool use, observability, and evaluation."""

from __future__ import annotations

import json
import operator
import re
import time
import uuid
from pathlib import Path

OUTPUT_DIR = Path("demo_output")
OUTPUT_DIR.mkdir(exist_ok=True)

OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
}


def redact(value):
    """Hide simple synthetic emails and key-like values before persistence."""
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(
            r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
            "[REDACTED_EMAIL]",
            value,
        )
        value = re.sub(
            r"\bkey-[A-Za-z0-9-]{8,}\b",
            "[REDACTED_KEY]",
            value,
        )
    return value


def calculator(expression: str) -> str:
    """Evaluate one deliberately tiny and safe arithmetic expression."""
    match = re.fullmatch(r"\s*(-?\d+)\s*([+\-*])\s*(-?\d+)\s*", expression)
    if not match:
        raise ValueError("Only expressions such as 12 * 3 are supported")

    left, symbol, right = match.groups()
    return str(OPS[symbol](int(left), int(right)))


def fake_llm(prompt: str, observation: str | None = None, *, broken=False):
    """Return a direct answer or a structured calculator request."""
    if observation is not None:
        return {"answer": f"The calculator result is {observation}."}

    expression = re.search(r"-?\d+\s*[+\-*]\s*-?\d+", prompt)

    if expression and broken:
        # This candidate intentionally avoids the required tool and guesses badly.
        return {"answer": "I think the answer is 42."}

    if expression:
        return {
            "tool_call": {
                "name": "calculator",
                "arguments": {"expression": expression.group(0)},
            }
        }

    return {"answer": "Hello! I am the offline demo assistant."}


def run_agent(prompt: str, *, config_name="baseline", broken=False):
    """Run one prompt and save one redacted JSONL trace."""
    trace_id = str(uuid.uuid4())
    started = time.perf_counter()
    events = []
    tool_used = None

    def emit(kind, phase, status=None, **attributes):
        events.append(
            {
                "trace_id": trace_id,
                "kind": kind,
                "phase": phase,
                "status": status,
                "timestamp": time.time(),
                "attributes": redact(attributes),
            }
        )

    emit("run", "started", prompt=prompt, config=config_name)

    emit("model", "started", step=1)
    response = fake_llm(prompt, broken=broken)
    estimated_tokens = len(prompt.split()) + len(json.dumps(response).split())
    emit(
        "model",
        "finished",
        "ok",
        step=1,
        outcome="tool_call" if "tool_call" in response else "final_answer",
        estimated_tokens=estimated_tokens,
    )

    if "tool_call" in response:
        call = response["tool_call"]
        tool_used = call["name"]

        emit("tool", "started", tool=tool_used, arguments=call["arguments"])
        try:
            observation = calculator(call["arguments"]["expression"])
            emit("tool", "finished", "ok", tool=tool_used, output=observation)
        except Exception as error:
            observation = f"ERROR: {type(error).__name__}"
            emit(
                "tool",
                "finished",
                "error",
                tool=tool_used,
                error_type=type(error).__name__,
            )

        emit("model", "started", step=2, has_observation=True)
        response = fake_llm(prompt, observation, broken=broken)
        estimated_tokens += len(json.dumps(response).split())
        emit(
            "model",
            "finished",
            "ok",
            step=2,
            outcome="final_answer",
            estimated_tokens=estimated_tokens,
        )

    answer = response["answer"]
    duration_ms = round((time.perf_counter() - started) * 1000, 3)

    emit(
        "run",
        "finished",
        "ok",
        answer=answer,
        tool_calls=1 if tool_used else 0,
        total_estimated_tokens=estimated_tokens,
        duration_ms=duration_ms,
    )

    trace_path = OUTPUT_DIR / f"trace-{trace_id[:8]}.jsonl"
    with trace_path.open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    return {
        "answer": answer,
        "tool_used": tool_used,
        "tokens": estimated_tokens,
        "duration_ms": duration_ms,
        "trace_path": str(trace_path),
    }


CASES = [
    {
        "id": "greeting",
        "prompt": "Hello",
        "expected_tool": None,
        "required_text": "hello",
    },
    {
        "id": "calculation",
        "prompt": "Calculate 12 * 3",
        "expected_tool": "calculator",
        "required_text": "36",
    },
]


def evaluate(config_name: str, *, broken=False):
    """Run fixed cases and aggregate their deterministic scores."""
    case_results = []

    for case in CASES:
        result = run_agent(
            case["prompt"],
            config_name=config_name,
            broken=broken,
        )

        answer_correct = case["required_text"] in result["answer"].lower()
        tool_correct = result["tool_used"] == case["expected_tool"]

        case_results.append(
            {
                "case_id": case["id"],
                "answer_correct": answer_correct,
                "tool_correct": tool_correct,
                "trace_path": result["trace_path"],
                "tokens": result["tokens"],
                "duration_ms": result["duration_ms"],
            }
        )

    count = len(case_results)
    return {
        "config_name": config_name,
        "case_count": count,
        "task_success_rate": sum(
            case["answer_correct"] for case in case_results
        ) / count,
        "tool_accuracy": sum(
            case["tool_correct"] for case in case_results
        ) / count,
        "total_tokens": sum(case["tokens"] for case in case_results),
        "average_latency_ms": sum(
            case["duration_ms"] for case in case_results
        ) / count,
        "cases": case_results,
    }


if __name__ == "__main__":
    prompt = input("Enter one prompt: ").strip()

    live_result = run_agent(prompt)
    print("\nAnswer:", live_result["answer"])
    print("Tool used:", live_result["tool_used"])
    print("Trace:", live_result["trace_path"])

    baseline = evaluate("baseline")
    candidate = evaluate("broken-candidate", broken=True)

    report = {
        "baseline": baseline,
        "candidate": candidate,
        "regression_detected": (
            candidate["task_success_rate"] < baseline["task_success_rate"]
            or candidate["tool_accuracy"] < baseline["tool_accuracy"]
        ),
    }

    report_path = OUTPUT_DIR / "evaluation-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nBaseline success:", baseline["task_success_rate"])
    print("Candidate success:", candidate["task_success_rate"])
    print("Regression detected:", report["regression_detected"])
    print("Evaluation report:", report_path)
