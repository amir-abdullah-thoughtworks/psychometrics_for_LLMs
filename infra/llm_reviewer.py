"""
LLM code reviewer and multi-model debater.

Backends (all stateful):
  - OpenAIBackend: Responses API (server-side memory via previous_response_id)
  - GeminiBackend: Interactions API with stateless fallback
  - ClaudeBackend: client-side history + ephemeral prompt caching (~10% input cost)

Reviewer: stateful code review — full files, git diff, or staged changes (pre-commit).
Debater:  multi-model debate with N critique rounds and a final synthesized summary.

Usage:
    from infra.llm_reviewer import Reviewer, Debater, ClaudeBackend

    Reviewer(backend=ClaudeBackend()).review(files={"x.py": code}, review_type="scientific")
    Debater().debate("Is our persona generation prompt introducing bias?", context="...")
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


STATE_DIR = os.path.expanduser("~/.cache/psychometrics-llm-reviewer/judge_state")


def _load_bash_profile():
    """Pull export VAR=... lines from ~/.bash_profile into os.environ if not already set."""
    profile = os.path.expanduser("~/.bash_profile")
    if not os.path.exists(profile):
        return
    import re
    with open(profile) as f:
        for line in f:
            m = re.match(r'^\s*export\s+([A-Z_][A-Z0-9_]*)=["\']?([^"\'#\n]*)["\']?', line)
            if m:
                key, val = m.group(1), m.group(2).strip()
                if key not in os.environ:
                    os.environ[key] = val


_load_bash_profile()

SYSTEM_PROMPT = """You are a senior ML engineer and psychometrics researcher embedded in a team studying LLM validity and personality measurement.

Project context:
- Generating synthetic Law Enforcement Officer (LEO) personas with specific demographic and psychological profiles using LLMs (OpenAI GPT-4.x and Anthropic Claude Sonnet).
- Administering standardized psychometric tests (HEXACO-100 personality inventory, Situational Judgment Tests) to LLMs prompted with those personas.
- Research goals: measure whether LLMs respond consistently and validly across persona conditions; assess response diversity, reliability, and construct validity.
- Persona generation uses Pydantic structured outputs with Literal-constrained fields for demographic seeds; prose fields (appearance, behavior, memoir narrative, psychological profile) are generated freely.
- Inference: local vLLM server (port 9000) for open-source models; Anthropic and OpenAI APIs for proprietary models; Modal.com (A100 GPUs) for large-scale runs.
- Embeddings: Qwen/Qwen3-Embedding-0.6B (1024-dim) over generated prose fields for diversity analysis.
- Key files: llm_psychometrics/src/persona_generation/, llm_psychometrics/src/prompt_templates/, llm_psychometrics/src/utils/openai_utils.py (caching), llm_psychometrics/src/evals/diversity_metrics.py.

Give honest, specific, actionable feedback. Contradict the questioner if warranted. Do not be sycophantic."""


class LLMBackend(ABC):
    """Base class for stateful LLM backends. Subclasses persist state to JSON."""

    def __init__(self, name: str, model: str, state_path: str):
        self.name = name
        self.model = model
        self.state_path = state_path
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        return self._default_state()

    def _save_state(self):
        Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    @abstractmethod
    def _default_state(self) -> dict: ...

    @abstractmethod
    def send(self, prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 3000) -> str: ...

    @abstractmethod
    def reset(self): ...

    @property
    def n_calls(self) -> int:
        return self.state.get("n_calls", 0)


class OpenAIBackend(LLMBackend):
    """Stateful OpenAI backend via Responses API (server-side memory)."""

    def __init__(self, model: str = "gpt-4.1",
                 state_path: str = os.path.join(STATE_DIR, "backend_openai.json")):
        super().__init__("openai", model, state_path)

    def _default_state(self) -> dict:
        return {"last_response_id": None, "n_calls": 0}

    @property
    def _client(self):
        if not hasattr(self, "_openai_client"):
            from openai import OpenAI
            self._openai_client = OpenAI()
        return self._openai_client

    def send(self, prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 3000) -> str:
        kwargs = {
            "model": self.model,
            "input": prompt,
            "instructions": system,
            "max_output_tokens": max_tokens,
            "temperature": 0.3,
        }
        if self.state["last_response_id"]:
            kwargs["previous_response_id"] = self.state["last_response_id"]

        response = self._client.responses.create(**kwargs)
        self.state["last_response_id"] = response.id
        self.state["n_calls"] = self.state.get("n_calls", 0) + 1
        self._save_state()
        return response.output_text

    def reset(self):
        self.state = self._default_state()
        self._save_state()


class GeminiBackend(LLMBackend):
    """Stateful Gemini backend. Uses Interactions API (previousInteractionId) when
    available, falls back to stateless generateContent.

    Gemini 2.5 Pro's thinking budget counts against maxOutputTokens, so we pad
    generously (>= 8192) to leave room for both the thinking trace and the response.
    """

    def __init__(self, model: str = "gemini-2.5-pro",
                 state_path: str = os.path.join(STATE_DIR, "backend_gemini.json")):
        super().__init__("gemini", model, state_path)

    def _default_state(self) -> dict:
        return {"last_interaction_id": None, "n_calls": 0}

    def send(self, prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 3000) -> str:
        import requests

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return "[Gemini API key not set — skipped]"

        gemini_max = max(max_tokens * 3, 8192)
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={api_key}")
        body = {
            "contents": [{"parts": [{"text": f"{system}\n\n{prompt}"}]}],
            "generationConfig": {"maxOutputTokens": gemini_max},
        }
        prev_id = self.state.get("last_interaction_id")
        if prev_id:
            body["previousInteractionId"] = prev_id

        try:
            r = requests.post(url, json=body, timeout=180)
            r.raise_for_status()
            data = r.json()

            interaction_id = data.get("interactionId")
            if interaction_id:
                self.state["last_interaction_id"] = interaction_id
            self.state["n_calls"] = self.state.get("n_calls", 0) + 1
            self._save_state()

            parts = data["candidates"][0]["content"].get("parts", [])
            return "\n".join(p["text"] for p in parts if "text" in p) or "[No text in response]"
        except Exception as e:
            return f"[Gemini error: {e}]"

    def reset(self):
        self.state = self._default_state()
        self._save_state()


class ClaudeBackend(LLMBackend):
    """Client-side stateful Claude backend with ephemeral prompt caching.

    Anthropic's Messages API has no server-side session primitive (unlike
    OpenAI's previous_response_id), so history is maintained locally and sent
    each call. We cache the system prompt and up to 3 early messages (max 4
    cache_control blocks total) — cached tokens are 10% of input rate.
    """

    def __init__(self, model: str = "claude-sonnet-4-6",
                 state_path: str = os.path.join(STATE_DIR, "backend_claude.json")):
        super().__init__("claude", model, state_path)

    def _default_state(self) -> dict:
        return {"messages": [], "n_calls": 0}

    @property
    def _client(self):
        if not hasattr(self, "_anthropic_client"):
            import anthropic
            self._anthropic_client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        return self._anthropic_client

    def send(self, prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 3000) -> str:
        self.state["messages"].append({"role": "user", "content": prompt})

        system_blocks = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
        ]

        messages = []
        n_msgs = len(self.state["messages"])
        cache_budget = 3  # 4 total cache_control blocks - 1 system
        cache_count = 0
        for i, msg in enumerate(self.state["messages"]):
            m = {"role": msg["role"], "content": msg["content"]}
            if i < n_msgs - 2 and cache_count < cache_budget:
                m["content"] = [{
                    "type": "text", "text": msg["content"],
                    "cache_control": {"type": "ephemeral"},
                }]
                cache_count += 1
            messages.append(m)

        try:
            response = self._client.messages.create(
                model=self.model, max_tokens=max_tokens,
                system=system_blocks, messages=messages, temperature=0.3,
            )
            text = response.content[0].text
            self.state["messages"].append({"role": "assistant", "content": text})
            self.state["n_calls"] = self.state.get("n_calls", 0) + 1
            self._save_state()
            return text
        except Exception as e:
            self.state["messages"].pop()  # drop the failed user message
            return f"[Claude error: {e}]"

    def reset(self):
        self.state = self._default_state()
        self._save_state()


REVIEW_PROMPTS = {
    "architecture": (
        "ARCHITECTURE REVIEW: For each design decision in this code, state the "
        "decision explicitly, then evaluate: is there a better approach for "
        "accuracy, scalability, cost, or maintainability? What existing APIs/"
        "tools/patterns could replace custom code? What would you do differently?"
    ),
    "implementation": (
        "IMPLEMENTATION REVIEW: Check for bugs, concurrency issues, error "
        "handling gaps, resource leaks, race conditions, and anything that "
        "could produce incorrect results."
    ),
    "scientific": (
        "SCIENTIFIC REVIEW: This is research code for an LLM psychometrics experiment. Check:\n"
        "1. VALIDITY: Does the prompt/schema actually measure what's claimed? "
        "   Could the model be exploiting artifacts rather than the intended construct?\n"
        "2. RELIABILITY: Would the same persona seed produce consistent responses across runs? "
        "   Are stochastic elements (temperature, sampling) appropriate for the claim?\n"
        "3. CONFOUNDS: Are demographic, archetype, and memoir seeds fully crossed, "
        "   or could an unintended correlation drive the result?\n"
        "4. PROMPT LEAKAGE: Does the prompt inadvertently reveal the expected answer "
        "   or score direction to the model?\n"
        "5. SCHEMA CONSTRAINTS: Do Literal fields over-constrain the model in ways "
        "   that could inflate or deflate measured diversity?\n"
        "6. CLAIMS vs EVIDENCE: Does the code actually test what's claimed? "
        "   Any unstated assumptions?"
    ),
}


class Reviewer:
    """Stateful code reviewer over any LLMBackend."""

    def __init__(self, backend: Optional[LLMBackend] = None):
        self.backend = backend or ClaudeBackend()

    def review(self, files: dict = None, prompt: str = "",
               review_type: str = "all", max_tokens: int = 3000) -> str:
        parts = self._review_prompts(review_type)
        if prompt:
            parts.append(prompt)
        if files:
            parts.append("")
            for name, content in files.items():
                parts.append(f"=== {name} ===\n{content}")
        return self.backend.send("\n\n".join(parts), max_tokens=max_tokens)

    def review_diff(self, diff: str = None, files_context: dict = None,
                    prompt: str = "", review_type: str = "all",
                    max_tokens: int = 3000) -> str:
        import subprocess
        if diff is None:
            r = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True)
            diff = r.stdout
            if not diff.strip():
                r = subprocess.run(["git", "diff"], capture_output=True, text=True)
                diff = r.stdout
            if not diff.strip():
                return "No diff found (nothing staged or modified)."

        parts = self._review_prompts(review_type)
        parts.append(
            "DIFF REVIEW: Focus on the CHANGES below. Flag bugs or scientific issues "
            "introduced by this diff. Don't comment on unchanged code unless a change "
            "breaks it. Be concise — one comment per issue, reference the diff line."
        )
        if prompt:
            parts.append(prompt)
        if files_context:
            parts.append("\n--- Full file context (for reference) ---")
            for name, content in files_context.items():
                parts.append(f"=== {name} ===\n{content}")
        parts.append(f"\n--- Diff ---\n{diff}")
        return self.backend.send("\n\n".join(parts), max_tokens=max_tokens)

    def review_staged(self, prompt: str = "", review_type: str = "implementation",
                      include_context: bool = True, max_tokens: int = 3000) -> str:
        import subprocess
        r = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True)
        diff = r.stdout
        if not diff.strip():
            return "Nothing staged."

        files_context = {}
        if include_context:
            r = subprocess.run(["git", "diff", "--cached", "--name-only"],
                               capture_output=True, text=True)
            for fname in r.stdout.strip().split("\n"):
                if fname and os.path.exists(fname):
                    try:
                        files_context[fname] = open(fname).read()
                    except Exception:
                        pass
        return self.review_diff(diff=diff, files_context=files_context,
                                prompt=prompt, review_type=review_type,
                                max_tokens=max_tokens)

    def reset(self):
        self.backend.reset()

    @property
    def n_reviews(self) -> int:
        return self.backend.n_calls

    @staticmethod
    def _review_prompts(review_type: str) -> list:
        if review_type == "all":
            types = ["implementation", "architecture", "scientific"]
        elif review_type == "both":
            types = ["implementation", "architecture"]
        else:
            types = [review_type]
        return [REVIEW_PROMPTS[t] for t in types if t in REVIEW_PROMPTS]


CRITIQUE_TEMPLATE = """Here are responses from other models to the same question.

{other_responses}

Your task:
1. Where do you AGREE with the other responses? Be specific.
2. Where do you DISAGREE? Explain why with technical reasoning.
3. What did the others MISS that you think is important?
4. What is your final position given all perspectives?

Do not be sycophantic. If you think another model is wrong, say so directly."""


@dataclass
class DebateRound:
    responses: dict = field(default_factory=dict)


@dataclass
class DebateResult:
    question: str = ""
    context: str = ""
    rounds: list = field(default_factory=list)
    summary: str = ""


class Debater:
    """Multi-model debate with critique rounds. Plug in any LLMBackends."""

    def __init__(self, backends: Optional[List[LLMBackend]] = None):
        self.backends = backends or [ClaudeBackend(), OpenAIBackend(), GeminiBackend()]

    def debate(self, question: str, context: str = "",
               rounds: int = 2, max_tokens: int = 3000) -> DebateResult:
        result = DebateResult(question=question, context=context)
        full_prompt = f"{context}\n\n{question}" if context else question

        r1 = DebateRound()
        for b in self.backends:
            r1.responses[b.name] = b.send(full_prompt, max_tokens=max_tokens)
        result.rounds.append(r1)

        for _ in range(1, rounds):
            r = DebateRound()
            prev = result.rounds[-1]
            for b in self.backends:
                others = "\n\n".join(
                    f"**{name}**: {text}"
                    for name, text in prev.responses.items()
                    if name != b.name
                )
                r.responses[b.name] = b.send(
                    CRITIQUE_TEMPLATE.format(other_responses=others),
                    max_tokens=max_tokens,
                )
            result.rounds.append(r)

        result.summary = self.backends[0].send(
            "Synthesize the debate into a concise summary. For each point: state "
            "consensus or disagreement with evidence. End with a clear list of "
            "recommended actions.",
            max_tokens=2000,
        )
        return result

    def reset(self):
        for b in self.backends:
            b.reset()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        for B in [ClaudeBackend, OpenAIBackend, GeminiBackend]:
            B().reset()
        print("All backend sessions reset.")

    elif len(sys.argv) > 1 and sys.argv[1] == "status":
        for B in [ClaudeBackend, OpenAIBackend, GeminiBackend]:
            b = B()
            print(f"{b.name}: {b.n_calls} calls")

    elif len(sys.argv) > 2 and sys.argv[1] == "review":
        prompt = " ".join(sys.argv[2:])
        for B in [ClaudeBackend, OpenAIBackend, GeminiBackend]:
            r = Reviewer(backend=B())
            print(f"\n{'='*60}\n{B.__name__}\n{'='*60}")
            print(r.review(prompt=prompt))

    elif len(sys.argv) > 2 and sys.argv[1] == "debate":
        question = " ".join(sys.argv[2:])
        d = Debater()
        result = d.debate(question)
        for i, rnd in enumerate(result.rounds):
            print(f"\n{'='*60}\nROUND {i+1}\n{'='*60}")
            for name, text in rnd.responses.items():
                print(f"\n--- {name} ---\n{text}")
        print(f"\n{'='*60}\nSUMMARY\n{'='*60}\n{result.summary}")

    else:
        print("Usage:")
        print("  python -m infra.llm_reviewer status")
        print("  python -m infra.llm_reviewer reset")
        print("  python -m infra.llm_reviewer review 'Is the RNG replay in generate_one_from_seed correct?'")
        print("  python -m infra.llm_reviewer debate 'Does passing appearance examples vs prose change persona diversity?'")
