"""Secret detection guardrail — regex-based secret/credential detection.

Detects API keys, AWS credentials, and private key headers.
Default action is "block" — secrets should not reach the LLM.
"""

from __future__ import annotations

import re
from typing import Literal

from dendrux.guardrails._protocol import Finding, Pattern

_SECRET_PATTERNS: list[Pattern] = [
    Pattern("AWS_ACCESS_KEY", r"AKIA[0-9A-Z]{16}"),
    Pattern("AWS_SECRET_KEY", r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),
    Pattern(
        "GENERIC_API_KEY",
        r"(?i)(?:api[_\-]?key|apikey|token|secret[_\-]?key)[=:\s]+[\"']?[A-Za-z0-9_\-]{20,}[\"']?",
    ),
    Pattern("PRIVATE_KEY", r"-----BEGIN\s+(?:RSA\s+)?PRIVATE KEY-----"),
]


class SecretDetection:
    """Regex-based secret/credential detection guardrail.

    Detects AWS keys, API keys, and private key headers.
    Default action is "block" — secrets should not reach the LLM.

    Args:
        action: What the framework does with findings.
            "block" (default) terminates the run.
            "redact" replaces secrets with placeholders.
            "warn" logs but doesn't modify.
        extra_patterns: Additional Pattern objects for custom secret types.
        include_defaults: Include built-in patterns. Set False to use
            only extra_patterns.
    """

    def __init__(
        self,
        *,
        action: Literal["redact", "block", "warn"] = "block",
        extra_patterns: list[Pattern] | None = None,
        include_defaults: bool = True,
    ) -> None:
        if action not in ("redact", "block", "warn"):
            raise ValueError(f"Invalid action: {action!r}. Must be 'redact', 'block', or 'warn'.")
        self.action: Literal["redact", "block", "warn"] = action
        self._patterns: list[Pattern] = []
        if include_defaults:
            self._patterns.extend(_SECRET_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)
        self._compiled: list[tuple[str, re.Pattern[str]]] = [
            (p.name, re.compile(p.regex)) for p in self._patterns
        ]

    async def scan(self, text: str) -> list[Finding]:
        """Scan text for secret patterns."""
        findings: list[Finding] = []
        for entity_type, pattern in self._compiled:
            for match in pattern.finditer(text):
                findings.append(
                    Finding(
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        score=1.0,
                        text=match.group(),
                    )
                )
        findings.sort(key=lambda f: f.start)
        return findings
