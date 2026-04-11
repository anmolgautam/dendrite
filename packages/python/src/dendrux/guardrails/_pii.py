"""PII guardrail — regex-based PII detection.

Detects common PII patterns: email, phone, SSN, credit card, IP address.
Extensible via extra_patterns for domain-specific PII.

Future: auto-detects Presidio if installed for NLP-based detection.
"""

from __future__ import annotations

import re
from typing import Literal

from dendrux.guardrails._protocol import Finding, Pattern

_DEFAULT_PATTERNS: list[Pattern] = [
    Pattern("EMAIL", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    Pattern("PHONE", r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    Pattern("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    Pattern(
        "CREDIT_CARD",
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    ),
    Pattern("IP_ADDRESS", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
]


class PII:
    """Regex-based PII detection guardrail.

    Scans text for common PII patterns and returns findings.
    Framework applies the action (redact/block/warn).

    Args:
        action: What the framework does with findings.
            "redact" (default) replaces PII with <<TYPE_N>> placeholders.
            "block" terminates the run.
            "warn" logs but doesn't modify.
        extra_patterns: Additional Pattern objects for domain-specific PII.
        include_defaults: Include built-in patterns (email, phone, SSN,
            credit card, IP). Set False to use only extra_patterns.
    """

    def __init__(
        self,
        *,
        action: Literal["redact", "block", "warn"] = "redact",
        extra_patterns: list[Pattern] | None = None,
        include_defaults: bool = True,
    ) -> None:
        if action not in ("redact", "block", "warn"):
            raise ValueError(f"Invalid action: {action!r}. Must be 'redact', 'block', or 'warn'.")
        self.action: Literal["redact", "block", "warn"] = action
        self._patterns: list[Pattern] = []
        if include_defaults:
            self._patterns.extend(_DEFAULT_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)
        # Pre-compile for performance
        self._compiled: list[tuple[str, re.Pattern[str]]] = [
            (p.name, re.compile(p.regex)) for p in self._patterns
        ]

    async def scan(self, text: str) -> list[Finding]:
        """Scan text for PII patterns."""
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
        # Sort by position for deterministic replacement order
        findings.sort(key=lambda f: f.start)
        return findings
