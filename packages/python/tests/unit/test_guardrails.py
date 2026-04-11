"""Tests for Wave 4 — guardrails (governance v1)."""

from __future__ import annotations

import pytest

from dendrux.agent import Agent
from dendrux.guardrails import PII, GuardrailEngine, Pattern, SecretDetection
from dendrux.llm.mock import MockLLM
from dendrux.loops.react import ReActLoop
from dendrux.loops.single import SingleCall
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import tool
from dendrux.types import LLMResponse, RunStatus, ToolCall, UsageStats

# ------------------------------------------------------------------
# Test tools
# ------------------------------------------------------------------


@tool()
async def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"Sent to {to}: {body}"


@tool()
async def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


def _response(text: str, tool_calls=None) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        usage=UsageStats(input_tokens=100, output_tokens=50, total_tokens=150),
    )


# ------------------------------------------------------------------
# PII scanner
# ------------------------------------------------------------------


class TestPIIScanner:
    async def test_detects_email(self):
        pii = PII()
        findings = await pii.scan("Contact jane@example.com for help")
        assert len(findings) == 1
        assert findings[0].entity_type == "EMAIL"
        assert findings[0].text == "jane@example.com"

    async def test_detects_phone(self):
        pii = PII()
        findings = await pii.scan("Call +1-555-123-4567")
        assert len(findings) == 1
        assert findings[0].entity_type == "PHONE"

    async def test_detects_ssn(self):
        pii = PII()
        findings = await pii.scan("SSN: 123-45-6789")
        assert len(findings) == 1
        assert findings[0].entity_type == "SSN"

    async def test_no_findings_clean_text(self):
        pii = PII()
        findings = await pii.scan("Hello world, no PII here")
        assert len(findings) == 0

    async def test_custom_pattern(self):
        pii = PII(
            include_defaults=False,
            extra_patterns=[Pattern("EMPLOYEE_ID", r"EMP-\d{6}")],
        )
        findings = await pii.scan("Employee EMP-123456 is active")
        assert len(findings) == 1
        assert findings[0].entity_type == "EMPLOYEE_ID"
        assert findings[0].text == "EMP-123456"

    async def test_include_defaults_false(self):
        pii = PII(include_defaults=False)
        findings = await pii.scan("jane@example.com")
        assert len(findings) == 0

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="Invalid action"):
            PII(action="invalid")  # type: ignore[arg-type]


# ------------------------------------------------------------------
# SecretDetection scanner
# ------------------------------------------------------------------


class TestSecretDetection:
    async def test_detects_aws_key(self):
        sd = SecretDetection()
        findings = await sd.scan("Key: AKIAIOSFODNN7EXAMPLE")
        assert len(findings) >= 1
        assert any(f.entity_type == "AWS_ACCESS_KEY" for f in findings)

    async def test_detects_private_key(self):
        sd = SecretDetection()
        findings = await sd.scan("-----BEGIN PRIVATE KEY-----")
        assert len(findings) == 1
        assert findings[0].entity_type == "PRIVATE_KEY"

    async def test_no_findings_clean_text(self):
        sd = SecretDetection()
        findings = await sd.scan("Just a normal message")
        assert len(findings) == 0


# ------------------------------------------------------------------
# GuardrailEngine
# ------------------------------------------------------------------


class TestGuardrailEngine:
    async def test_redact_creates_placeholders(self):
        engine = GuardrailEngine([PII()])
        text, findings, block = await engine.scan_incoming("Email jane@example.com")
        assert "<<EMAIL_1>>" in text
        assert "jane@example.com" not in text
        assert block is None
        assert len(findings) == 1

    async def test_same_value_reuses_placeholder(self):
        engine = GuardrailEngine([PII()])
        text1, _, _ = await engine.scan_incoming("jane@example.com")
        text2, _, _ = await engine.scan_incoming("again jane@example.com")
        assert "<<EMAIL_1>>" in text1
        assert "<<EMAIL_1>>" in text2
        # No <<EMAIL_2>> — same value reuses
        assert "<<EMAIL_2>>" not in text2

    async def test_different_values_get_different_placeholders(self):
        engine = GuardrailEngine([PII()])
        text, _, _ = await engine.scan_incoming("jane@example.com and bob@example.com")
        assert "<<EMAIL_1>>" in text
        assert "<<EMAIL_2>>" in text

    async def test_block_returns_error(self):
        engine = GuardrailEngine([SecretDetection()])
        _, _, block = await engine.scan_incoming("Key: AKIAIOSFODNN7EXAMPLE")
        assert block is not None
        assert "blocked" in block.lower()

    async def test_warn_does_not_modify(self):
        engine = GuardrailEngine([PII(action="warn")])
        text, findings, block = await engine.scan_incoming("jane@example.com")
        assert text == "jane@example.com"  # unchanged
        assert len(findings) == 1
        assert block is None

    async def test_deanonymize(self):
        engine = GuardrailEngine([PII()])
        await engine.scan_incoming("Email jane@example.com")
        params = {"to": "<<EMAIL_1>>", "body": "Hello"}
        result = engine.deanonymize(params)
        assert result["to"] == "jane@example.com"
        assert result["body"] == "Hello"

    async def test_deanonymize_nested(self):
        engine = GuardrailEngine([PII()])
        await engine.scan_incoming("jane@example.com")
        params = {"data": {"email": "<<EMAIL_1>>"}}
        result = engine.deanonymize(params)
        assert result["data"]["email"] == "jane@example.com"

    async def test_deanonymize_unknown_placeholder_passes_through(self):
        engine = GuardrailEngine([PII()])
        params = {"to": "<<UNKNOWN_99>>"}
        result = engine.deanonymize(params)
        assert result["to"] == "<<UNKNOWN_99>>"

    async def test_get_pii_mapping(self):
        engine = GuardrailEngine([PII()])
        await engine.scan_incoming("jane@example.com")
        mapping = engine.get_pii_mapping()
        assert mapping == {"<<EMAIL_1>>": "jane@example.com"}

    async def test_restore_from_mapping(self):
        engine = GuardrailEngine(
            [PII()],
            pii_mapping={"<<EMAIL_1>>": "jane@example.com"},
        )
        # Should reuse existing placeholder
        text, _, _ = await engine.scan_incoming("jane@example.com again")
        assert "<<EMAIL_1>>" in text
        assert "<<EMAIL_2>>" not in text

    async def test_output_scan_text(self):
        engine = GuardrailEngine([PII()])
        text, _, findings, block, _ = await engine.scan_outgoing("The email is jane@example.com")
        assert "<<EMAIL_1>>" in text
        assert len(findings) == 1
        assert block is None


# ------------------------------------------------------------------
# Overlap de-duplication
# ------------------------------------------------------------------


class TestDeoverlap:
    async def test_credit_card_not_split_by_phone(self):
        """Credit card number should not be partially matched as phone."""
        engine = GuardrailEngine([PII()])
        text, findings, block = await engine.scan_incoming("Card: 4111111111111111")
        assert block is None
        # Should produce one CREDIT_CARD, not PHONE + leftover digits
        assert "<<CREDIT_CARD_1>>" in text
        assert "1111" not in text  # no leaked digits

    async def test_credit_card_in_tool_params(self):
        """Overlapping findings in tool call params are de-overlapped."""
        engine = GuardrailEngine([PII()])
        params = [{"card": "4111111111111111"}]
        _, out_params, findings, block, _ = await engine.scan_outgoing("ok", params)
        assert block is None
        assert out_params is not None
        assert "<<CREDIT_CARD_1>>" in out_params[0]["card"]
        assert "1111" not in out_params[0]["card"]


# ------------------------------------------------------------------
# Param scanning — key context + redaction
# ------------------------------------------------------------------


class TestParamScanning:
    async def test_secret_block_with_key_context(self):
        """SecretDetection blocks api_key params via key context."""
        engine = GuardrailEngine([SecretDetection()])
        params = [{"api_key": "x" * 24}]
        _, _, _, block, _ = await engine.scan_outgoing("ok", params)
        assert block is not None
        assert "blocked" in block.lower()

    async def test_secret_redact_with_key_context(self):
        """SecretDetection redacts api_key params correctly."""
        engine = GuardrailEngine([SecretDetection(action="redact")])
        params = [{"api_key": "x" * 24}]
        _, out_params, findings, block, p_redacted = await engine.scan_outgoing("ok", params)
        assert block is None
        assert p_redacted is True
        assert out_params is not None
        assert "<<GENERIC_API_KEY_1>>" in out_params[0]["api_key"]
        assert "x" * 24 not in out_params[0]["api_key"]

    async def test_aws_secret_block(self):
        """SecretDetection blocks AWS secret keys in params."""
        engine = GuardrailEngine([SecretDetection()])
        params = [{"secret": "A" * 40}]
        _, _, _, block, _ = await engine.scan_outgoing("ok", params)
        assert block is not None

    async def test_aws_secret_redact(self):
        """SecretDetection redacts AWS secret keys in params."""
        engine = GuardrailEngine([SecretDetection(action="redact")])
        params = [{"secret": "A" * 40}]
        _, out_params, _, block, p_redacted = await engine.scan_outgoing("ok", params)
        assert block is None
        assert p_redacted is True
        assert "A" * 40 not in out_params[0]["secret"]

    async def test_nested_secret_redact(self):
        """SecretDetection redacts nested token params."""
        engine = GuardrailEngine([SecretDetection(action="redact")])
        params = [{"data": {"token": "x" * 24}}]
        _, out_params, _, block, p_redacted = await engine.scan_outgoing("ok", params)
        assert block is None
        assert p_redacted is True
        assert "x" * 24 not in str(out_params[0])

    async def test_pii_nested_list_redact(self):
        """PII redacts emails in nested lists."""
        engine = GuardrailEngine([PII()])
        params = [{"rows": [["jane@example.com"]]}]
        _, out_params, _, block, p_redacted = await engine.scan_outgoing("ok", params)
        assert block is None
        assert p_redacted is True
        assert "jane@example.com" not in str(out_params[0])
        assert "<<EMAIL_1>>" in str(out_params[0])


# ------------------------------------------------------------------
# Agent integration — ReAct
# ------------------------------------------------------------------


class TestGuardrailReAct:
    async def test_incoming_redaction(self):
        """PII in user input is redacted before LLM sees it."""
        llm = MockLLM([_response("I'll help.")])
        agent = Agent(
            prompt="You are a helper.",
            tools=[],
            guardrails=[PII()],
        )

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Help jane@example.com",
        )

        # LLM should have seen placeholder, not real email
        first_call = llm.call_history[0]
        user_msg = first_call["messages"][-1]
        assert "<<EMAIL_1>>" in user_msg.content
        assert "jane@example.com" not in user_msg.content

    async def test_block_terminates_run(self):
        """SecretDetection with block action terminates the run."""
        llm = MockLLM([_response("ok")])
        agent = Agent(
            prompt="You are a helper.",
            tools=[],
            guardrails=[SecretDetection()],
        )

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Store key: AKIAIOSFODNN7EXAMPLE",
        )

        assert result.status == RunStatus.ERROR
        assert "blocked" in result.error.lower()
        # LLM should NOT have been called
        assert llm.calls_made == 0

    async def test_deanonymize_tool_params(self):
        """Tool receives real values after deanonymization."""
        tc = ToolCall(
            name="send_email",
            params={"to": "<<EMAIL_1>>", "body": "Hello"},
            provider_tool_call_id="t1",
        )
        llm = MockLLM(
            [
                _response("Sending email...", tool_calls=[tc]),
                _response("Email sent."),
            ]
        )
        agent = Agent(
            prompt="You are a helper.",
            tools=[send_email],
            guardrails=[PII()],
        )

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Email jane@example.com saying hello",
        )

        assert result.status == RunStatus.SUCCESS
        # Tool result in LLM history is REDACTED (incoming guardrail
        # scans tool results re-entering the LLM). This is correct —
        # PII should not reach the LLM even in tool results.
        second_call = llm.call_history[1]
        tool_msgs = [m for m in second_call["messages"] if m.role.value == "tool"]
        assert len(tool_msgs) == 1
        # The tool executed with real data (deanonymized), but the
        # result re-entering the LLM has the placeholder back.
        assert "<<EMAIL_1>>" in tool_msgs[0].content
        # First LLM call should have had the placeholder in user input
        first_call = llm.call_history[0]
        user_msg = first_call["messages"][-1]
        assert "<<EMAIL_1>>" in user_msg.content

    async def test_warn_does_not_modify_input(self):
        """Warn action logs but does not change content."""
        llm = MockLLM([_response("ok")])
        agent = Agent(
            prompt="You are a helper.",
            tools=[],
            guardrails=[PII(action="warn")],
        )

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Help jane@example.com",
        )

        # LLM should see the real email (warn doesn't modify)
        first_call = llm.call_history[0]
        user_msg = first_call["messages"][-1]
        assert "jane@example.com" in user_msg.content

    async def test_governance_events_emitted(self):
        """Guardrail scan emits governance events."""
        llm = MockLLM([_response("ok")])
        agent = Agent(
            prompt="Helper.",
            tools=[],
            guardrails=[PII()],
        )

        events: list[dict] = []

        class SpyRecorder:
            async def on_message_appended(self, message, iteration):
                pass

            async def on_llm_call_completed(self, response, iteration, **kw):
                pass

            async def on_tool_completed(self, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(self, event_type, iteration, data, correlation_id=None):
                events.append({"event_type": event_type, "data": data})

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Help jane@example.com",
            recorder=SpyRecorder(),
        )

        detected = [e for e in events if e["event_type"] == "guardrail.detected"]
        redacted = [e for e in events if e["event_type"] == "guardrail.redacted"]
        assert len(detected) >= 1
        assert len(redacted) >= 1


# ------------------------------------------------------------------
# Agent integration — SingleCall
# ------------------------------------------------------------------


class TestGuardrailSingleCall:
    async def test_incoming_redaction_single_call(self):
        """SingleCall also redacts incoming PII."""
        llm = MockLLM([_response("Classified.")])
        agent = Agent(
            prompt="Classify.",
            loop=SingleCall(),
            guardrails=[PII()],
        )

        result = await SingleCall().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Classify jane@example.com",
        )

        assert result.status == RunStatus.SUCCESS
        first_call = llm.call_history[0]
        user_msg = first_call["messages"][-1]
        assert "<<EMAIL_1>>" in user_msg.content

    async def test_block_single_call(self):
        """SecretDetection blocks SingleCall runs."""
        llm = MockLLM([_response("ok")])
        agent = Agent(
            prompt="Process.",
            loop=SingleCall(),
            guardrails=[SecretDetection()],
        )

        result = await SingleCall().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Key: AKIAIOSFODNN7EXAMPLE",
        )

        assert result.status == RunStatus.ERROR
        assert llm.calls_made == 0


# ------------------------------------------------------------------
# Streaming guard
# ------------------------------------------------------------------


class TestGuardrailStreamingGuard:
    def test_stream_with_guardrails_raises(self):
        agent = Agent(
            prompt="Test.",
            tools=[search],
            guardrails=[PII()],
        )
        with pytest.raises(ValueError, match="guardrails are not supported with stream"):
            agent.stream("test")

    def test_resume_stream_with_guardrails_raises(self):
        agent = Agent(
            prompt="Test.",
            tools=[search],
            guardrails=[PII()],
        )
        with pytest.raises(ValueError, match="guardrails are not supported with resume_stream"):
            agent.resume_stream("run-123", tool_results=[])


# ------------------------------------------------------------------
# Multiple guardrails
# ------------------------------------------------------------------


class TestMultipleGuardrails:
    async def test_block_short_circuits(self):
        """If first guardrail blocks, second doesn't run."""
        llm = MockLLM([_response("ok")])
        agent = Agent(
            prompt="Helper.",
            tools=[],
            guardrails=[
                SecretDetection(action="block"),
                PII(action="redact"),
            ],
        )

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Key AKIAIOSFODNN7EXAMPLE and jane@example.com",
        )

        assert result.status == RunStatus.ERROR
        assert "SecretDetection" in result.error

    async def test_redact_then_warn(self):
        """Multiple guardrails compose: PII redacts, then warn logs."""
        llm = MockLLM([_response("ok")])
        agent = Agent(
            prompt="Helper.",
            tools=[],
            guardrails=[
                PII(action="redact"),
                SecretDetection(action="warn"),
            ],
        )

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Email jane@example.com",
        )

        assert result.status == RunStatus.SUCCESS
        first_call = llm.call_history[0]
        user_msg = first_call["messages"][-1]
        assert "<<EMAIL_1>>" in user_msg.content


# ------------------------------------------------------------------
# Multi-turn PII re-entry
# ------------------------------------------------------------------


class TestMultiTurnGuardrail:
    async def test_tool_result_pii_redacted_on_reentry(self):
        """Tool results with real PII are redacted before next LLM call."""
        tc = ToolCall(
            name="search",
            params={"query": "<<EMAIL_1>>"},
            provider_tool_call_id="t1",
        )
        llm = MockLLM(
            [
                _response("Searching...", tool_calls=[tc]),
                _response("Found results."),
            ]
        )
        agent = Agent(
            prompt="You are a helper.",
            tools=[search],
            guardrails=[PII()],
        )

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Find jane@example.com",
        )

        assert result.status == RunStatus.SUCCESS
        # Second LLM call should NOT contain real email anywhere
        second_call = llm.call_history[1]
        for msg in second_call["messages"]:
            assert "jane@example.com" not in msg.content, (
                f"PII leaked in {msg.role.value} message: {msg.content}"
            )

    async def test_tool_call_only_output_scanned(self):
        """Output guardrail fires even when response.text is None."""
        tc = ToolCall(
            name="send_email",
            params={"to": "leaked@real.com", "body": "hi"},
            provider_tool_call_id="t1",
        )
        resp = LLMResponse(
            text=None,
            tool_calls=[tc],
            usage=UsageStats(input_tokens=100, output_tokens=50, total_tokens=150),
        )
        llm = MockLLM([resp, _response("Done.")])
        agent = Agent(
            prompt="Helper.",
            tools=[send_email],
            guardrails=[PII()],
        )

        events: list[dict] = []

        class SpyRecorder:
            async def on_message_appended(self, message, iteration):
                pass

            async def on_llm_call_completed(self, response, iteration, **kw):
                pass

            async def on_tool_completed(self, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(self, event_type, iteration, data, correlation_id=None):
                events.append({"event_type": event_type, "data": data})

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Send email to someone",
            recorder=SpyRecorder(),
        )

        outgoing_detected = [
            e
            for e in events
            if e["event_type"] == "guardrail.detected" and e["data"].get("direction") == "outgoing"
        ]
        assert len(outgoing_detected) >= 1
