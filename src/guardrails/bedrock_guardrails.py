"""Wraps AWS Bedrock's standalone ApplyGuardrail API as a content-safety and
grounding check, independent of which model (or no model) produced the text."""

import boto3

GUARDRAIL_ID = "slotphsvs4ge"
GUARDRAIL_VERSION = "1"
REGION = "us-east-1"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=REGION)
    return _client


def check_input(text: str) -> dict:
    """Checks user-provided text: content filters, prompt-attack detection, PII."""
    response = _apply_guardrail(source="INPUT", content=[{"text": {"text": text}}])
    return _build_result(response, original_text=text)


def check_output(text: str, query: str | None = None, grounding_source: str | None = None) -> dict:
    """Checks generated answer text: content filters, PII redaction, contextual grounding."""
    content = []
    if grounding_source:
        content.append({"text": {"text": grounding_source, "qualifiers": ["grounding_source"]}})
    if query:
        content.append({"text": {"text": query, "qualifiers": ["query"]}})
    content.append({"text": {"text": text}})

    response = _apply_guardrail(source="OUTPUT", content=content)
    return _build_result(response, original_text=text)


def _apply_guardrail(source: str, content: list) -> dict:
    client = _get_client()
    return client.apply_guardrail(
        guardrailIdentifier=GUARDRAIL_ID,
        guardrailVersion=GUARDRAIL_VERSION,
        source=source,
        content=content,
    )


def _build_result(response: dict, original_text: str) -> dict:
    intervened = response["action"] == "GUARDRAIL_INTERVENED"
    safe_text = (
        response["outputs"][0]["text"]
        if intervened and response.get("outputs")
        else original_text
    )
    return {
        "intervened": intervened,
        "action_reason": response.get("actionReason", "No action."),
        "safe_text": safe_text,
    }