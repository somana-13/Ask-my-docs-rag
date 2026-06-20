from __future__ import annotations

import re


def clean_claim_text(claim: str) -> str:
    """
    Remove agent/template wording so the NLI model scores only factual claims.
    """
    claim = claim.strip()

    prefix_patterns = [
        r"^For ['\"].+?['\"], the most relevant documentation says:\s*",
        r"^For ['\"].+?['\"], the relevant documentation says:\s*",
        r"^For sub-question ['\"].+?['\"],\s*",
        r"^Based on the retrieved documentation, here is the most relevant evidence:\s*",
        r"^Based on the retrieved documentation,?\s*",
        r"^This question requires evidence from multiple parts of the documentation\.?\s*",
    ]

    for pattern in prefix_patterns:
        claim = re.sub(pattern, "", claim, flags=re.IGNORECASE)

    return claim.strip()


def is_bad_claim(claim: str) -> bool:
    """
    Filter out incomplete or structural fragments that should not be NLI-scored.
    """
    lowered = claim.lower().strip()

    if not lowered:
        return True

    skip_prefixes = [
        "based on the retrieved documentation",
        "this question requires evidence",
        "for sub-question",
        "for '",
        'for "',
    ]

    if any(lowered.startswith(prefix) for prefix in skip_prefixes):
        return True

    # Incomplete markdown/list fragments create false NLI failures.
    bad_endings = [
        "...",
        "following...",
        "a.k.a.",
        "e.g.",
        "i.e.",
    ]

    if any(lowered.endswith(ending) for ending in bad_endings):
        return True

    if len(claim.split()) < 5:
        return True

    return False


def extract_claims(answer: str) -> list[str]:
    """
    Extract sentence-level factual claims from an answer.

    This deterministic version removes template/meta text before NLI scoring.
    """
    if not answer or not answer.strip():
        return []

    cleaned = answer.strip()

    # Remove bullet markers while preserving sentence text.
    cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned, flags=re.MULTILINE)

    # Split on sentence boundaries.
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)

    claims = []
    for sentence in sentences:
        claim = clean_claim_text(sentence)

        if is_bad_claim(claim):
            continue

        claims.append(claim)

    return claims


if __name__ == "__main__":
    sample = """
    For 'How do I configure authentication in HTTPX?', the relevant documentation says: HTTPX can be configured to use a .netrc config file for authentication.
    For 'How do I configure SSL certificates in HTTPX?', the relevant documentation says: When making a request over HTTPS, HTTPX needs to verify the identity of the requested host.
    The auth argument may be one of the following...
    To do this, it uses a bundle of SSL certificates (a.k.a.
    """
    for claim in extract_claims(sample):
        print(f"- {claim}")
