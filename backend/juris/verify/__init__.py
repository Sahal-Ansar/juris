"""Verification: the exact-quote rule and the Citation Verifier (IDEA_final §6.1)."""

from juris.verify.entailment import (
    ClaimEvidencePair,
    EntailmentJudge,
    Judgement,
    LLMEntailmentJudge,
    PairJudge,
    VerifierOutputError,
)
from juris.verify.quote import normalise_quote, quote_in_text
from juris.verify.verifier import (
    CitationVerifier,
    EvidenceTexts,
    ViewEvidence,
    check_citation,
    chunk_text,
    claim_statuses,
)

__all__ = [
    "CitationVerifier",
    "ClaimEvidencePair",
    "EntailmentJudge",
    "EvidenceTexts",
    "Judgement",
    "LLMEntailmentJudge",
    "PairJudge",
    "VerifierOutputError",
    "ViewEvidence",
    "check_citation",
    "chunk_text",
    "claim_statuses",
    "normalise_quote",
    "quote_in_text",
]
