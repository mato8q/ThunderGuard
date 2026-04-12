"""
pii_filter.py – PII detection and masking using Microsoft Presidio.

Runs in parallel with the jailbreak detector as part of the
two-layer input guardrail.  Masks names, email addresses, phone
numbers, and other PII before the sanitised prompt is forwarded
to the main LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

log = logging.getLogger(__name__)

# Entity types to detect and mask
_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "LOCATION",
    "NRP",             # Nationality, Religious, Political
    "DATE_TIME",
    "URL",
]

# Replacement tokens shown in the masked output
_REPLACEMENT = {
    "PERSON":        "[NAME]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "PHONE_NUMBER":  "[PHONE]",
    "CREDIT_CARD":   "[CARD]",
    "IBAN_CODE":     "[IBAN]",
    "IP_ADDRESS":    "[IP]",
    "LOCATION":      "[LOCATION]",
    "NRP":           "[NRP]",
    "DATE_TIME":     "[DATE]",
    "URL":           "[URL]",
}


@dataclass
class PIIResult:
    masked_text: str
    entities_found: list[str] = field(default_factory=list)
    original_text: str = ""


class PIIFilter:
    """
    Wraps Presidio Analyzer + Anonymizer for fast PII masking.

    Latency is typically 5–30 ms for typical prompt lengths,
    well within the 20–100 ms target mentioned in the design spec.
    """

    def __init__(self, language: str = "en", score_threshold: float = 0.5) -> None:
        log.info("Loading Presidio engines …")
        self._analyzer  = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._language = language
        self._score_threshold = score_threshold

        # Build operator config: replace each entity type with its token
        self._operators = {
            entity: OperatorConfig("replace", {"new_value": token})
            for entity, token in _REPLACEMENT.items()
        }
        log.info("PIIFilter ready")

    def mask(self, text: str) -> PIIResult:
        """
        Analyse and mask PII in `text`.

        Returns a PIIResult with the masked text and list of detected entity types.
        """
        results = self._analyzer.analyze(
            text=text,
            entities=_ENTITIES,
            language=self._language,
            score_threshold=self._score_threshold,
        )

        entities_found = list({r.entity_type for r in results})

        if not results:
            return PIIResult(
                masked_text=text,
                entities_found=[],
                original_text=text,
            )

        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=self._operators,
        )

        log.debug("PIIFilter: detected %s → masked", entities_found)
        return PIIResult(
            masked_text=anonymized.text,
            entities_found=entities_found,
            original_text=text,
        )
