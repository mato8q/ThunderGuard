"""
Adaptive thunder Firewall
====================
Input-level guardrail: Jailbreak Detection (e5-base-v2 + FAISS)
+ PII Masking (Presidio) + Adaptive Memory Loop (Judge Agent)

References:
  - JBFuzz (Gohil 2025): e5-base-v2 evaluator is 16x faster than GPT-4o
  - Galinkin & Sablotny (2024): embedding + classifier outperforms BERT-based guards

Install:
  pip install sentence-transformers faiss-cpu presidio-analyzer \
              presidio-anonymizer openai python-dotenv
  python -m spacy download en_core_web_lg
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from jailbreak_detector import JailbreakDetector, DetectionResult
from pii_filter import PIIFilter, PIIResult
from judge_agent import JudgeAgent, JudgeVerdict
from vector_store import VectorStore

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FirewallDecision:
    allowed: bool
    reason: str                          # jailbreak_detected | pii_masked | "passed"
    sanitized_prompt: Optional[str] = None
    jailbreak_score: float = 0.0
    pii_entities: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class FirewallResponse:
    decision: FirewallDecision
    llm_response: Optional[str] = None
    judge_verdict: Optional[JudgeVerdict] = None


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive AI Firewall
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveFirewall:
    """
    Two-layer parallel input guardrail before the main LLM.

    Layer 1a – Jailbreak Detection (embedding-based, RAD)
    Layer 1b – PII Masking (Presidio)
    Layer 2  – Judge Agent + Adaptive Memory update
    """

    def __init__(
        self,
        jailbreak_threshold: float = 0.85,
        vector_store_path: str = "jailbreak_store",
        openai_model: str = "gpt-4o-mini",
        enable_adaptive_loop: bool = True,
        llm_base_url: str = None,
        llm_api_key: str = None,
    ) -> None:
        log.info("Initialising Adaptive Firewall …")

        self.vector_store = VectorStore(store_path=vector_store_path)
        self.jailbreak_detector = JailbreakDetector(
            vector_store=self.vector_store,
            threshold=jailbreak_threshold,
        )
        self.pii_filter = PIIFilter()
        self.judge_agent = JudgeAgent(
            model=openai_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        self.enable_adaptive_loop = enable_adaptive_loop

        log.info("Firewall ready (threshold=%.2f, adaptive=%s)",
                 jailbreak_threshold, enable_adaptive_loop)

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, user_prompt: str, llm_callable=None) -> FirewallResponse:
        """
        Full pipeline: guard → LLM → judge → adaptive update.

        Args:
            user_prompt:   Raw text from the user.
            llm_callable:  Function(prompt: str) -> str  (your main LLM).
                           If None, only the guardrail decision is returned.

        Returns:
            FirewallResponse with decision, optional LLM response, optional verdict.
        """
        t0 = time.perf_counter()

        # ── Layer 1: Parallel guardrail ────────────────────────────────────
        detection: DetectionResult = self.jailbreak_detector.check(user_prompt)
        pii_result: PIIResult = self.pii_filter.mask(user_prompt)

        latency_ms = (time.perf_counter() - t0) * 1000
        log.info("Guardrail latency: %.1f ms | jailbreak_score=%.3f | pii=%s",
                 latency_ms, detection.score, pii_result.entities_found)

        # ── Reject if jailbreak detected ───────────────────────────────────
        if detection.is_jailbreak:
            decision = FirewallDecision(
                allowed=False,
                reason="jailbreak_detected",
                jailbreak_score=detection.score,
                latency_ms=latency_ms,
            )
            log.warning("BLOCKED (jailbreak_score=%.3f): %.80s …", detection.score, user_prompt)
            return FirewallResponse(decision=decision)

        # ── Sanitised prompt (PII masked) ─────────────────────────────────
        sanitized = pii_result.masked_text
        decision = FirewallDecision(
            allowed=True,
            reason="pii_masked" if pii_result.entities_found else "passed",
            sanitized_prompt=sanitized,
            jailbreak_score=detection.score,
            pii_entities=pii_result.entities_found,
            latency_ms=latency_ms,
        )

        if llm_callable is None:
            return FirewallResponse(decision=decision)

        # ── Layer 2: Main LLM ──────────────────────────────────────────────
        llm_response: str = llm_callable(sanitized)

        # ── Layer 3: Judge Agent ───────────────────────────────────────────
        verdict: JudgeVerdict = self.judge_agent.evaluate(
            prompt=user_prompt,
            response=llm_response,
        )
        log.info("Judge verdict: %s (confidence=%.2f)", verdict.label, verdict.confidence)

        # ── Adaptive Memory Loop ───────────────────────────────────────────
        if self.enable_adaptive_loop and verdict.is_jailbreak:
            log.warning("Judge confirmed jailbreak – updating vector store …")
            self.vector_store.add(user_prompt, label="jailbreak")

        return FirewallResponse(
            decision=decision,
            llm_response=llm_response,
            judge_verdict=verdict,
        )
