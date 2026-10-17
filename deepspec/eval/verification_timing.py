"""Opt-in synchronized wall-clock timing for speculative verification.

This profiler deliberately synchronizes the accelerator at phase boundaries. It
is intended for small diagnostic experiments, not production throughput runs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class VerificationPhaseTimer:
    device: torch.device
    target_model_forward_seconds: float = 0.0
    acceptance_rule_seconds: float = 0.0
    verification_call_count: int = 0
    profiled_generation_seconds: float = 0.0
    _phase: str | None = field(default=None, init=False, repr=False)
    _started_at: float | None = field(default=None, init=False, repr=False)
    _generation_started_at: float | None = field(default=None, init=False, repr=False)

    def _synchronize(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def start_generation(self) -> None:
        if self._generation_started_at is not None:
            raise RuntimeError("verification timing generation interval already started")
        self._synchronize()
        self._generation_started_at = time.perf_counter()

    def stop_generation(self) -> None:
        if self._generation_started_at is None or self._phase is not None:
            raise RuntimeError("invalid verification timing generation boundary")
        self._synchronize()
        self.profiled_generation_seconds += time.perf_counter() - self._generation_started_at
        self._generation_started_at = None

    def start_target_model(self) -> None:
        self._start("target_model")

    def stop_target_model(self, *, count_as_call: bool = False) -> None:
        self.target_model_forward_seconds += self._stop("target_model")



        if count_as_call:
            self.verification_call_count += 1

    def start_acceptance_rule(self) -> None:
        self._start("acceptance_rule")

    def stop_acceptance_rule(self) -> None:
        self.acceptance_rule_seconds += self._stop("acceptance_rule")
        self.verification_call_count += 1

    def _start(self, phase: str) -> None:
        if self._phase is not None:
            raise RuntimeError(f"verification timing phase {self._phase!r} is still active")
        self._synchronize()
        self._phase = phase
        self._started_at = time.perf_counter()

    def _stop(self, phase: str) -> float:
        if self._phase != phase or self._started_at is None:
            raise RuntimeError(f"cannot stop verification timing phase {phase!r}")
        self._synchronize()
        elapsed = time.perf_counter() - self._started_at
        self._phase = None
        self._started_at = None
        return elapsed

    def summary(self) -> dict[str, int | float | str]:
        if self._phase is not None or self._generation_started_at is not None:
            raise RuntimeError("verification timing summary requested before all phases stopped")
        total = self.target_model_forward_seconds + self.acceptance_rule_seconds
        return {
            "measurement": "accelerator_synchronized_wall_clock",
            "verification_call_count": int(self.verification_call_count),
            "target_model_forward_seconds": float(self.target_model_forward_seconds),
            "acceptance_rule_seconds": float(self.acceptance_rule_seconds),
            "total_verification_seconds": float(total),
            "profiled_generation_seconds": float(self.profiled_generation_seconds),
        }
