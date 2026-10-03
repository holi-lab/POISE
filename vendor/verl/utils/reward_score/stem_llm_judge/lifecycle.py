"""Optional lifecycle control for a local vLLM STEM verifier.

The controller is disabled by default.  It is intended for a verifier that is
owned exclusively by the current training job and was launched with vLLM's
``--enable-sleep-mode`` option and ``VLLM_SERVER_DEV_MODE=1``.
"""

import argparse
import os
import time
from collections.abc import Iterable

import requests


_CONTROL_ENABLED_ENV = "STEM_VERIFIER_LIFECYCLE_CONTROL"
_CONTROL_URL_ENV = "STEM_VERIFIER_LIFECYCLE_URL"
_CONTROL_TIMEOUT_ENV = "STEM_VERIFIER_LIFECYCLE_TIMEOUT_SECONDS"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class StemVerifierLifecycle:
    """Sleep and wake an exclusively owned vLLM verifier."""

    def __init__(
        self,
        *,
        enabled: bool,
        base_url: str | None,
        transition_timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 0.2,
        request_timeout_seconds: float = 10.0,
    ) -> None:
        self.enabled = enabled
        self.base_url = base_url.rstrip("/") if base_url else None
        self.transition_timeout_seconds = transition_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.request_timeout_seconds = request_timeout_seconds

        if self.enabled and not self.base_url:
            raise ValueError(f"{_CONTROL_URL_ENV} must be set when {_CONTROL_ENABLED_ENV}=1")

    @classmethod
    def from_env(cls) -> "StemVerifierLifecycle":
        return cls(
            enabled=_env_flag(_CONTROL_ENABLED_ENV),
            base_url=os.getenv(_CONTROL_URL_ENV),
            transition_timeout_seconds=float(os.getenv(_CONTROL_TIMEOUT_ENV, "120")),
            # The sleep/wake control calls hit a vLLM server that may be busy
            # finishing a batch. A 10 s read timeout is enough in the common
            # case but propagates out of the reward manager and kills the run
            # when it is not, so keep the deadline configurable.
            request_timeout_seconds=float(
                os.getenv("STEM_VERIFIER_REQUEST_TIMEOUT_SECONDS", "10")
            ),
        )

    @staticmethod
    def batch_uses_llm_judge(data_sources: Iterable[object]) -> bool:
        return any(str(data_source).startswith("stem_web") for data_source in data_sources)

    def should_control(self, data_sources: Iterable[object]) -> bool:
        return self.enabled and self.batch_uses_llm_judge(data_sources)

    def _is_sleeping(self) -> bool:
        assert self.base_url is not None
        response = requests.get(
            f"{self.base_url}/is_sleeping",
            timeout=self.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if "is_sleeping" not in payload:
            raise RuntimeError(f"Malformed verifier lifecycle response: {payload!r}")
        return bool(payload["is_sleeping"])

    def _wait_for_state(self, expected_sleeping: bool) -> None:
        deadline = time.monotonic() + self.transition_timeout_seconds
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            try:
                if self._is_sleeping() is expected_sleeping:
                    return
                last_error = None
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
            time.sleep(self.poll_interval_seconds)

        state_name = "sleeping" if expected_sleeping else "awake"
        detail = f"; last error: {last_error}" if last_error is not None else ""
        raise TimeoutError(
            f"Timed out after {self.transition_timeout_seconds:.1f}s waiting for "
            f"the STEM verifier to become {state_name}{detail}"
        )

    def _transition(self, *, sleep: bool) -> bool:
        if not self.enabled:
            return False

        assert self.base_url is not None
        if self._is_sleeping() is sleep:
            return False

        endpoint = "/sleep?level=1" if sleep else "/wake_up"
        started_at = time.monotonic()
        response = requests.post(
            f"{self.base_url}{endpoint}",
            timeout=self.request_timeout_seconds,
        )
        response.raise_for_status()
        self._wait_for_state(sleep)

        action = "sleep" if sleep else "wake_up"
        elapsed = time.monotonic() - started_at
        print(f"[stem-verifier-lifecycle] {action} completed in {elapsed:.2f}s", flush=True)
        return True

    def sleep(self) -> bool:
        """Offload weights to CPU and release the verifier's GPU allocations."""
        return self._transition(sleep=True)

    def wake_up(self) -> bool:
        """Restore the verifier on GPU before an LLM-judge reward call."""
        return self._transition(sleep=False)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("sleep", "wake", "status"))
    parser.add_argument("--url", default=os.getenv(_CONTROL_URL_ENV))
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv(_CONTROL_TIMEOUT_ENV, "120")),
    )
    args = parser.parse_args()

    controller = StemVerifierLifecycle(
        enabled=True,
        base_url=args.url,
        transition_timeout_seconds=args.timeout,
    )
    if args.action == "sleep":
        controller.sleep()
    elif args.action == "wake":
        controller.wake_up()
    else:
        print("sleeping" if controller._is_sleeping() else "awake")


if __name__ == "__main__":
    _main()
