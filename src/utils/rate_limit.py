import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, TypeVar

from src.logger import get_logger

logger = get_logger("rate_limit")

T = TypeVar("T")


def _parse_delay(value: str) -> float:
    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        pass

    units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", value.lower())
    if parts and "".join(number + unit for number, unit in parts) == value.lower().replace(" ", ""):
        return sum(float(number) * units[unit] for number, unit in parts)
    return 0.0


def _error_message(error: Exception) -> str:
    body = getattr(error, "body", None)
    return f"{error} {body or ''}".strip()


def _rate_limit_scope(error: Exception) -> str:
    message = _error_message(error).lower()
    if re.search(
        r"\b(?:tokens?|requests?)\s+(?:per|/)\s*day\b|\b(?:tpd|rpd)\b|\bdaily\b",
        message,
    ):
        return "per-day"
    if re.search(
        r"\b(?:tokens?|requests?)\s+(?:per|/)\s*minute\b|\b(?:tpm|rpm)\b",
        message,
    ):
        return "per-minute"

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    header_names = {str(key).lower() for key in headers}
    if "x-ratelimit-limit-tokens" in header_names or "x-ratelimit-limit-requests" in header_names:
        return "per-minute"
    return "unknown"


def _retry_delay(error: Exception, retry_number: int) -> tuple[float, str]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    normalized_headers = {str(key).lower(): str(value) for key, value in headers.items()}

    for header in (
        "retry-after",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-reset-requests",
    ):
        value = normalized_headers.get(header)
        if value:
            delay = _parse_delay(value)
            if delay > 0:
                return delay, header

    return min(2.0 * (2 ** (retry_number - 1)), 60.0), "exponential backoff"


def call_with_rate_limit_retry(
    operation: Callable[[], T],
    operation_name: str,
    max_retries: int = 5,
) -> T:
    """Retry rate-limited API calls, honoring provider reset headers when available."""
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as error:
            if getattr(error, "status_code", None) != 429:
                raise

            scope = _rate_limit_scope(error)
            if scope == "per-day":
                logger.error(
                    "%s hit a per-day rate limit; automatic retry skipped. Provider error: %s",
                    operation_name,
                    _error_message(error),
                )
                raise

            if attempt >= max_retries:
                logger.error(
                    "%s exhausted %d retries for a %s rate limit. Provider error: %s",
                    operation_name,
                    max_retries,
                    scope,
                    _error_message(error),
                )
                raise

            retry_number = attempt + 1
            delay, delay_source = _retry_delay(error, retry_number)
            logger.warning(
                "%s hit HTTP 429 (%s limit); retry %d/%d in %.2f seconds "
                "using %s. Provider error: %s",
                operation_name,
                scope,
                retry_number,
                max_retries,
                delay,
                delay_source,
                _error_message(error),
            )
            time.sleep(delay)

    raise RuntimeError("Rate-limit retry loop exited unexpectedly.")