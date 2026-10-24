from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.utils.rate_limit import call_with_rate_limit_retry


class FakeRateLimitError(Exception):
    status_code = 429

    def __init__(self, reset_header="0.25s", message="rate limited"):
        super().__init__(message)
        self.response = SimpleNamespace(
            headers={"x-ratelimit-reset-tokens": reset_header}
        )


class TestRateLimitRetry(unittest.TestCase):
    def test_retries_429_using_token_reset_header_and_reports_minute_limit(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise FakeRateLimitError(message="tokens per minute (TPM)")
            return "success"

        with patch("src.utils.rate_limit.time.sleep") as sleep:
            result = call_with_rate_limit_retry(operation, "test operation")

        self.assertEqual(result, "success")
        self.assertEqual(attempts, 2)
        sleep.assert_called_once_with(0.25)

    def test_does_not_retry_daily_rate_limit(self):
        operation = unittest.mock.Mock(
            side_effect=FakeRateLimitError(message="requests per day (RPD)")
        )

        with patch("src.utils.rate_limit.time.sleep") as sleep:
            with self.assertLogs("rate_limit", level="ERROR") as logs:
                with self.assertRaises(FakeRateLimitError):
                    call_with_rate_limit_retry(operation, "test operation")

        operation.assert_called_once()
        sleep.assert_not_called()
        self.assertIn("per-day", logs.output[0])

    def test_does_not_retry_non_rate_limit_errors(self):
        operation = unittest.mock.Mock(side_effect=RuntimeError("failure"))

        with patch("src.utils.rate_limit.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "failure"):
                call_with_rate_limit_retry(operation, "test operation")

        operation.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()