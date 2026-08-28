import pytest

from companion.rate_limit_estimator import RateLimitEstimator


def test_estimate_is_none_with_no_real_reading_yet():
    estimator = RateLimitEstimator()
    assert estimator.estimate("five_hour") is None


def test_estimate_is_none_after_a_single_real_reading_with_nothing_to_calibrate_against():
    estimator = RateLimitEstimator()
    estimator.record_real_reading("five_hour", 0.40)
    assert estimator.estimate("five_hour") is None
    assert estimator.calibrated_types() == []


def test_two_real_readings_with_new_tokens_between_them_calibrate_a_slope():
    estimator = RateLimitEstimator()
    estimator.record_real_reading("five_hour", 0.40)
    estimator.record_turn_usage({"claude-opus-4-7": {"inputTokens": 1000, "outputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0}})
    # 1000 tokens moved utilization from 0.40 to 0.50 - 10000 tokens per
    # 1.0 (100%) of utilization.
    estimator.record_real_reading("five_hour", 0.50)

    assert estimator.calibrated_types() == ["five_hour"]
    # No new tokens since the second real reading - estimate matches it
    # exactly (token_delta <= 0 short-circuits before any extrapolation).
    assert estimator.estimate("five_hour") == 0.50


def test_estimate_extrapolates_forward_from_the_calibrated_slope_as_new_tokens_arrive():
    estimator = RateLimitEstimator()
    estimator.record_real_reading("five_hour", 0.40)
    estimator.record_turn_usage({"m": {"inputTokens": 1000, "outputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0}})
    estimator.record_real_reading("five_hour", 0.50)  # calibrates 10,000 tokens/1.0 utilization

    estimator.record_turn_usage({"m": {"inputTokens": 500, "outputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0}})

    # 500 more tokens / 10,000 tokens-per-utilization = 0.05 more utilization.
    assert estimator.estimate("five_hour") == pytest.approx(0.55)


def test_record_turn_usage_sums_across_multiple_models_and_all_four_token_fields():
    estimator = RateLimitEstimator()
    estimator.record_real_reading("five_hour", 0.10)
    estimator.record_turn_usage(
        {
            "claude-opus-4-7": {
                "inputTokens": 100,
                "outputTokens": 50,
                "cacheReadInputTokens": 25,
                "cacheCreationInputTokens": 25,
            },
            "claude-sonnet-4-7": {
                "inputTokens": 100,
                "outputTokens": 100,
                "cacheReadInputTokens": 0,
                "cacheCreationInputTokens": 0,
            },
        }
    )
    estimator.record_real_reading("five_hour", 0.20)

    # 100+50+25+25 + 100+100+0+0 = 400 tokens moved utilization by 0.10 ->
    # 4,000 tokens per 1.0 utilization. Confirm via a further extrapolation.
    estimator.record_turn_usage({"m": {"inputTokens": 400, "outputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0}})
    assert estimator.estimate("five_hour") == pytest.approx(0.30)


def test_record_turn_usage_with_none_or_empty_is_a_noop():
    estimator = RateLimitEstimator()
    estimator.record_turn_usage(None)
    estimator.record_turn_usage({})
    estimator.record_real_reading("five_hour", 0.10)
    estimator.record_real_reading("five_hour", 0.20)
    # No tokens were ever recorded, so utilization_delta > 0 but
    # token_delta is 0 - no slope calibrates.
    assert estimator.calibrated_types() == []


def test_record_task_usage_only_counts_the_delta_for_a_repeated_task_id():
    estimator = RateLimitEstimator()
    estimator.record_real_reading("five_hour", 0.10)

    # A task's own usage is cumulative per task_id (TaskProgressMessage's
    # own running total) - two progress ticks for the SAME task must not
    # double-count the first reading's tokens.
    estimator.record_task_usage("task-1", {"total_tokens": 100, "tool_uses": 1, "duration_ms": 500})
    estimator.record_task_usage("task-1", {"total_tokens": 300, "tool_uses": 2, "duration_ms": 1000})

    estimator.record_real_reading("five_hour", 0.13)
    # Only 300 total tokens ever attributed to task-1 (not 100 + 300 = 400)
    # moved utilization by 0.03 -> 10,000 tokens per 1.0 utilization.
    estimator.record_turn_usage({"m": {"inputTokens": 1000, "outputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0}})
    assert estimator.estimate("five_hour") == pytest.approx(0.23)


def test_record_task_usage_with_no_total_tokens_key_is_a_noop():
    estimator = RateLimitEstimator()
    estimator.record_task_usage("task-1", {"tool_uses": 1, "duration_ms": 500})
    estimator.record_task_usage("task-1", None)
    estimator.record_real_reading("five_hour", 0.10)
    estimator.record_real_reading("five_hour", 0.20)
    assert estimator.calibrated_types() == []


def test_a_window_reset_does_not_poison_the_slope_with_a_negative_utilization_delta():
    """Regression: utilization dropping (a real window reset) while the
    estimator's own token counter keeps climbing must not be read as "a
    huge amount of tokens produced a huge negative slope" - the stale
    pre-reset slope is kept instead, and the new baseline reading is still
    recorded so a future real pair recalibrates cleanly."""
    estimator = RateLimitEstimator()
    estimator.record_real_reading("five_hour", 0.40)
    estimator.record_turn_usage({"m": {"inputTokens": 1000, "outputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0}})
    estimator.record_real_reading("five_hour", 0.50)  # calibrates 10,000 tokens/1.0

    # Window resets - utilization drops even though more tokens have been
    # consumed overall.
    estimator.record_turn_usage({"m": {"inputTokens": 200, "outputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0}})
    estimator.record_real_reading("five_hour", 0.02)

    # The stale slope (10,000 tokens/1.0) is still there, extrapolating
    # forward from the new (post-reset) baseline of 0.02.
    estimator.record_turn_usage({"m": {"inputTokens": 100, "outputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0}})
    assert estimator.estimate("five_hour") == pytest.approx(0.03)


def test_estimate_clamps_at_1_0():
    estimator = RateLimitEstimator()
    estimator.record_real_reading("five_hour", 0.90)
    estimator.record_turn_usage({"m": {"inputTokens": 100, "outputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0}})
    estimator.record_real_reading("five_hour", 0.95)  # 100 tokens / 0.05 = 2,000 tokens per 1.0

    estimator.record_turn_usage({"m": {"inputTokens": 100_000, "outputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0}})
    assert estimator.estimate("five_hour") == 1.0


def test_each_rate_limit_type_calibrates_independently():
    estimator = RateLimitEstimator()
    estimator.record_real_reading("five_hour", 0.40)
    estimator.record_real_reading("seven_day", 0.10)
    estimator.record_turn_usage({"m": {"inputTokens": 1000, "outputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0}})
    estimator.record_real_reading("five_hour", 0.50)
    # seven_day never got a second reading - stays uncalibrated.
    assert set(estimator.calibrated_types()) == {"five_hour"}
    assert estimator.estimate("seven_day") is None
