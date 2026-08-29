"""Config hygiene tests: bool/numeric coercion, unknown-key sweep, ghost-key
messages, wizard rename/reorder, and extra_request_params merge order.

Guards the "family hygiene wave" fixes for provider-litellm:
  - drop_params/raw_debug/retry_jitter/use_streaming previously used bare
    truthiness (``bool("false")`` is ``True`` in Python).
  - the dead `debug` key (assigned, never read) now gets a targeted message
    pointing at `raw_debug`.
  - the wizard's `model` ConfigField id is renamed to `default_model`
    (canonical), with `model` kept as a read-alias.
  - `extra_request_params` merges last into litellm_kwargs.
"""

from __future__ import annotations

import logging

from amplifier_module_provider_litellm.provider import (
    LiteLLMProvider,
    _coerce_bool,
    _coerce_float,
    _coerce_int,
    _warn_unknown_config_keys,
)


class TestCoerceBool:
    def test_string_false_is_false(self):
        assert _coerce_bool("false", key="x", default=True) is False

    def test_string_true_is_true(self):
        assert _coerce_bool("true", key="x", default=False) is True

    def test_none_uses_default(self):
        assert _coerce_bool(None, key="x", default=True) is True

    def test_unrecognized_string_warns_and_defaults(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = _coerce_bool("maybe", key="drop_params", default=True)
        assert result is True
        assert "drop_params" in caplog.text


class TestCoerceNumeric:
    def test_float_from_string(self):
        assert _coerce_float("12.5", key="timeout", default=0.0) == 12.5

    def test_float_invalid_warns_and_defaults(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = _coerce_float("garbage", key="timeout", default=300.0)
        assert result == 300.0
        assert "timeout" in caplog.text

    def test_int_from_string(self):
        assert _coerce_int("5", key="max_retries", default=0) == 5


class TestUnknownConfigKeySweep:
    def test_known_keys_silent(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"default_model": "x", "priority": 1})
        assert caplog.text == ""

    def test_extra_request_params_allowlisted(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"extra_request_params": {"top_p": 0.9}})
        assert caplog.text == ""

    def test_unknown_key_warns_with_suggestion(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"tiemout": 5})
        assert "tiemout" in caplog.text
        assert "timeout" in caplog.text

    def test_debug_key_gets_targeted_message(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"debug": True})
        assert "raw_debug" in caplog.text
        assert "never wired" in caplog.text


class TestProviderConfigCoercionIntegration:
    def test_drop_params_string_false_is_false(self):
        provider = LiteLLMProvider(config={"drop_params": "false"})
        assert provider._drop_params is False

    def test_raw_debug_string_true_is_true(self):
        provider = LiteLLMProvider(config={"raw_debug": "true"})
        assert provider._raw_debug is True

    def test_debug_attribute_no_longer_set(self):
        provider = LiteLLMProvider(config={"debug": "true"})
        assert not hasattr(provider, "debug")

    def test_retry_jitter_string_false(self):
        provider = LiteLLMProvider(config={"retry_jitter": "false"})
        assert provider._retry_config.jitter == 0.0

    def test_use_streaming_string_false_is_false(self):
        provider = LiteLLMProvider(config={"use_streaming": "false"})
        assert provider.use_streaming is False

    def test_numeric_keys_from_strings(self):
        provider = LiteLLMProvider(
            config={
                "timeout": "45",
                "overloaded_delay_multiplier": "2.5",
                "max_retries": "7",
                "min_retry_delay": "0.2",
                "max_retry_delay": "20",
            }
        )
        assert provider._timeout == 45.0
        assert provider._overloaded_delay_multiplier == 2.5
        assert provider._retry_config.max_retries == 7
        assert provider._retry_config.initial_delay == 0.2
        assert provider._retry_config.max_delay == 20.0

    def test_invalid_numeric_string_defaults_instead_of_crashing(self):
        provider = LiteLLMProvider(config={"timeout": "not-a-number"})
        assert provider._timeout == 300.0

    def test_extra_request_params_stored(self):
        provider = LiteLLMProvider(config={"extra_request_params": {"top_p": 0.5}})
        assert provider.extra_request_params == {"top_p": 0.5}

    def test_extra_request_params_non_dict_ignored(self, caplog):
        with caplog.at_level(logging.WARNING):
            provider = LiteLLMProvider(config={"extra_request_params": "nope"})
        assert provider.extra_request_params == {}
        assert "extra_request_params" in caplog.text


class TestWizardFieldRename:
    def test_default_model_field_present_not_model(self):
        provider = LiteLLMProvider(config={})
        info = provider.get_info()
        ids = [f.id for f in info.config_fields]
        assert "default_model" in ids
        assert "model" not in ids

    def test_wizard_order_is_key_base_model(self):
        provider = LiteLLMProvider(config={})
        info = provider.get_info()
        ids = [f.id for f in info.config_fields]
        assert ids == ["api_key", "api_base", "default_model"]

    def test_model_still_works_as_read_alias(self):
        provider = LiteLLMProvider(config={"model": "openai/gpt-4o"})
        assert provider.default_model == "openai/gpt-4o"

    def test_default_model_takes_priority_over_model_alias(self):
        provider = LiteLLMProvider(
            config={"model": "openai/gpt-4o", "default_model": "anthropic/claude-opus-4-6"}
        )
        assert provider.default_model == "anthropic/claude-opus-4-6"
