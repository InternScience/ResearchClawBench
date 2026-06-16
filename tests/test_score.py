from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from evaluation.score import JudgeLLMAgent, _completion_token_limit_kwargs


class FakeChatCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return object()


class FakeClient:
    def __init__(self):
        self.chat = type("FakeChat", (), {})()
        self.chat.completions = FakeChatCompletions()


class ScoreTokenLimitTests(TestCase):
    def test_completion_token_limit_uses_new_parameter_for_gpt5(self):
        self.assertEqual(
            _completion_token_limit_kwargs("gpt-5.1", 500),
            {"max_completion_tokens": 500},
        )

    def test_completion_token_limit_keeps_legacy_parameter_for_other_models(self):
        self.assertEqual(
            _completion_token_limit_kwargs("gpt-4.1-mini", 500),
            {"max_tokens": 500},
        )
        self.assertEqual(_completion_token_limit_kwargs("gpt-5.1", None), {})

    def test_judge_agent_sends_max_completion_tokens_for_gpt5_chat_completions(self):
        agent = JudgeLLMAgent(
            api_key="test-key",
            api_base="http://example.invalid/v1",
            model_version="gpt-5.1",
            max_tokens=500,
        )
        fake_client = FakeClient()
        agent.client = fake_client

        with patch("evaluation.score.extract_text_outputs", return_value=["ok"]):
            self.assertEqual(agent._llm_api_impl("prompt"), ["ok"])

        sent = fake_client.chat.completions.kwargs
        self.assertEqual(sent["max_completion_tokens"], 500)
        self.assertNotIn("max_tokens", sent)
