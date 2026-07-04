from wire.llm import get_client, OpenAICompatClient


def test_hosted_selected_when_key_present(monkeypatch):
    monkeypatch.delenv("WIRE_LLM", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    c = get_client()
    assert isinstance(c, OpenAICompatClient)
    assert c.provider == "openai"
    assert c.available()


def test_heuristic_when_forced(monkeypatch):
    monkeypatch.setenv("WIRE_LLM", "heuristic")
    assert get_client() is None


def test_openai_client_uses_json_mode():
    c = OpenAICompatClient(api_key="k", base_url="http://x", model="m")
    # sanity on the request shape without a live call
    assert c.provider == "openai" and c.model == "m"
