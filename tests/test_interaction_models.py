"""Tests for InteractionRequest/InteractionResponse Pydantic models."""

from agentic_research.models.interaction import (
    InteractionOption,
    InteractionRequest,
    InteractionResponse,
)


class TestInteractionOption:
    def test_basic(self):
        opt = InteractionOption(label="Option A", value=42, score=0.85)
        assert opt.label == "Option A"
        assert opt.value == 42
        assert opt.score == 0.85

    def test_default_score(self):
        opt = InteractionOption(label="x", value="v")
        assert opt.score == 0.0


class TestInteractionRequest:
    def test_serialization_roundtrip(self):
        req = InteractionRequest(
            type="select",
            prompt="Pick one",
            options=[
                InteractionOption(label="A", value=0, score=0.9),
                InteractionOption(label="B", value=1, score=0.7),
            ],
            default_value=0,
        )
        data = req.model_dump()
        restored = InteractionRequest.model_validate(data)
        assert restored.type == "select"
        assert restored.prompt == "Pick one"
        assert len(restored.options) == 2
        assert restored.options[0].label == "A"
        assert restored.default_value == 0

    def test_json_roundtrip(self):
        req = InteractionRequest(
            type="select",
            prompt="Choose",
            options=[InteractionOption(label="X", value="x", score=1.0)],
        )
        json_str = req.model_dump_json()
        restored = InteractionRequest.model_validate_json(json_str)
        assert restored.prompt == "Choose"
        assert restored.options[0].value == "x"

    def test_empty_options(self):
        req = InteractionRequest(type="select", prompt="Nothing here")
        assert req.options == []
        assert req.default_value is None

    def test_default_value_none(self):
        req = InteractionRequest(type="select", prompt="test")
        assert req.default_value is None


class TestInteractionResponse:
    def test_selected_value(self):
        resp = InteractionResponse(selected_value=2)
        assert resp.selected_value == 2
        assert resp.aborted is False

    def test_aborted(self):
        resp = InteractionResponse(aborted=True)
        assert resp.aborted is True
        assert resp.selected_value is None

    def test_serialization_roundtrip(self):
        resp = InteractionResponse(selected_value="picked", aborted=False)
        data = resp.model_dump()
        restored = InteractionResponse.model_validate(data)
        assert restored.selected_value == "picked"
        assert restored.aborted is False

    def test_json_roundtrip(self):
        resp = InteractionResponse(selected_value=42, aborted=True)
        json_str = resp.model_dump_json()
        restored = InteractionResponse.model_validate_json(json_str)
        assert restored.selected_value == 42
        assert restored.aborted is True
