from typing import Any

import pytest

from kitchenwatch.edge.http_receiver import HttpReceiver


def test_edge_http_receiver(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], **kwargs: Any) -> Any:
        called["url"] = url
        called["json"] = json

        class Resp:
            status_code = 200

            def raise_for_status(self) -> None:
                pass

        return Resp()

    monkeypatch.setattr("requests.post", fake_post)
    receiver = HttpReceiver(edge_url="http://mock:8000/ingest")
    receiver.ingest({"timestamp_ns": 123})
    assert called["url"].endswith("/ingest")
    assert receiver.sent_count == 1
