"""Test-wide guardrails: no provider connections or dotenv credential loading."""

import socket

import dotenv
import pytest


@pytest.fixture(autouse=True)
def block_network_and_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_connect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("tests must not open network connections")

    def reject_connect_ex(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", reject_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_connect_ex)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *_args, **_kwargs: False)
