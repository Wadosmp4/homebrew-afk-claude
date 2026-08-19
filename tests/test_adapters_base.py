"""Multi-Agent Adapter plan (009), U1: pure interface-extraction unit - no
runtime behavior changes to either existing adapter, so conformance is the
only thing to verify here (see AdapterProtocol's own docstring)."""
from __future__ import annotations

from companion.adapters.base import AdapterProtocol
from companion.adapters.observe_adapter import ObserveAdapter
from companion.adapters.sdk_adapter import SDKAdapter


def test_sdk_adapter_conforms_to_adapter_protocol():
    assert isinstance(SDKAdapter(), AdapterProtocol)


def test_observe_adapter_conforms_to_adapter_protocol():
    assert isinstance(ObserveAdapter(), AdapterProtocol)
