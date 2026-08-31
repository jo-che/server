"""
Native async client for the Raumfeld host webservice (topology/zone control only).

Playback control itself (AVTransport/RenderingControl) happens directly against the
UPnP renderer locations this client resolves; see `music_assistant.providers.teufel_raumfeld.player`.
"""

from __future__ import annotations

from .exceptions import (
    RaumfeldConnectionError,
    RaumfeldError,
    RaumfeldInvalidHostError,
    RaumfeldNotFoundError,
)
from .models import RaumfeldDevice, RaumfeldRoom, RaumfeldTopology, RaumfeldZone
from .webservice import DEFAULT_PORT, RaumfeldWebserviceClient

__all__ = [
    "DEFAULT_PORT",
    "RaumfeldConnectionError",
    "RaumfeldDevice",
    "RaumfeldError",
    "RaumfeldInvalidHostError",
    "RaumfeldNotFoundError",
    "RaumfeldRoom",
    "RaumfeldTopology",
    "RaumfeldWebserviceClient",
    "RaumfeldZone",
]
