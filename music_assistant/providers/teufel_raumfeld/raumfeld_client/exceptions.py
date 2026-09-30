"""Exceptions raised by the Raumfeld host webservice client."""

from __future__ import annotations


class RaumfeldError(Exception):
    """Base exception for all Raumfeld webservice client errors."""


class RaumfeldConnectionError(RaumfeldError):
    """Raised when the Raumfeld host webservice can not be reached."""


class RaumfeldInvalidHostError(RaumfeldError):
    """Raised when the configured host does not respond like a Raumfeld host webservice."""


class RaumfeldCommandError(RaumfeldError):
    """Raised when the host webservice rejects a command, e.g. for an unknown room or zone."""
