"""Tests for the room info the Teufel Raumfeld player takes from the topology."""

from __future__ import annotations

import pytest
from music_assistant_models.enums import PlayerFeature

from music_assistant.providers.teufel_raumfeld.player import TeufelRaumfeldPlayer
from music_assistant.providers.teufel_raumfeld.raumfeld_client import RaumfeldRoom
from tests.common import MockProvider

ROOM_UDN = "uuid:Room-Workshop"
RENDERER_UDN = "uuid:Renderer-Workshop"


def _player() -> TeufelRaumfeldPlayer:
    """Return a player for a single room."""
    provider = MockProvider("teufel_raumfeld", instance_id="teufel_raumfeld_test")
    return TeufelRaumfeldPlayer(provider, ROOM_UDN, "Workshop")  # type: ignore[arg-type]


def _room(power_state: str | None) -> RaumfeldRoom:
    """
    Return the topology entry of the room, with the given power state.

    :param power_state: The room's powerState, or None when the host reports none.
    """
    return RaumfeldRoom(
        udn=ROOM_UDN, name="Werkstatt", renderer_udn=RENDERER_UDN, power_state=power_state
    )


@pytest.mark.parametrize(
    ("power_state", "powered"),
    [("ACTIVE", True), ("AUTOMATIC_STANDBY", False), ("MANUAL_STANDBY", False)],
)
def test_room_with_power_state_supports_power(power_state: str, powered: bool) -> None:
    """A room that reports a power state gets the power feature and its state."""
    player = _player()

    player.update_room_info(_room(power_state))

    assert PlayerFeature.POWER in player.supported_features
    assert player._attr_powered is powered
    assert player._attr_name == "Werkstatt"


def test_room_without_power_state_does_not_support_power() -> None:
    """A room without a power state (a speaker without standby) has no power feature."""
    player = _player()
    player.update_room_info(_room("ACTIVE"))

    player.update_room_info(_room(None))

    assert PlayerFeature.POWER not in player.supported_features
    assert player._attr_powered is None
