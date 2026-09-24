"""Tests for how a Teufel Raumfeld room follows zone changes and handles playback commands."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from async_upnp_client.exceptions import UpnpError
from music_assistant_models.enums import PlaybackState
from music_assistant_models.errors import PlayerUnavailableError

from music_assistant.providers.teufel_raumfeld.player import TeufelRaumfeldPlayer
from music_assistant.providers.teufel_raumfeld.raumfeld_client import (
    RaumfeldDevice,
    RaumfeldRoom,
    RaumfeldTopology,
    RaumfeldZone,
)
from tests.common import MockProvider

ROOM = "uuid:Room-Living"
OTHER_ROOM = "uuid:Room-Kitchen"
ZONE_A = "uuid:Zone-A"
ZONE_B = "uuid:Zone-B"

DMR_DEVICE = "music_assistant.providers.teufel_raumfeld.player.DmrDevice"


def _topology(zone_udn: str | None, *, zone_rooms: list[str] | None = None) -> RaumfeldTopology:
    """
    Return a topology in which the room is part of the given zone.

    :param zone_udn: UDN of the room's zone, or None while it is not part of any zone.
    :param zone_rooms: Rooms combined into that zone, defaults to just this room.
    """
    rooms = {ROOM: RaumfeldRoom(ROOM, "Living", "uuid:Renderer-Living", "ACTIVE", zone_udn)}
    zones: dict[str, RaumfeldZone] = {}
    devices: dict[str, RaumfeldDevice] = {}
    addressable = zone_udn or ROOM
    devices[addressable] = RaumfeldDevice(addressable, f"http://10.0.0.5/{addressable}.xml", "r")
    if zone_udn:
        zones[zone_udn] = RaumfeldZone(zone_udn, zone_rooms or [ROOM])
    return RaumfeldTopology(rooms=rooms, zones=zones, devices=devices)


def _provider() -> Any:
    """Return a provider stand-in with a mocked webservice client and UPnP factory."""
    provider: Any = MockProvider("teufel_raumfeld", instance_id="teufel_raumfeld_test")
    provider.client = MagicMock()
    provider.client.get_topology = AsyncMock()
    provider.client.connect_room_to_zone = AsyncMock()
    provider.client.leave_standby = AsyncMock()
    provider.upnp_factory = MagicMock()
    provider.upnp_factory.async_create_device = AsyncMock(return_value=MagicMock())
    provider.notify_server = MagicMock()
    provider.apply_zone_change = AsyncMock()
    confirmed: set[str] = set()
    provider.is_zone_confirmed = confirmed.__contains__
    provider.confirm_zone = confirmed.add
    provider.forget_zone = MagicMock()
    return provider


def _dmr_device() -> MagicMock:
    """Return a mocked UPnP renderer that accepts every command."""
    device = MagicMock()
    device.async_subscribe_services = AsyncMock()
    device.async_unsubscribe_services = AsyncMock()
    device.async_stop = AsyncMock()
    device.async_pause = AsyncMock()
    device.can_stop = True
    device.can_pause = True
    return device


def _player(provider: Any | None = None) -> TeufelRaumfeldPlayer:
    """
    Return a player for the room.

    :param provider: The provider stand-in to attach the player to.
    """
    return TeufelRaumfeldPlayer(provider or _provider(), ROOM, "Living")


async def test_connect_follows_the_rooms_zone() -> None:
    """The player controls the renderer of whatever zone its room is currently part of."""
    provider = _provider()
    player = _player(provider)

    with patch(DMR_DEVICE, return_value=_dmr_device()):
        await player.connect(_topology(ZONE_A))

    provider.upnp_factory.async_create_device.assert_awaited_once_with(
        f"http://10.0.0.5/{ZONE_A}.xml"
    )
    assert player.device is not None


async def test_connect_to_the_same_location_is_a_no_op() -> None:
    """Reconnecting to an unchanged zone keeps the existing device and subscription."""
    provider = _provider()
    player = _player(provider)

    with patch(DMR_DEVICE, return_value=_dmr_device()):
        await player.connect(_topology(ZONE_A))
        device = player.device
        await player.connect(_topology(ZONE_A))

    assert player.device is device
    provider.upnp_factory.async_create_device.assert_awaited_once()


async def test_connect_resumes_playback_after_a_zone_swap() -> None:
    """A zone change while playing resumes the same media on the new zone renderer."""
    player = _player()
    player._apply_transport_uri = AsyncMock()  # type: ignore[method-assign]
    media = MagicMock()

    with patch(DMR_DEVICE, side_effect=[_dmr_device(), _dmr_device()]):
        await player.connect(_topology(ZONE_A))
        player._attr_playback_state = PlaybackState.PLAYING
        player._last_play_media = media
        player._last_play_url = "http://ma/stream"
        await player.connect(_topology(ZONE_B))

    player._apply_transport_uri.assert_awaited_once_with(media, "http://ma/stream")


async def test_connect_does_not_resume_after_an_explicit_stop() -> None:
    """A stopped room stays quiet when its zone changes afterwards."""
    player = _player()
    player._apply_transport_uri = AsyncMock()  # type: ignore[method-assign]

    with patch(DMR_DEVICE, side_effect=[_dmr_device(), _dmr_device()]):
        await player.connect(_topology(ZONE_A))
        player._attr_playback_state = PlaybackState.PLAYING
        player._last_play_media = MagicMock()
        player._last_play_url = "http://ma/stream"
        await player.stop()
        await player.connect(_topology(ZONE_B))

    player._apply_transport_uri.assert_not_awaited()


async def test_unreachable_zone_is_forgotten() -> None:
    """A zone that cannot be connected to is no longer treated as a real speaker group."""
    provider = _provider()
    provider.upnp_factory.async_create_device = AsyncMock(side_effect=UpnpError("gone"))
    player = _player(provider)

    await player.connect(_topology(ZONE_A))

    provider.forget_zone.assert_called_once_with(ZONE_A)
    assert player.device is None


async def test_real_solo_zone_is_created_for_an_unconfirmed_room() -> None:
    """A solo room gets an explicitly created zone, since its own renderer is silent."""
    provider = _provider()
    provider.client.get_topology = AsyncMock(return_value=_topology(ZONE_B))
    player = _player(provider)

    with patch("music_assistant.providers.teufel_raumfeld.player.asyncio.sleep"):
        udn, _topo = await player._ensure_real_solo_zone(ROOM, _topology(None))

    provider.client.connect_room_to_zone.assert_awaited_once_with(ROOM)
    assert udn == ZONE_B
    assert provider.is_zone_confirmed(ZONE_B)


@pytest.mark.parametrize(
    ("confirmed", "zone_rooms"),
    [(True, [ROOM]), (False, [ROOM, OTHER_ROOM])],
    ids=["confirmed_zone", "multi_room_zone"],
)
async def test_real_solo_zone_leaves_real_zones_alone(
    confirmed: bool, zone_rooms: list[str]
) -> None:
    """A confirmed or multi-room zone is never recreated, which would stop its playback."""
    provider = _provider()
    if confirmed:
        provider.confirm_zone(ZONE_A)
    player = _player(provider)

    udn, _topo = await player._ensure_real_solo_zone(
        ZONE_A, _topology(ZONE_A, zone_rooms=zone_rooms)
    )

    provider.client.connect_room_to_zone.assert_not_awaited()
    assert udn == ZONE_A


async def test_pause_stops_instead_of_pausing() -> None:
    """Pause stops the renderer, so resuming starts a fresh stream session."""
    player = _player()
    player.device = _dmr_device()
    player._last_play_media = MagicMock()

    await player.pause()

    player.device.async_stop.assert_awaited_once()
    player.device.async_pause.assert_not_awaited()
    assert player._last_play_media is None


async def test_set_members_extends_the_rooms_current_zone() -> None:
    """Grouping targets the leader's current zone and makes it the MA sync leader."""
    provider = _provider()
    provider.client.get_topology = AsyncMock(return_value=_topology(ZONE_A))
    player = _player(provider)

    await player.set_members(player_ids_to_add=[OTHER_ROOM])

    provider.apply_zone_change.assert_awaited_once_with(
        rooms_to_drop=[],
        rooms_to_group=sorted([ROOM, OTHER_ROOM]),
        target_zone_udn=ZONE_A,
        prefer_leader=ROOM,
    )


async def test_poll_of_a_standby_room_without_renderer_is_not_an_error() -> None:
    """A room in standby legitimately has no renderer to poll."""
    provider = _provider()
    provider.topology = RaumfeldTopology()
    player = _player(provider)
    player._attr_powered = False

    await player.poll()

    assert player.device is None


@pytest.mark.parametrize("powered", [True, None])
async def test_poll_of_an_awake_room_without_renderer_is_unavailable(
    powered: bool | None,
) -> None:
    """A room that should have a renderer but has none is reported unavailable."""
    provider = _provider()
    provider.topology = RaumfeldTopology()
    player = _player(provider)
    player._attr_powered = powered

    with pytest.raises(PlayerUnavailableError):
        await player.poll()
