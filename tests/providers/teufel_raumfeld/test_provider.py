"""Tests for the zone bookkeeping and lifecycle of the Teufel Raumfeld provider."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from music_assistant.helpers.webserver import Webserver
from music_assistant.providers.teufel_raumfeld.helpers import RaumfeldNotifyServer
from music_assistant.providers.teufel_raumfeld.provider import TeufelRaumfeldPlayerProvider
from music_assistant.providers.teufel_raumfeld.raumfeld_client import (
    RaumfeldDevice,
    RaumfeldRoom,
    RaumfeldTopology,
    RaumfeldZone,
)

LIVING = "uuid:Room-Living"
KITCHEN = "uuid:Room-Kitchen"
BEDROOM = "uuid:Room-Bedroom"
ZONE = "uuid:Zone-Living-Kitchen"
RENDERER = "uuid:Renderer-Living"


def _provider() -> Any:
    """Return a provider with just enough state for its zone bookkeeping."""
    provider = cast("Any", TeufelRaumfeldPlayerProvider.__new__(TeufelRaumfeldPlayerProvider))
    provider.mass = MagicMock()
    provider.mass.players.unregister = AsyncMock()
    provider.client = MagicMock()
    provider.client.drop_room = AsyncMock()
    provider.client.connect_rooms_to_zone = AsyncMock()
    provider._players = {}
    provider._confirmed_zones = set()
    provider._device_models = {}
    provider._reconcile_lock = asyncio.Lock()
    provider.discover_players = AsyncMock()
    return provider


def _player(group_members: list[str] | None = None) -> MagicMock:
    """
    Return a registered player stand-in.

    :param group_members: The MA sync group members the player currently claims.
    """
    player = MagicMock()
    player.player_id = "unused"
    player._attr_group_members = group_members or []
    player.refresh_from_topology = AsyncMock()
    return player


def _topology(*, synced: bool, with_locations: bool = True) -> RaumfeldTopology:
    """
    Return a topology with living room, kitchen and bedroom.

    :param synced: Whether living room and kitchen are combined into one zone.
    :param with_locations: Whether every room already resolves a device location.
    """
    zone_udn = ZONE if synced else None
    rooms = {
        LIVING: RaumfeldRoom(LIVING, "Living", RENDERER, "ACTIVE", zone_udn),
        KITCHEN: RaumfeldRoom(KITCHEN, "Kitchen", "uuid:Renderer-Kitchen", "ACTIVE", zone_udn),
        BEDROOM: RaumfeldRoom(BEDROOM, "Bedroom", "uuid:Renderer-Bedroom", "ACTIVE"),
    }
    zones = {ZONE: RaumfeldZone(ZONE, [LIVING, KITCHEN])} if synced else {}
    devices: dict[str, RaumfeldDevice] = {}
    if with_locations:
        for udn in (ZONE, LIVING, KITCHEN, BEDROOM, RENDERER):
            devices[udn] = RaumfeldDevice(udn, f"http://10.0.0.5/{udn}.xml", "renderer")
    return RaumfeldTopology(rooms=rooms, zones=zones, devices=devices)


def test_pick_leader_prefers_the_requested_room() -> None:
    """The room a grouping command was issued on becomes the leader."""
    provider = _provider()

    assert provider._pick_leader([LIVING, KITCHEN], prefer=KITCHEN) == KITCHEN


def test_pick_leader_ignores_a_preferred_room_outside_the_zone() -> None:
    """A preferred room that is not part of the zone does not become its leader."""
    provider = _provider()

    assert provider._pick_leader([LIVING, KITCHEN], prefer=BEDROOM) == min(LIVING, KITCHEN)


def test_pick_leader_keeps_the_established_leader() -> None:
    """Without a preference, the room already leading exactly this zone stays leader."""
    provider = _provider()
    provider._players = {LIVING: _player(), KITCHEN: _player([KITCHEN, LIVING])}

    assert provider._pick_leader([LIVING, KITCHEN]) == KITCHEN


def test_pick_leader_falls_back_to_the_lowest_udn() -> None:
    """A zone formed outside MA gets a stable, deterministic leader."""
    provider = _provider()

    assert provider._pick_leader([LIVING, KITCHEN]) == min(LIVING, KITCHEN)


async def test_reconcile_gives_the_leader_all_zone_members() -> None:
    """The leader lists every room of its zone; the other members and solo rooms none."""
    provider = _provider()
    provider._players = {udn: _player() for udn in (LIVING, KITCHEN, BEDROOM)}
    provider.topology = _topology(synced=True)

    await provider._reconcile(prefer_leader=LIVING)

    assert provider._players[LIVING]._attr_group_members == [LIVING, KITCHEN]
    assert provider._players[KITCHEN]._attr_group_members == []
    assert provider._players[BEDROOM]._attr_group_members == []
    for player in provider._players.values():
        player.refresh_from_topology.assert_awaited_once_with(provider.topology)


async def test_reconcile_confirms_multi_room_zones() -> None:
    """A zone seen with more than one room is known to be a real speaker group."""
    provider = _provider()
    provider._players = {udn: _player() for udn in (LIVING, KITCHEN, BEDROOM)}
    provider.topology = _topology(synced=True)

    await provider._reconcile()

    assert provider.is_zone_confirmed(ZONE)


async def test_reconcile_unregisters_rooms_that_disappeared() -> None:
    """A room no longer in the topology is unregistered and forgotten."""
    provider = _provider()
    gone = _player()
    provider._players = {udn: _player() for udn in (LIVING, KITCHEN, BEDROOM)}
    provider._players["uuid:Room-Gone"] = gone
    provider.topology = _topology(synced=False)

    await provider._reconcile()

    provider.mass.players.unregister.assert_awaited_once_with(gone.player_id)
    assert "uuid:Room-Gone" not in provider._players


async def test_apply_zone_change_targets_the_existing_zone() -> None:
    """Rooms are dropped first, then grouped into the given zone, then reconciled."""
    provider = _provider()
    provider._settle_and_reconcile = AsyncMock()

    await provider.apply_zone_change(
        rooms_to_drop=[BEDROOM],
        rooms_to_group=[LIVING, KITCHEN],
        target_zone_udn=ZONE,
        prefer_leader=LIVING,
    )

    provider.client.drop_room.assert_awaited_once_with(BEDROOM)
    provider.client.connect_rooms_to_zone.assert_awaited_once_with([LIVING, KITCHEN], zone_udn=ZONE)
    provider._settle_and_reconcile.assert_awaited_once_with(prefer_leader=LIVING)


async def test_apply_zone_change_does_not_group_a_single_room() -> None:
    """Grouping a single room would destroy its zone for nothing, so it is skipped."""
    provider = _provider()
    provider._settle_and_reconcile = AsyncMock()

    await provider.apply_zone_change(rooms_to_drop=[KITCHEN], rooms_to_group=[LIVING])

    provider.client.connect_rooms_to_zone.assert_not_awaited()
    provider._settle_and_reconcile.assert_awaited_once()


async def test_settle_waits_until_every_room_resolves_a_location() -> None:
    """A topology still missing device locations is re-fetched before reconciling."""
    provider = _provider()
    provider.topology = _topology(synced=False)
    settled = _topology(synced=True)
    provider.client.get_topology = AsyncMock(
        side_effect=[_topology(synced=True, with_locations=False), settled]
    )
    provider._reconcile = AsyncMock()

    with patch("music_assistant.providers.teufel_raumfeld.provider.asyncio.sleep"):
        await provider._settle_and_reconcile(prefer_leader=LIVING)

    assert provider.topology is settled
    provider._reconcile.assert_awaited_once_with(prefer_leader=LIVING)


def test_forget_zone_unconfirms_it() -> None:
    """A zone proven dead is no longer treated as a real speaker group."""
    provider = _provider()
    provider.confirm_zone(ZONE)

    provider.forget_zone(ZONE)

    assert not provider.is_zone_confirmed(ZONE)


@pytest.mark.parametrize("model", ["Raumfeld One M", None])
async def test_get_device_model_reads_each_renderer_until_it_succeeds(model: str | None) -> None:
    """A read model is cached per renderer; a failed read is retried next time."""
    provider = _provider()
    topology = _topology(synced=False)
    reader = AsyncMock(return_value=model)

    with patch("music_assistant.providers.teufel_raumfeld.provider.get_device_model", reader):
        assert await provider.get_device_model(RENDERER, topology) == model
        assert await provider.get_device_model(RENDERER, topology) == model

    assert reader.await_count == (1 if model else 2)


async def test_get_device_model_without_location_reads_nothing() -> None:
    """A renderer the topology has no location for yields no model and no request."""
    provider = _provider()
    reader = AsyncMock()

    with patch("music_assistant.providers.teufel_raumfeld.provider.get_device_model", reader):
        assert await provider.get_device_model(RENDERER, RaumfeldTopology()) is None

    reader.assert_not_awaited()


def _mass_with_webserver() -> MagicMock:
    """Return a MusicAssistant stand-in whose stream server really tracks dynamic routes."""
    mass = MagicMock()
    mass.streams = Webserver(logging.getLogger("test"), enable_dynamic_routes=True)
    mass.streams.base_url = "http://10.0.0.86:8097"
    mass.players.unregister = AsyncMock()
    return mass


def test_each_instance_gets_its_own_notify_route() -> None:
    """Two instances (two Raumfeld systems) can be set up side by side."""
    mass = _mass_with_webserver()

    first = RaumfeldNotifyServer(MagicMock(), mass, "teufel_raumfeld--a")
    second = RaumfeldNotifyServer(MagicMock(), mass, "teufel_raumfeld--b")

    assert first.callback_url != second.callback_url


async def test_unload_releases_the_notify_route() -> None:
    """A deleted provider can be set up again without restarting MA."""
    mass = _mass_with_webserver()
    provider = _provider()
    provider.mass = mass
    provider._watch_tasks = []
    provider.notify_server = RaumfeldNotifyServer(MagicMock(), mass, "teufel_raumfeld--a")

    await provider.unload(is_removed=True)

    RaumfeldNotifyServer(MagicMock(), mass, "teufel_raumfeld--a")


async def test_failed_init_leaves_no_notify_route_behind() -> None:
    """A host that fails during setup does not block the next setup attempt."""
    mass = _mass_with_webserver()
    provider = _provider()
    provider.mass = mass
    provider.get_setup_value = MagicMock(side_effect=["10.0.0.125", 47365])
    client = MagicMock()
    client.ping = AsyncMock(return_value=True)
    client.get_topology = AsyncMock(side_effect=TimeoutError)

    with (
        patch(
            "music_assistant.providers.teufel_raumfeld.provider.RaumfeldWebserviceClient",
            return_value=client,
        ),
        pytest.raises(TimeoutError),
    ):
        await provider.handle_async_init()

    assert mass.streams._dynamic_routes == {}
