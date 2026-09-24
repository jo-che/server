"""Tests for the room info the Teufel Raumfeld player takes from the topology."""

from __future__ import annotations

from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web
from music_assistant_models.enums import PlayerFeature

from music_assistant.providers.teufel_raumfeld.helpers import get_device_model
from music_assistant.providers.teufel_raumfeld.player import TeufelRaumfeldPlayer
from music_assistant.providers.teufel_raumfeld.raumfeld_client import (
    RaumfeldRoom,
    RaumfeldTopology,
)
from tests.common import MockProvider

ROOM_UDN = "uuid:Room-Workshop"
RENDERER_UDN = "uuid:Renderer-Workshop"

# Trimmed-down shape of the description a Raumfeld speaker serves for its own renderer.
DEVICE_DESCRIPTION_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<root xmlns="urn:schemas-upnp-org:device-1-0">
  <specVersion><major>1</major><minor>0</minor></specVersion>
  <device>
    <deviceType>urn:schemas-upnp-org:device:MediaRenderer:1</deviceType>
    <friendlyName>Speaker Workshop</friendlyName>
    <manufacturer>Lautsprecher Teufel GmbH</manufacturer>
    <modelDescription>Digital Media Player</modelDescription>
    <modelName> Raumfeld One M </modelName>
    <UDN>uuid:Renderer-Workshop</UDN>
  </device>
</root>
"""


def _player(model: str | None = None) -> TeufelRaumfeldPlayer:
    """
    Return a player for a single room, on a provider that reports the given model.

    :param model: Hardware model name the provider reports for the room's renderer.
    """
    provider = MockProvider("teufel_raumfeld", instance_id="teufel_raumfeld_test")
    provider.get_device_model = AsyncMock(return_value=model)  # type: ignore[attr-defined]
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
async def test_room_with_power_state_supports_power(power_state: str, powered: bool) -> None:
    """A room that reports a power state gets the power feature and its state."""
    player = _player()

    await player.update_room_info(_room(power_state), RaumfeldTopology())

    assert PlayerFeature.POWER in player.supported_features
    assert player._attr_powered is powered
    assert player._attr_name == "Werkstatt"


async def test_room_without_power_state_does_not_support_power() -> None:
    """A room without a power state (a speaker without standby) has no power feature."""
    player = _player()
    await player.update_room_info(_room("ACTIVE"), RaumfeldTopology())

    await player.update_room_info(_room(None), RaumfeldTopology())

    assert PlayerFeature.POWER not in player.supported_features
    assert player._attr_powered is None


async def test_model_comes_from_the_rooms_renderer() -> None:
    """The model is read for the room's physical renderer, not its zone renderer."""
    player = _player(model="Raumfeld One M")
    topology = RaumfeldTopology()

    await player.update_room_info(_room("ACTIVE"), topology)

    player._prov.get_device_model.assert_awaited_once_with(  # type: ignore[attr-defined]
        RENDERER_UDN, topology
    )
    assert player.device_info.model == "Raumfeld One M"


async def test_unreadable_model_keeps_the_previous_one() -> None:
    """A model that cannot be read does not replace one that was read before."""
    player = _player(model="Raumfeld One M")
    await player.update_room_info(_room("ACTIVE"), RaumfeldTopology())
    player._prov.get_device_model.return_value = None  # type: ignore[attr-defined]

    await player.update_room_info(_room("ACTIVE"), RaumfeldTopology())

    assert player.device_info.model == "Raumfeld One M"


@pytest.fixture
async def description_session(aiohttp_client: object) -> tuple[aiohttp.ClientSession, str]:
    """Return a session and base URL of a local server serving device descriptions."""

    async def description(_request: web.Request) -> web.Response:
        return web.Response(body=DEVICE_DESCRIPTION_XML, content_type="text/xml")

    async def malformed(_request: web.Request) -> web.Response:
        return web.Response(body=b"<root><device>", content_type="text/xml")

    async def no_model(_request: web.Request) -> web.Response:
        return web.Response(body=b"<root><device/></root>", content_type="text/xml")

    app = web.Application()
    app.router.add_get("/description.xml", description)
    app.router.add_get("/malformed.xml", malformed)
    app.router.add_get("/no_model.xml", no_model)
    test_client = await aiohttp_client(app)  # type: ignore[operator]
    return test_client.session, str(test_client.make_url("/"))


async def test_get_device_model_reads_model_name(
    description_session: tuple[aiohttp.ClientSession, str],
) -> None:
    """The model name is read from the description, without surrounding whitespace."""
    session, base_url = description_session

    assert await get_device_model(session, f"{base_url}description.xml") == "Raumfeld One M"


@pytest.mark.parametrize("path", ["malformed.xml", "no_model.xml", "missing.xml"])
async def test_get_device_model_returns_none_for_unusable_descriptions(
    description_session: tuple[aiohttp.ClientSession, str], path: str
) -> None:
    """A malformed, model-less or missing description yields no model."""
    session, base_url = description_session

    assert await get_device_model(session, f"{base_url}{path}") is None
