"""Tests for the native Raumfeld host webservice client."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web

from music_assistant.providers.teufel_raumfeld.raumfeld_client import (
    RaumfeldCommandError,
    RaumfeldConnectionError,
    RaumfeldTopology,
    RaumfeldWebserviceClient,
)
from music_assistant.providers.teufel_raumfeld.raumfeld_client.models import (
    parse_devices,
    parse_zone_config,
)

# Hand-written fixture XML matching the attribute/element shapes observed while reading
# the (GPLv3, design-reference-only) hassfeld library's XML parsing code - not copied
# from any existing test fixtures.
GET_ZONES_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<zoneConfig>
  <zones>
    <zone udn="uuid:Zone-Living-Kitchen">
      <room name="Living Room" udn="uuid:Room-Living" powerState="ACTIVE">
        <renderer udn="uuid:Renderer-Living"/>
      </room>
      <room name="Kitchen" udn="uuid:Room-Kitchen" powerState="ACTIVE">
        <renderer udn="uuid:Renderer-Kitchen"/>
      </room>
    </zone>
  </zones>
  <unassignedRooms>
    <room name="Bedroom" udn="uuid:Room-Bedroom" powerState="AUTOMATIC_STANDBY">
      <renderer udn="uuid:Renderer-Bedroom"/>
    </room>
  </unassignedRooms>
</zoneConfig>
"""

LIST_DEVICES_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<devices>
  <device location="http://10.0.0.5:8080/desc-living.xml"
          type="urn:schemas-upnp-org:device:MediaRenderer:1"
          udn="uuid:Room-Living">Living Room</device>
  <device location="http://10.0.0.5:8080/desc-zone.xml"
          type="urn:schemas-upnp-org:device:MediaRenderer:1"
          udn="uuid:Zone-Living-Kitchen">Living Room + Kitchen</device>
  <device location="http://10.0.0.5:8080/desc-mediaserver.xml"
          type="urn:schemas-upnp-org:device:MediaServer:1"
          udn="uuid:MediaServer-1">Raumfeld Media Server</device>
</devices>
"""


def test_parse_zone_config_groups_rooms_by_zone() -> None:
    """Rooms inside a <zone> get that zone's UDN; unassigned rooms get None."""
    rooms, zones = parse_zone_config(GET_ZONES_XML)

    assert set(rooms) == {"uuid:Room-Living", "uuid:Room-Kitchen", "uuid:Room-Bedroom"}
    assert rooms["uuid:Room-Living"].zone_udn == "uuid:Zone-Living-Kitchen"
    assert rooms["uuid:Room-Kitchen"].zone_udn == "uuid:Zone-Living-Kitchen"
    assert rooms["uuid:Room-Bedroom"].zone_udn is None
    assert rooms["uuid:Room-Bedroom"].power_state == "AUTOMATIC_STANDBY"
    assert rooms["uuid:Room-Living"].renderer_udn == "uuid:Renderer-Living"

    assert set(zones) == {"uuid:Zone-Living-Kitchen"}
    assert zones["uuid:Zone-Living-Kitchen"].room_udns == [
        "uuid:Room-Living",
        "uuid:Room-Kitchen",
    ]


def test_parse_devices_keys_by_udn() -> None:
    """Devices are keyed by their UDN and include both room and zone entries."""
    devices = parse_devices(LIST_DEVICES_XML)

    assert devices["uuid:Room-Living"].location == "http://10.0.0.5:8080/desc-living.xml"
    assert devices["uuid:Zone-Living-Kitchen"].name == "Living Room + Kitchen"
    assert devices["uuid:MediaServer-1"].device_type == "urn:schemas-upnp-org:device:MediaServer:1"


def test_topology_resolves_addressable_udn_for_synced_and_solo_rooms() -> None:
    """A synced room resolves to its zone's UDN; a solo room resolves to its own."""
    rooms, zones = parse_zone_config(GET_ZONES_XML)
    devices = parse_devices(LIST_DEVICES_XML)
    topology = RaumfeldTopology(rooms=rooms, zones=zones, devices=devices)

    assert topology.addressable_udn_for_room("uuid:Room-Living") == "uuid:Zone-Living-Kitchen"
    assert topology.addressable_udn_for_room("uuid:Room-Bedroom") == "uuid:Room-Bedroom"
    assert (
        topology.location_for(topology.addressable_udn_for_room("uuid:Room-Living"))
        == "http://10.0.0.5:8080/desc-zone.xml"
    )
    assert topology.zones["uuid:Zone-Living-Kitchen"].room_udns == [
        "uuid:Room-Living",
        "uuid:Room-Kitchen",
    ]


async def _fixture_app() -> web.Application:
    async def get_zones(_request: web.Request) -> web.Response:
        return web.Response(body=GET_ZONES_XML, content_type="text/xml")

    async def list_devices(_request: web.Request) -> web.Response:
        return web.Response(body=LIST_DEVICES_XML, content_type="text/xml")

    async def get_host_info(_request: web.Request) -> web.Response:
        return web.Response(body=b"<hostInfo/>", content_type="text/xml")

    connect_calls: list[dict[str, str]] = []

    async def connect_rooms_to_zone(request: web.Request) -> web.Response:
        connect_calls.append(dict(request.query))
        return web.Response(status=200)

    async def reject_command(_request: web.Request) -> web.Response:
        # what a real host answers for an unknown room (checked live)
        return web.Response(status=400)

    app = web.Application()
    app["connect_calls"] = connect_calls
    app.router.add_get("/getZones", get_zones)
    app.router.add_get("/listDevices", list_devices)
    app.router.add_get("/getHostInfo", get_host_info)
    app.router.add_get("/connectRoomsToZone", connect_rooms_to_zone)
    app.router.add_get("/dropRoomJob", reject_command)
    return app  # app["connect_calls"] is asserted on by the caller of this fixture


@pytest.fixture
async def fixture_app() -> web.Application:
    """Return the raw fixture aiohttp application (exposes app["connect_calls"])."""
    return await _fixture_app()


@pytest.fixture
async def raumfeld_client(
    aiohttp_client: object, fixture_app: web.Application
) -> RaumfeldWebserviceClient:
    """Return a client wired up against a local fixture aiohttp server."""
    test_client = await aiohttp_client(fixture_app)  # type: ignore[operator]
    session: aiohttp.ClientSession = test_client.session
    client = RaumfeldWebserviceClient(host=test_client.host, port=test_client.port, session=session)
    # point base_url at the test server (host/port alone don't carry the test scheme/path)
    client.base_url = str(test_client.make_url(""))
    return client


async def test_ping_succeeds_against_valid_host(
    raumfeld_client: RaumfeldWebserviceClient,
) -> None:
    """ping() returns True when /getHostInfo responds with 200."""
    assert await raumfeld_client.ping() is True


async def test_get_topology_combines_zones_and_devices(
    raumfeld_client: RaumfeldWebserviceClient,
) -> None:
    """get_topology() fetches and merges both endpoints into one snapshot."""
    topology = await raumfeld_client.get_topology()

    assert topology.rooms["uuid:Room-Kitchen"].zone_udn == "uuid:Zone-Living-Kitchen"
    # the kitchen room itself has no standalone device entry while synced into a zone -
    # it must be addressed via the zone's own device/location instead
    assert "uuid:Room-Kitchen" not in topology.devices
    assert topology.location_for(topology.addressable_udn_for_room("uuid:Room-Kitchen")) == (
        "http://10.0.0.5:8080/desc-zone.xml"
    )


async def test_connect_rooms_to_zone_sends_expected_query_params(
    raumfeld_client: RaumfeldWebserviceClient, fixture_app: web.Application
) -> None:
    """connect_rooms_to_zone() sends a comma-joined roomUDNs list and the zone UDN."""
    await raumfeld_client.connect_rooms_to_zone(
        ["uuid:Room-Living", "uuid:Room-Kitchen"], zone_udn="uuid:Zone-Living-Kitchen"
    )

    assert fixture_app["connect_calls"] == [
        {
            "roomUDNs": "uuid:Room-Living,uuid:Room-Kitchen",
            "zoneUDN": "uuid:Zone-Living-Kitchen",
        }
    ]


async def test_rejected_command_raises(raumfeld_client: RaumfeldWebserviceClient) -> None:
    """A command the host rejects is reported to the caller, not just logged."""
    with pytest.raises(RaumfeldCommandError):
        await raumfeld_client.drop_room("uuid:Room-Unknown")


async def _client_for(aiohttp_client: object, app: web.Application) -> RaumfeldWebserviceClient:
    """
    Return a client wired up against the given aiohttp application.

    :param aiohttp_client: The pytest-aiohttp client factory.
    :param app: The application standing in for the host webservice.
    """
    test_client = await aiohttp_client(app)  # type: ignore[operator]
    client = RaumfeldWebserviceClient(test_client.host, test_client.port, test_client.session)
    client.base_url = str(test_client.make_url(""))
    return client


async def test_host_that_stops_answering_times_out(aiohttp_client: object) -> None:
    """A hanging host fails the topology fetch instead of stalling it indefinitely."""

    async def hang(_request: web.Request) -> web.Response:
        await asyncio.sleep(10)
        return web.Response(status=200)

    app = web.Application()
    app.router.add_get("/getZones", hang)
    app.router.add_get("/listDevices", hang)
    client = await _client_for(aiohttp_client, app)

    with (
        patch(
            "music_assistant.providers.teufel_raumfeld.raumfeld_client.webservice._REQUEST_TIMEOUT",
            0.1,
        ),
        pytest.raises(RaumfeldConnectionError),
    ):
        await client.get_topology()


async def test_long_poll_sends_the_last_update_id_back(aiohttp_client: object) -> None:
    """Each long-poll request carries the updateID of the previous answer."""
    received: list[str | None] = []

    async def get_zones(request: web.Request) -> web.Response:
        received.append(request.headers.get("updateID"))
        return web.Response(
            body=GET_ZONES_XML, content_type="text/xml", headers={"updateID": str(len(received))}
        )

    app = web.Application()
    app.router.add_get("/getZones", get_zones)
    client = await _client_for(aiohttp_client, app)

    polls = client.long_poll("/getZones")
    assert await anext(polls) == GET_ZONES_XML
    assert await anext(polls) == GET_ZONES_XML
    await polls.aclose()

    assert received == [None, "1"]


# Captured from a real 5-room system (room names, UDNs and addresses anonymized). Unlike the
# hand-written XML above, it carries what real firmware sends: renderer names, extra
# attributes, a speaker without standby (no powerState) and a room in no zone at all.
FIXTURES = Path(__file__).parent / "fixtures"


def _real_topology() -> RaumfeldTopology:
    """Return the topology of the captured real system."""
    rooms, zones = parse_zone_config((FIXTURES / "get_zones.xml").read_bytes())
    devices = parse_devices((FIXTURES / "list_devices.xml").read_bytes())
    return RaumfeldTopology(rooms=rooms, zones=zones, devices=devices)


def test_real_system_rooms_are_parsed() -> None:
    """Every room of a real system is found, with what real firmware does and leaves out."""
    topology = _real_topology()
    rooms = {room.name: room for room in topology.rooms.values()}

    assert set(rooms) == {"Living Room", "Workshop", "Office", "Kitchen", "Bedroom"}
    # a first-generation One M has no standby, so its room reports no power state at all
    assert rooms["Workshop"].power_state is None
    assert rooms["Kitchen"].power_state == "AUTOMATIC_STANDBY"
    # a room outside any zone
    assert rooms["Bedroom"].zone_udn is None
    assert rooms["Workshop"].renderer_udn == "uuid:00000000-0000-4000-8000-000000000006"


def test_real_system_zone_rooms_resolve_their_renderer() -> None:
    """Each room in a zone resolves the zone renderer to control it by."""
    topology = _real_topology()

    for room in topology.rooms.values():
        if room.zone_udn is None:
            continue
        location = topology.location_for(topology.addressable_udn_for_room(room.udn))
        assert location is not None
        assert location.startswith("http://192.0.2.125:")  # zone renderers run on the host
