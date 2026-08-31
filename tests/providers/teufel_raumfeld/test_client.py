"""Tests for the native Raumfeld host webservice client."""

from __future__ import annotations

import aiohttp
import pytest
from aiohttp import web

from music_assistant.providers.teufel_raumfeld.raumfeld_client import (
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
    assert [r.udn for r in topology.rooms_in_zone("uuid:Zone-Living-Kitchen")] == [
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

    app = web.Application()
    app["connect_calls"] = connect_calls
    app.router.add_get("/getZones", get_zones)
    app.router.add_get("/listDevices", list_devices)
    app.router.add_get("/getHostInfo", get_host_info)
    app.router.add_get("/connectRoomsToZone", connect_rooms_to_zone)
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
