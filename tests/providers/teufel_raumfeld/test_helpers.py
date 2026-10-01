"""Tests for the per-room volume actions and the UPnP event route."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from async_upnp_client.client import UpnpDevice, UpnpRequester
from async_upnp_client.client_factory import UpnpFactory
from async_upnp_client.const import HttpRequest, HttpResponse

from music_assistant.helpers.webserver import Webserver
from music_assistant.providers.teufel_raumfeld.helpers import (
    RaumfeldNotifyServer,
    get_room_mute,
    get_room_volume,
    set_room_mute,
    set_room_volume,
)

FIXTURES = Path(__file__).parent / "fixtures"
ZONE_DESCRIPTION_URL = "http://192.0.2.125:50000/zone.xml"
ROOM = "uuid:00000000-0000-4000-8000-000000000005"
SERVICE = "urn:schemas-upnp-org:service:RenderingControl:1"

# Only the parts of a zone renderer's description the room actions need; the service
# description it points at (fixtures/rendering_service.xml) is the one a real zone
# renderer serves, so the actions are checked against its actual argument names.
ZONE_DESCRIPTION = f"""<?xml version="1.0"?>
<root xmlns="urn:schemas-upnp-org:device-1-0">
  <specVersion><major>1</major><minor>0</minor></specVersion>
  <device>
    <deviceType>urn:schemas-upnp-org:device:MediaRenderer:1</deviceType>
    <friendlyName>Workshop</friendlyName>
    <manufacturer>Lautsprecher Teufel GmbH</manufacturer>
    <modelName>Teufel One S</modelName>
    <UDN>uuid:00000000-0000-4000-8000-000000000004</UDN>
    <serviceList>
      <service>
        <serviceType>{SERVICE}</serviceType>
        <serviceId>urn:upnp-org:serviceId:RenderingControl</serviceId>
        <SCPDURL>/RenderingService.xml</SCPDURL>
        <controlURL>/RenderingService/Control</controlURL>
        <eventSubURL>/RenderingService/Event</eventSubURL>
      </service>
    </serviceList>
  </device>
</root>
"""


class FakeZoneRenderer(UpnpRequester):
    """Stands in for a zone renderer: serves its descriptions and answers its actions."""

    def __init__(self, results: dict[str, dict[str, str]] | None = None) -> None:
        """
        Initialize the renderer with the output arguments each action answers with.

        :param results: Output arguments per action name, e.g. {"GetRoomVolume": {...}}.
        """
        self.results = results or {}
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def async_http_request(self, http_request: HttpRequest) -> HttpResponse:
        """Answer a description download or a SOAP action call."""
        if http_request.method == "GET":
            if http_request.url == ZONE_DESCRIPTION_URL:
                return HttpResponse(200, {}, ZONE_DESCRIPTION)
            return HttpResponse(200, {}, (FIXTURES / "rendering_service.xml").read_text())
        soap_action = next(v for k, v in http_request.headers.items() if k.lower() == "soapaction")
        action = soap_action.strip('"').split("#")[1]
        self.calls.append(
            (action, dict(re.findall(r"<(\w+)>([^<]*)</\1>", http_request.body or "")))
        )
        out = "".join(f"<{k}>{v}</{k}>" for k, v in self.results.get(action, {}).items())
        body = (
            '<?xml version="1.0"?><s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"'
            ' s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/"><s:Body>'
            f'<u:{action}Response xmlns:u="{SERVICE}">{out}</u:{action}Response>'
            "</s:Body></s:Envelope>"
        )
        return HttpResponse(200, {}, body)


async def _zone(renderer: FakeZoneRenderer) -> UpnpDevice:
    """
    Return the zone renderer as async_upnp_client sees it.

    :param renderer: The stand-in renderer to load the device from.
    """
    return await UpnpFactory(renderer).async_create_device(ZONE_DESCRIPTION_URL)


async def test_get_room_volume_asks_for_the_room() -> None:
    """The volume of one room is read with the room's UDN, as the renderer expects."""
    renderer = FakeZoneRenderer({"GetRoomVolume": {"CurrentVolume": "40"}})

    assert await get_room_volume(await _zone(renderer), ROOM) == 40
    assert renderer.calls == [("GetRoomVolume", {"InstanceID": "0", "Room": ROOM})]


async def test_set_room_volume_targets_the_room() -> None:
    """Setting a room's volume names the room and the desired level."""
    renderer = FakeZoneRenderer()

    await set_room_volume(await _zone(renderer), ROOM, 25)

    assert renderer.calls == [
        ("SetRoomVolume", {"InstanceID": "0", "Room": ROOM, "DesiredVolume": "25"})
    ]


@pytest.mark.parametrize(("reported", "muted"), [("0", False), ("1", True)])
async def test_get_room_mute_reads_the_boolean(reported: str, muted: bool) -> None:
    """The renderer's "0"/"1" mute state is read as a real boolean (an unmuted "0" too)."""
    renderer = FakeZoneRenderer({"GetRoomMute": {"CurrentMute": reported}})

    assert await get_room_mute(await _zone(renderer), ROOM) is muted


async def test_set_room_mute_targets_the_room() -> None:
    """Muting a room names the room and sends the mute state as a UPnP boolean."""
    renderer = FakeZoneRenderer()

    await set_room_mute(await _zone(renderer), ROOM, True)

    assert renderer.calls == [
        ("SetRoomMute", {"InstanceID": "0", "Room": ROOM, "DesiredMute": "1"})
    ]


def _notify_server() -> tuple[RaumfeldNotifyServer, AsyncMock]:
    """Return a notify server on a stand-in for MA's stream server, and its event handler."""
    mass = MagicMock()
    mass.streams = Webserver(logging.getLogger("test"), enable_dynamic_routes=True)
    mass.streams.base_url = "http://192.0.2.86:8097"
    server = RaumfeldNotifyServer(MagicMock(), mass, "teufel_raumfeld--a")
    handle_notify = AsyncMock(return_value=200)
    server.event_handler.handle_notify = handle_notify  # type: ignore[method-assign]
    return server, handle_notify


def _request(method: str = "NOTIFY", body: bytes = b"<e:propertyset/>") -> MagicMock:
    """
    Return an incoming event request.

    :param method: The HTTP method of the request.
    :param body: The request body.
    """
    request = MagicMock(
        method=method, url="http://192.0.2.86:8097/x", headers={}, remote="192.0.2.125"
    )
    request.read = AsyncMock(return_value=body)
    return request


async def test_event_is_passed_to_the_event_handler() -> None:
    """An event is handed to async_upnp_client, whose answer goes back to the renderer."""
    server, handle_notify = _notify_server()

    response = await server._handle_request(_request())

    assert response.status == 200
    handle_notify.assert_awaited_once()


async def test_malformed_event_is_rejected() -> None:
    """An event with broken XML is answered with 400 instead of raising."""
    server, handle_notify = _notify_server()
    handle_notify.side_effect = ET.ParseError("broken")

    response = await server._handle_request(_request(body=b"<broken"))

    assert response.status == 400


async def test_request_other_than_an_event_is_refused() -> None:
    """Only NOTIFY requests are events."""
    server, handle_notify = _notify_server()

    response = await server._handle_request(_request(method="GET"))

    assert response.status == 405
    handle_notify.assert_not_awaited()
