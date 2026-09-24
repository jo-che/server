"""Helpers for UPnP eventing and the vendor per-room-volume action."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import aiohttp
import defusedxml.ElementTree as DefusedET
from aiohttp.web import Request, Response
from async_upnp_client.const import HttpRequest
from async_upnp_client.event_handler import UpnpEventHandler, UpnpNotifyServer
from defusedxml import DefusedXmlException

from .const import (
    ACTION_GET_ROOM_MUTE,
    ACTION_GET_ROOM_VOLUME,
    ACTION_SET_ROOM_MUTE,
    ACTION_SET_ROOM_VOLUME,
    SERVICE_RENDERING_CONTROL,
    UPNP_DEVICE_NAMESPACE,
)

_DESCRIPTION_TIMEOUT = 5

if TYPE_CHECKING:
    from async_upnp_client.client import UpnpDevice, UpnpRequester

    from music_assistant import MusicAssistant


class RaumfeldNotifyServer(UpnpNotifyServer):  # type: ignore[misc,unused-ignore]
    """Notify server for async_upnp_client which uses the MA webserver."""

    def __init__(self, requester: UpnpRequester, mass: MusicAssistant, instance_id: str) -> None:
        """
        Initialize the notify server and register it on MA's shared webserver.

        Call `close()` when done, the route stays registered until then.

        :param requester: async_upnp_client requester used for the event subscriptions.
        :param mass: The Music Assistant instance to register the NOTIFY route on.
        :param instance_id: The provider instance, so each instance gets its own route.
        """
        self.mass = mass
        self.event_handler = UpnpEventHandler(self, requester)
        self._path = f"/teufel_raumfeld_notify/{instance_id}"
        self._unregister_route = self.mass.streams.register_dynamic_route(
            self._path, self._handle_request, method="NOTIFY"
        )

    @property
    def callback_url(self) -> str:
        """Return callback URL on which we are callable."""
        return f"{self.mass.streams.base_url}{self._path}"

    def close(self) -> None:
        """Unregister the NOTIFY route from MA's shared webserver."""
        self._unregister_route()

    async def _handle_request(self, request: Request) -> Response:
        """Handle incoming NOTIFY requests from a subscribed device."""
        if request.method != "NOTIFY":
            return Response(status=405)
        body = (await request.read()).decode("utf-8", errors="replace")
        http_request = HttpRequest(
            method=request.method, url=str(request.url), headers=request.headers, body=body
        )
        try:
            status = await self.event_handler.handle_notify(http_request)
        except ET.ParseError as err:
            self.mass.logger.debug(
                "Ignoring malformed XML in Raumfeld notify from %s: %s", request.remote, err
            )
            return Response(status=400)
        return Response(status=status)


async def get_room_volume(device: UpnpDevice, room_udn: str) -> int | None:
    """
    Read the current volume of a single room synced into the given zone device.

    :param device: The UPnP device of the room's currently addressable zone.
    :param room_udn: UDN of the room to read the volume for.
    """
    service = device.service(SERVICE_RENDERING_CONTROL)
    action = service.action(ACTION_GET_ROOM_VOLUME)
    result = await action.async_call(InstanceID=0, Room=room_udn)
    volume = result.get("CurrentVolume")
    return int(volume) if volume is not None else None


async def set_room_volume(device: UpnpDevice, room_udn: str, volume: int) -> None:
    """
    Set the volume of a single room synced into the given zone device.

    :param device: The UPnP device of the room's currently addressable zone.
    :param room_udn: UDN of the room to set the volume for.
    :param volume: Desired volume level (0-100).
    """
    service = device.service(SERVICE_RENDERING_CONTROL)
    action = service.action(ACTION_SET_ROOM_VOLUME)
    await action.async_call(InstanceID=0, Room=room_udn, DesiredVolume=volume)


async def get_room_mute(device: UpnpDevice, room_udn: str) -> bool | None:
    """
    Read the current mute state of a single room synced into the given zone device.

    :param device: The UPnP device of the room's currently addressable zone.
    :param room_udn: UDN of the room to read the mute state for.
    """
    service = device.service(SERVICE_RENDERING_CONTROL)
    action = service.action(ACTION_GET_ROOM_MUTE)
    result = await action.async_call(InstanceID=0, Room=room_udn)
    muted = result.get("CurrentMute")
    return bool(muted) if muted is not None else None


async def set_room_mute(device: UpnpDevice, room_udn: str, muted: bool) -> None:
    """
    Set the mute state of a single room synced into the given zone device.

    :param device: The UPnP device of the room's currently addressable zone.
    :param room_udn: UDN of the room to set the mute state for.
    :param muted: Desired mute state.
    """
    service = device.service(SERVICE_RENDERING_CONTROL)
    action = service.action(ACTION_SET_ROOM_MUTE)
    await action.async_call(InstanceID=0, Room=room_udn, DesiredMute=muted)


async def get_device_model(session: aiohttp.ClientSession, location: str) -> str | None:
    """
    Read the hardware model name from a UPnP device description.

    Returns None if the description cannot be fetched or has no model name.

    :param session: aiohttp session to fetch the description with.
    :param location: URL of the device's UPnP description XML.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=_DESCRIPTION_TIMEOUT)
        async with session.get(location, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            body = await resp.read()
        root = DefusedET.fromstring(body)
    except TimeoutError, aiohttp.ClientError, ET.ParseError, DefusedXmlException:
        return None
    model = root.findtext(
        f"{{{UPNP_DEVICE_NAMESPACE}}}device/{{{UPNP_DEVICE_NAMESPACE}}}modelName", default=""
    )
    return model.strip() or None
