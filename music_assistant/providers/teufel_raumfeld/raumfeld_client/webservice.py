"""
Async client for the Raumfeld host webservice.

The Raumfeld "host" (one designated speaker on the system) exposes a small HTTP
webservice with plain GET endpoints returning XML, plus long-polling support (a
``Prefer: wait=<seconds>`` request header and an ``updateID`` response/request header,
where HTTP 200 means "changed" and 304 means "unchanged") for live topology updates.
This is a clean-room implementation of that protocol shape based on publicly observable
endpoint behavior; see the provider's ``manifest.json`` credits for the projects whose
existing (GPLv3) implementations were read to understand the protocol.

This module only speaks to the webservice. It knows nothing about UPnP/DLNA playback
control, which happens directly against the room/zone UPnP renderer devices whose
locations this client resolves (see ``models.RaumfeldTopology.location_for``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import aiohttp

from .exceptions import RaumfeldConnectionError, RaumfeldInvalidHostError
from .models import RaumfeldTopology, parse_devices, parse_zone_config

LOGGER = logging.getLogger(__name__)

DEFAULT_PORT = 47365

_VALIDATION_TIMEOUT = 3
_LONG_POLL_WAIT_SECONDS = 300
_LONG_POLL_TOTAL_TIMEOUT = 330
_LONG_POLL_ERROR_BACKOFF = 15


class RaumfeldWebserviceClient:
    """Async client for one Raumfeld host webservice instance."""

    def __init__(self, host: str, port: int, session: aiohttp.ClientSession) -> None:
        """
        Initialize the client for a given host webservice.

        :param host: Hostname or IP address of the Raumfeld host webservice.
        :param port: Port of the Raumfeld host webservice.
        :param session: aiohttp session to issue requests on (typically MA's shared session).
        """
        self.host = host
        self.port = port
        self.session = session
        self.base_url = f"http://{host}:{port}"

    async def ping(self) -> bool:
        """Check whether the configured host responds like a Raumfeld host webservice."""
        try:
            timeout = aiohttp.ClientTimeout(total=_VALIDATION_TIMEOUT)
            async with self.session.get(self._url("/getHostInfo"), timeout=timeout) as resp:
                return resp.status == 200
        except TimeoutError, aiohttp.ClientError:
            return False

    async def get_topology(self) -> RaumfeldTopology:
        """Fetch a fresh snapshot of the full room/zone/device topology."""
        try:
            zones_xml, devices_xml = await asyncio.gather(
                self._get_bytes("/getZones"), self._get_bytes("/listDevices")
            )
        except aiohttp.ClientError as err:
            raise RaumfeldConnectionError(str(err)) from err
        rooms, zones = parse_zone_config(zones_xml)
        devices = parse_devices(devices_xml)
        return RaumfeldTopology(rooms=rooms, zones=zones, devices=devices)

    async def connect_room_to_zone(self, room_udn: str, zone_udn: str | None = None) -> None:
        """
        Put a single room into a zone.

        :param room_udn: UDN of the room to move.
        :param zone_udn: UDN of the target zone. A new zone is created if omitted.
        """
        params = {"roomUDN": room_udn}
        if zone_udn:
            params["zoneUDN"] = zone_udn
        await self._get("/connectRoomToZone", params=params)

    async def connect_rooms_to_zone(
        self, room_udns: list[str], zone_udn: str | None = None
    ) -> None:
        """
        Put multiple rooms into a zone together.

        :param room_udns: UDNs of the rooms to combine into the zone.
        :param zone_udn: UDN of the target zone. A new zone is created if omitted.
        """
        params = {"roomUDNs": ",".join(room_udns)}
        if zone_udn:
            params["zoneUDN"] = zone_udn
        await self._get("/connectRoomsToZone", params=params)

    async def drop_room(self, room_udn: str) -> None:
        """Drop a room from whatever zone it is currently in."""
        await self._get("/dropRoomJob", params={"roomUDN": room_udn})

    async def enter_automatic_standby(self, room_udn: str) -> None:
        """Put a room into automatic standby."""
        await self._get("/enterAutomaticStandby", params={"roomUDN": room_udn})

    async def enter_manual_standby(self, room_udn: str) -> None:
        """Put a room into manual standby."""
        await self._get("/enterManualStandby", params={"roomUDN": room_udn})

    async def leave_standby(self, room_udn: str) -> None:
        """Wake a room from standby."""
        await self._get("/leaveStandby", params={"roomUDN": room_udn})

    async def long_poll(self, path: str) -> AsyncIterator[bytes]:
        """
        Yield the response body of `path` every time the webservice reports a change.

        Runs forever until the surrounding task is cancelled. A request timeout (no
        change within the requested wait window) is treated as "keep waiting", not an
        error; only actual connection failures back off before retrying.

        :param path: Webservice path to long-poll, e.g. "/getZones".
        """
        update_id: str | None = None
        timeout = aiohttp.ClientTimeout(total=_LONG_POLL_TOTAL_TIMEOUT)
        while True:
            headers = {"Prefer": f"wait={_LONG_POLL_WAIT_SECONDS}"}
            if update_id:
                headers["updateID"] = update_id
            try:
                async with self.session.get(
                    self._url(path), headers=headers, timeout=timeout
                ) as resp:
                    if resp.status == 200:
                        update_id = resp.headers.get("updateID", update_id)
                        yield await resp.read()
                        continue
                    if resp.status != 304:
                        LOGGER.debug(
                            "Long-poll of %s returned unexpected status %s", path, resp.status
                        )
                        await asyncio.sleep(_LONG_POLL_ERROR_BACKOFF)
            except TimeoutError:
                # normal: nothing changed within the requested wait window
                continue
            except aiohttp.ClientError as err:
                LOGGER.debug("Long-poll of %s failed: %r", path, err)
                await asyncio.sleep(_LONG_POLL_ERROR_BACKOFF)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    async def _get(self, path: str, params: dict[str, str] | None = None) -> None:
        try:
            async with self.session.get(self._url(path), params=params) as resp:
                if resp.status >= 400:
                    LOGGER.warning("Request to %s failed with status %s", path, resp.status)
        except aiohttp.ClientError as err:
            raise RaumfeldConnectionError(str(err)) from err

    async def _get_bytes(self, path: str) -> bytes:
        async with self.session.get(self._url(path)) as resp:
            if resp.status != 200:
                raise RaumfeldInvalidHostError(f"Unexpected status {resp.status} from {path}")
            return await resp.read()
