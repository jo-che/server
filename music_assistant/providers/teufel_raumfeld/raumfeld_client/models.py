"""
Parsed representations of the Raumfeld host webservice's topology data.

The host webservice returns simple attribute-based XML for its topology endpoints
(``/getZones``, ``/listDevices``). The shapes below were derived by observing that XML,
not by copying any third-party parsing code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as DefusedET


@dataclass
class RaumfeldDevice:
    """
    A UPnP device known to the Raumfeld host, as returned by ``/listDevices``.

    Both physical rooms and (once formed) multi-room zones show up here, each under
    their own UDN with the location of their (dynamically created) UPnP description.xml.
    This is the only reliable way to resolve "which UPnP renderer belongs to this
    room/zone right now", since those locations change as zones form and dissolve.
    """

    udn: str
    location: str
    device_type: str
    name: str | None = None


@dataclass
class RaumfeldRoom:
    """A single Raumfeld "room" (one physical speaker/speaker-group position)."""

    udn: str
    name: str
    renderer_udn: str
    power_state: str | None = None
    zone_udn: str | None = None  # None while the room is not part of any zone


@dataclass
class RaumfeldZone:
    """A (possibly dynamically formed) zone combining one or more rooms."""

    udn: str
    room_udns: list[str] = field(default_factory=list)


@dataclass
class RaumfeldTopology:
    """Snapshot of the full room/zone/device topology."""

    rooms: dict[str, RaumfeldRoom] = field(default_factory=dict)
    zones: dict[str, RaumfeldZone] = field(default_factory=dict)
    devices: dict[str, RaumfeldDevice] = field(default_factory=dict)

    def location_for(self, udn: str | None) -> str | None:
        """Return the current UPnP device location for a room or zone UDN, if known."""
        if udn and (device := self.devices.get(udn)):
            return device.location
        return None

    def addressable_udn_for_room(self, room_udn: str) -> str | None:
        """
        Return the UDN that must be addressed to control this room right now.

        This is the room's own UDN while it is unassigned/solo, or its zone's UDN
        while it is synced with other rooms.

        :param room_udn: UDN of the room to resolve.
        """
        room = self.rooms.get(room_udn)
        if room is None:
            return None
        return room.zone_udn or room.udn

    def rooms_in_zone(self, zone_udn: str) -> list[RaumfeldRoom]:
        """Return all rooms currently combined into the given zone."""
        zone = self.zones.get(zone_udn)
        if zone is None:
            return []
        return [self.rooms[udn] for udn in zone.room_udns if udn in self.rooms]


def parse_zone_config(xml_bytes: bytes) -> tuple[dict[str, RaumfeldRoom], dict[str, RaumfeldZone]]:
    """
    Parse a ``/getZones`` response into rooms and zones, keyed by UDN.

    :param xml_bytes: Raw XML body of a ``/getZones`` response.
    """
    root = DefusedET.fromstring(xml_bytes)
    rooms: dict[str, RaumfeldRoom] = {}
    zones: dict[str, RaumfeldZone] = {}

    def parse_room(room_el: Element, zone_udn: str | None) -> None:
        room_udn = room_el.attrib["udn"]
        renderer_el = room_el.find("renderer")
        renderer_udn = renderer_el.attrib["udn"] if renderer_el is not None else room_udn
        rooms[room_udn] = RaumfeldRoom(
            udn=room_udn,
            name=room_el.attrib.get("name", room_udn),
            renderer_udn=renderer_udn,
            power_state=room_el.attrib.get("powerState"),
            zone_udn=zone_udn,
        )

    zones_el = root.find("zones")
    if zones_el is not None:
        for zone_el in zones_el.findall("zone"):
            zone_udn = zone_el.attrib["udn"]
            room_udns: list[str] = []
            for room_el in zone_el.findall("room"):
                parse_room(room_el, zone_udn)
                room_udns.append(room_el.attrib["udn"])
            zones[zone_udn] = RaumfeldZone(udn=zone_udn, room_udns=room_udns)

    unassigned_el = root.find("unassignedRooms")
    if unassigned_el is not None:
        for room_el in unassigned_el.findall("room"):
            parse_room(room_el, None)

    return rooms, zones


def parse_devices(xml_bytes: bytes) -> dict[str, RaumfeldDevice]:
    """
    Parse a ``/listDevices`` response into devices, keyed by UDN.

    :param xml_bytes: Raw XML body of a ``/listDevices`` response.
    """
    root = DefusedET.fromstring(xml_bytes)
    devices: dict[str, RaumfeldDevice] = {}
    for device_el in root.findall("device"):
        udn = device_el.attrib["udn"]
        devices[udn] = RaumfeldDevice(
            udn=udn,
            location=device_el.attrib["location"],
            device_type=device_el.attrib.get("type", ""),
            name=device_el.text,
        )
    return devices
