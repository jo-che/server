"""Constants for the Teufel Raumfeld player provider."""

from __future__ import annotations

DOMAIN = "teufel_raumfeld"

CONF_HOST = "host"
CONF_PORT = "port"

DEVICE_MANUFACTURER = "Lautsprecher Teufel GmbH"

# Raumfeld's own power-state strings, as reported per room in the /getZones topology.
POWER_STATE_ACTIVE = "ACTIVE"
POWER_STATE_AUTOMATIC_STANDBY = "AUTOMATIC_STANDBY"
POWER_STATE_MANUAL_STANDBY = "MANUAL_STANDBY"

# UPnP service/action names for the vendor RenderingControl extension that sets/gets the
# volume of a single room while it is synced into a (possibly multi-room) zone.
SERVICE_RENDERING_CONTROL = "urn:schemas-upnp-org:service:RenderingControl:1"
ACTION_GET_ROOM_VOLUME = "GetRoomVolume"
ACTION_SET_ROOM_VOLUME = "SetRoomVolume"
ACTION_GET_ROOM_MUTE = "GetRoomMute"
ACTION_SET_ROOM_MUTE = "SetRoomMute"

POLL_INTERVAL = 30
