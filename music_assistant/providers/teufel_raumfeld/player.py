"""Teufel Raumfeld Player implementation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from contextlib import suppress
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast

from async_upnp_client.exceptions import UpnpError, UpnpResponseError
from async_upnp_client.profiles.dlna import DmrDevice, TransportState
from music_assistant_models.enums import IdentifierType, PlaybackState, PlayerFeature, PlayerType
from music_assistant_models.errors import PlayerUnavailableError

from music_assistant.helpers.upnp import create_didl_metadata
from music_assistant.models.player import DeviceInfo, Player

from .const import DEVICE_MANUFACTURER, POLL_INTERVAL, POWER_STATE_ACTIVE
from .helpers import get_room_mute, get_room_volume, set_room_mute, set_room_volume

if TYPE_CHECKING:
    from async_upnp_client.client import UpnpService, UpnpStateVariable
    from music_assistant_models.player import PlayerMedia

    from .provider import TeufelRaumfeldPlayerProvider
    from .raumfeld_client import RaumfeldTopology


class TeufelRaumfeldPlayer(Player):
    """
    A single Raumfeld "room".

    Rooms are the stable unit of identity: a room's player_id (its Raumfeld room UDN)
    never changes, but the UPnP device it actually controls does - it points at the
    room's own virtual renderer while unsynced, and at the shared zone renderer while
    combined with other rooms into a zone. `connect()` keeps that pointer current.
    """

    _attr_type = PlayerType.PLAYER

    def __init__(self, provider: TeufelRaumfeldPlayerProvider, room_udn: str, name: str) -> None:
        """
        Initialize the player for a single Raumfeld room.

        :param provider: The owning player provider.
        :param room_udn: UDN of the Raumfeld room this player represents.
        :param name: Initial display name of the room.
        """
        super().__init__(provider, room_udn)
        self.room_udn = room_udn
        self.device: DmrDevice | None = None
        self._current_location: str | None = None
        self._confirmed_solo_zone_udn: str | None = None
        self.lock = asyncio.Lock()
        self.force_poll = False
        self.last_seen = time.time()
        self._attr_name = name
        # the UUID identifier is set (and kept current) by connect() instead of here -
        # see the comment there for why it must track the room's currently addressed
        # zone/renderer UDN rather than the room's own stable UDN
        self._attr_device_info = DeviceInfo(model="Raumfeld", manufacturer=DEVICE_MANUFACTURER)
        self._attr_needs_poll = True
        self._attr_poll_interval = POLL_INTERVAL
        self._attr_can_group_with = {provider.instance_id}
        self._attr_supported_features = {
            PlayerFeature.PLAY_MEDIA,
            PlayerFeature.ENQUEUE,
            PlayerFeature.GAPLESS_PLAYBACK,
            PlayerFeature.VOLUME_SET,
            PlayerFeature.VOLUME_MUTE,
            PlayerFeature.PAUSE,
            PlayerFeature.SEEK,
            PlayerFeature.NEXT_PREVIOUS,
            PlayerFeature.SET_MEMBERS,
            PlayerFeature.POWER,
        }

    async def connect(self, topology: RaumfeldTopology) -> None:
        """
        (Re)connect the underlying UPnP device to this room's currently addressable location.

        A no-op if already connected to that same location. Marks the player unavailable
        (without raising) if the topology has no resolvable location for this room, e.g.
        during the brief window between a zone change being announced and its new device
        location becoming known.

        :param topology: The current topology snapshot to resolve this room's location from.
        """
        # The whole method runs under self.lock, not just the device-swap at the end:
        # _ensure_real_solo_zone's check-and-act (read _confirmed_solo_zone_udn, maybe
        # call the destructive connectRoomToZone, then set it) must be atomic across
        # concurrent callers (a background reconcile pass, poll()'s self-heal, and
        # play_media's own connect() call can all race here) - otherwise two callers can
        # both decide a fresh zone is needed and each create one, with the second
        # silently orphaning whatever the first had already started playing on.
        async with self.lock:
            addressable_udn = topology.addressable_udn_for_room(self.room_udn)
            addressable_udn, topology = await self._ensure_real_solo_zone(addressable_udn, topology)
            location = topology.location_for(addressable_udn)
            if not location or not addressable_udn:
                # a room in standby has no live UPnP renderer at all (Raumfeld tears it
                # down until the room wakes up) - that's an expected, normal state, not
                # a connectivity failure, so it must not mark the player unavailable.
                # See poll() for how this is told apart from a room that unexpectedly
                # has no device while it should (a real problem, unavailable is right).
                await self._disconnect_device()
                return
            if location == self._current_location and self.device is not None:
                return
            await self._disconnect_device()
            try:
                upnp_device = await self._prov.upnp_factory.async_create_device(location)
            except (UpnpError, TimeoutError) as err:
                # The zone we resolved to is unreachable - it most likely no longer
                # exists as a live UPnP device even though the topology still reports
                # it (Raumfeld can leave a stale zone_udn behind, e.g. across a standby
                # cycle - the same class of staleness `_ensure_real_solo_zone` guards
                # against on the create side). Forget a matching confirmation so the
                # next attempt creates a genuinely fresh zone instead of retrying this
                # same dead location forever.
                if addressable_udn == self._confirmed_solo_zone_udn:
                    self._confirmed_solo_zone_udn = None
                self.logger.debug("Failed to connect to %s: %r", location, err)
                return
            self.device = DmrDevice(upnp_device, self._prov.notify_server.event_handler)
            self._current_location = location
            # Kept in sync with the room's *currently addressed* zone/renderer UDN
            # (not the room's own stable UDN, used as player_id): this is what a
            # generic DLNA-discovered player for the same physical renderer reports as
            # its own UUID identifier, so MA's protocol-linking can recognize the two
            # as the same device and hide/link the DLNA one instead of leaving both to
            # independently subscribe to (and corrupt each other's view of) it. The zone
            # UDN changes as rooms group/ungroup, hence updating it here on every connect.
            self._attr_device_info.add_identifier(
                IdentifierType.UUID, addressable_udn.removeprefix("uuid:")
            )
            self.device.on_event = self._handle_event
            try:
                await self.device.async_subscribe_services(auto_resubscribe=True)
            except UpnpResponseError as err:
                self.logger.debug("Device rejected subscription: %r", err)
            except UpnpError as err:
                self.logger.debug("Error while subscribing during connect: %r", err)

    async def refresh_from_topology(self, topology: RaumfeldTopology) -> None:
        """
        Reconnect if this room's addressable location changed and refresh static room info.

        :param topology: The current topology snapshot.
        """
        room = topology.rooms.get(self.room_udn)
        if room is None:
            self._attr_available = False
            self.update_state()
            return
        # available means "known to the Raumfeld system", not "currently has a live
        # UPnP renderer" - a standby room is legitimately available with no renderer;
        # see connect()/poll() for how that case is handled without a device.
        self._attr_available = True
        self._attr_name = room.name
        self._attr_powered = (
            room.power_state == POWER_STATE_ACTIVE if room.power_state is not None else None
        )
        await self.connect(topology)
        self.force_poll = True
        self.update_state()

    async def on_unload(self) -> None:
        """Handle logic when the player is unloaded from the Player controller."""
        await super().on_unload()
        await self._disconnect_device()

    # COMMANDS

    async def power(self, powered: bool) -> None:
        """Handle POWER command on the player."""
        if powered:
            await self._prov.client.leave_standby(self.room_udn)
        else:
            await self._prov.client.enter_manual_standby(self.room_udn)
        self._attr_powered = powered
        self.update_state()

    async def volume_set(self, volume_level: int) -> None:
        """Handle VOLUME_SET command on the player."""
        if self.device is None:
            return
        await set_room_volume(self.device.device, self.room_udn, volume_level)
        self._attr_volume_level = volume_level
        self.update_state()

    async def volume_mute(self, muted: bool) -> None:
        """Handle VOLUME_MUTE command on the player."""
        if self.device is None:
            return
        await set_room_mute(self.device.device, self.room_udn, muted)
        self._attr_volume_muted = muted
        self.update_state()

    async def play(self) -> None:
        """Send PLAY command to the player."""
        if self.device is not None:
            await self.device.async_play()

    async def stop(self) -> None:
        """Send STOP command to the player."""
        if self.device is not None and self.device.can_stop:
            await self.device.async_stop()

    async def pause(self) -> None:
        """Send PAUSE command to the player."""
        if self.device is not None and self.device.can_pause:
            await self.device.async_pause()

    async def seek(self, position: int) -> None:
        """Send SEEK command to the player."""
        if self.device is not None:
            await self.device.async_seek_rel_time(timedelta(seconds=position))

    async def next_track(self) -> None:
        """Send NEXT command to the player."""
        if self.device is not None and self.device.can_next:
            await self.device.async_next()

    async def previous_track(self) -> None:
        """Send PREVIOUS command to the player."""
        if self.device is not None and self.device.can_previous:
            await self.device.async_previous()

    async def play_media(self, media: PlayerMedia) -> None:
        """Handle PLAY_MEDIA command on the player."""
        await self._ensure_connected_for_playback()
        url = await self.mass.streams.resolve_stream_url(self.player_id, media)
        didl_metadata = create_didl_metadata(media, url)
        title = media.title or media.uri
        # held for the whole UPnP sequence (not just connect()) so a concurrent
        # reconnect can't swap self.device out from under it mid-flight - see
        # connect()'s docstring for why that can otherwise silently orphan playback.
        async with self.lock:
            if self.device is None:
                return
            if self.device.can_stop:
                await self.device.async_stop()
            self.set_current_media(uri=url, clear_all=True)
            self._attr_playback_state = PlaybackState.PLAYING
            self._attr_elapsed_time = 0
            self._attr_elapsed_time_last_updated = time.time()
            await self.device.async_set_transport_uri(url, title, didl_metadata)
            await self.device.async_wait_for_can_play(10)
            await self.device.async_play()
        self.update_state()

    async def enqueue_next_media(self, media: PlayerMedia) -> None:
        """Handle enqueuing of the next queue item on the player."""
        if self.device is None:
            return
        url = await self.mass.streams.resolve_stream_url(self.player_id, media)
        didl_metadata = create_didl_metadata(media, url)
        title = media.title or media.uri
        try:
            await self.device.async_set_next_transport_uri(url, title, didl_metadata)
        except UpnpError:
            self.logger.warning("Failed to enqueue next track for player %s", self.display_name)

    async def set_members(
        self,
        player_ids_to_add: list[str] | None = None,
        player_ids_to_remove: list[str] | None = None,
    ) -> None:
        """Handle SET_MEMBERS command on the player."""
        current = set(self._attr_group_members) or {self.player_id}
        current -= set(player_ids_to_remove or [])
        current |= set(player_ids_to_add or [])
        # prefer_leader makes this room (the one the command was issued on) the MA sync
        # leader, rather than the reconcile pass's own deterministic tie-break - see
        # TeufelRaumfeldPlayerProvider._pick_leader.
        await self._prov.apply_zone_change(
            rooms_to_drop=list(player_ids_to_remove or []),
            rooms_to_group=sorted(current),
            prefer_leader=self.player_id,
        )

    async def ungroup(self) -> None:
        """Handle UNGROUP command on the player."""
        await self._prov.apply_zone_change(rooms_to_drop=[self.room_udn], rooms_to_group=[])

    async def poll(self) -> None:
        """Poll player for state updates."""
        if self.device is None:
            # a previous connect/poll failed, or this room hasn't had a topology-driven
            # reconcile pass reach it yet - retry here too so a room that failed to
            # connect once does not stay stuck until the next topology change event
            # happens to touch it (which may be a long time on an idle system).
            await self.connect(self._prov.topology)
            if self.device is None:
                if self._attr_powered is False:
                    # a standby room has no live renderer to poll - normal, not an error
                    return
                raise PlayerUnavailableError
        try:
            now = time.time()
            do_ping = self.force_poll or (now - self.last_seen) > 60
            await self.device.async_update(do_ping=do_ping)
            self.last_seen = now if do_ping else self.last_seen
            volume = await get_room_volume(self.device.device, self.room_udn)
            if volume is not None:
                self._attr_volume_level = volume
            muted = await get_room_mute(self.device.device, self.room_udn)
            if muted is not None:
                self._attr_volume_muted = muted
        except UpnpError as err:
            self.logger.debug("Device unavailable: %r", err)
            await self._disconnect_device()
            raise PlayerUnavailableError from err
        finally:
            self.force_poll = False
        self._sync_from_device()
        self.update_state()

    # PRIVATE

    async def _ensure_connected_for_playback(self) -> None:
        """
        Wake this room if needed and make sure it is connected to a real zone.

        `play_media` must not rely on a background reconcile pass having already
        reached this room (it may not have, e.g. right after a `power()` call that
        itself does not connect) or on `self.device` already being set (it could be a
        stale connection to a since-torn-down zone) - both would silently issue the
        play command against a defunct or otherwise non-audible endpoint. This always
        fetches a fresh topology and calls `connect()` (which itself only recreates
        the zone via `_ensure_real_solo_zone` when actually needed), retrying briefly
        since a room just woken from standby needs a moment to get a resolvable zone.
        """
        if self._attr_powered is False:
            await self._prov.client.leave_standby(self.room_udn)
        for attempt in range(5):
            topology = await self._prov.client.get_topology()
            await self.connect(topology)
            if self.device is not None:
                return
            if attempt + 1 < 5:
                await asyncio.sleep(0.5)

    def _handle_event(
        self, service: UpnpService, state_variables: Sequence[UpnpStateVariable[Any]]
    ) -> None:
        """Handle state variable(s) changed event from the UPnP device."""
        if not state_variables:
            self.force_poll = True
            return
        if service.service_id == "urn:upnp-org:serviceId:AVTransport":
            for state_variable in state_variables:
                if state_variable.name == "TransportState" and state_variable.value in (
                    TransportState.PLAYING,
                    TransportState.PAUSED_PLAYBACK,
                ):
                    self.force_poll = True
        self.last_seen = time.time()
        self.mass.create_task(self._async_update_after_event())

    async def _async_update_after_event(self) -> None:
        """Refresh player state after a UPnP event, polling first if requested."""
        if self.force_poll:
            with suppress(PlayerUnavailableError):
                await self.poll()
            return
        self._sync_from_device()
        self.update_state()

    def _sync_from_device(self) -> None:
        """Copy the connected UPnP device's current state onto this player's attributes."""
        if self.device is None:
            return
        self._attr_playback_state = self._get_playback_state()
        media_duration = self.device.media_duration
        self.set_current_media(
            uri=self.device.current_track_uri or "",
            clear_all=True,
            title=self.device.media_title,
            artist=self.device.media_artist,
            album=self.device.media_album_name,
            image_url=self.device.media_image_url,
            duration=int(media_duration) if media_duration is not None else None,
        )
        if (media_position := self.device.media_position) is not None:
            self._attr_elapsed_time = float(media_position)
            if (updated_at := self.device.media_position_updated_at) is not None:
                self._attr_elapsed_time_last_updated = updated_at.timestamp()

    def _get_playback_state(self) -> PlaybackState:
        """Map the connected device's UPnP transport state onto a MA PlaybackState."""
        if self.device is None or self.device.transport_state is None:
            return PlaybackState.IDLE
        if self.device.transport_state in (
            TransportState.PLAYING,
            TransportState.TRANSITIONING,
        ):
            return PlaybackState.PLAYING
        if self.device.transport_state in (
            TransportState.PAUSED_PLAYBACK,
            TransportState.PAUSED_RECORDING,
        ):
            return PlaybackState.PAUSED
        return PlaybackState.IDLE

    async def _disconnect_device(self) -> None:
        """Unsubscribe from and drop the currently connected UPnP device, if any."""
        if self.device is None:
            return
        self.device.on_event = None
        old_device = self.device
        self.device = None
        self._current_location = None
        with suppress(UpnpError):
            await old_device.async_unsubscribe_services()

    async def _ensure_real_solo_zone(
        self, addressable_udn: str | None, topology: RaumfeldTopology
    ) -> tuple[str | None, RaumfeldTopology]:
        """
        Make sure a solo room's zone is a real, explicitly-created speaker group.

        Raumfeld exposes a "virtual media renderer" for every room by default, distinct
        from an actual speaker group, even one containing only that room - the project's
        wiki (https://github.com/B5r1oJ0A9G/teufel_raumfeld/wiki) says this outright:
        "While rooms technically have a media renderer, it is not foreseen to use them
        directly. Instead a speaker group with only one room is created." Confirmed live:
        the room-level renderer accepts SetAVTransportURI/Play and correctly reports back
        transport state and track metadata, but produces no audible output - only a zone
        formed via an explicit `connectRoomToZone` call is a genuine, audible endpoint.

        `connectRoomToZone` is destructive (it always creates a brand new zone UDN, even
        for an already-active room), so this must not run on every call - only once per
        distinct solo zone identity, tracked via `_confirmed_solo_zone_udn`. A room that
        is grouped with others is left alone: multi-room zones are only ever formed via
        an explicit `connectRoomsToZone` call already (this provider's own grouping, or
        the Raumfeld app/remote doing the same thing), so they do not have this problem.

        Since `connect()` (the only caller) runs this under `self.lock`, two calls for
        this room can never run this concurrently - but they can still each carry a
        *different* `topology` snapshot, fetched by each caller at a different time, and
        queue up waiting for the lock. A call carrying a snapshot from before an
        already-completed, now-confirmed re-creation would otherwise look like a fresh
        mismatch and trigger a second, bogus, destructive re-creation - silently
        orphaning whatever the first one had already started playing on. So once a zone
        has been confirmed once, a mismatch against the passed-in `topology` is treated
        as merely a hint to double check, not acted on directly: this re-verifies with a
        fresh, authoritative fetch before ever concluding a new zone is actually needed.

        :param addressable_udn: This room's currently resolved zone/room UDN, if any.
        :param topology: The topology `addressable_udn` was resolved from.
        :return: The (possibly refreshed) addressable UDN and topology to use instead.
        """
        if addressable_udn is None:
            return addressable_udn, topology
        zone = topology.zones.get(addressable_udn)
        is_solo = zone is None or len(zone.room_udns) <= 1
        if not is_solo or addressable_udn == self._confirmed_solo_zone_udn:
            return addressable_udn, topology
        if self._confirmed_solo_zone_udn is not None:
            topology = await self._prov.client.get_topology()
            addressable_udn = topology.addressable_udn_for_room(self.room_udn)
            if addressable_udn == self._confirmed_solo_zone_udn:
                return addressable_udn, topology
            if addressable_udn is None:
                return addressable_udn, topology
        await self._prov.client.connect_room_to_zone(self.room_udn)
        topology = await self._prov.client.get_topology()
        addressable_udn = topology.addressable_udn_for_room(self.room_udn)
        self._confirmed_solo_zone_udn = addressable_udn
        return addressable_udn, topology

    @property
    def _prov(self) -> TeufelRaumfeldPlayerProvider:
        """Return the (typed) Teufel Raumfeld provider for this player."""
        return cast("TeufelRaumfeldPlayerProvider", self.provider)
