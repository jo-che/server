"""Teufel Raumfeld Player Provider for Music Assistant."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING, cast

from async_upnp_client.aiohttp import AiohttpSessionRequester
from async_upnp_client.client_factory import UpnpFactory
from music_assistant_models.errors import SetupFailedError

from music_assistant.helpers.json import SerializableType
from music_assistant.models.player_provider import PlayerProvider

from .const import CONF_HOST, CONF_PORT
from .helpers import RaumfeldNotifyServer, get_device_model
from .player import TeufelRaumfeldPlayer
from .raumfeld_client import RaumfeldTopology, RaumfeldWebserviceClient

# a zone change can be reflected in /getZones slightly before the corresponding
# device's location shows up in /listDevices - refresh_topology() retries within this
# budget so it does not hand back (and reconcile against) a room transiently missing
# its device location right after a command that itself just changed the zone it's in.
_TOPOLOGY_SETTLE_ATTEMPTS = 6
_TOPOLOGY_SETTLE_DELAY = 0.4

if TYPE_CHECKING:
    from async_upnp_client.client import UpnpRequester
    from music_assistant_models.config_entries import ConfigEntry


class TeufelRaumfeldPlayerProvider(PlayerProvider):
    """Player provider for Teufel Raumfeld multiroom systems."""

    client: RaumfeldWebserviceClient
    requester: UpnpRequester
    upnp_factory: UpnpFactory
    notify_server: RaumfeldNotifyServer
    topology: RaumfeldTopology
    _players: dict[str, TeufelRaumfeldPlayer]
    _watch_tasks: list[asyncio.Task[None]]
    _reconcile_lock: asyncio.Lock
    # Zone UDNs known to be real, audible speaker groups rather than a room's
    # discouraged default "virtual media renderer" (see
    # TeufelRaumfeldPlayer._ensure_real_solo_zone) - there is no protocol-level way to
    # tell the two apart from the topology alone, so this is built from every zone this
    # provider has ever observed with more than one room in it (formed by us or by
    # another controller, e.g. the Raumfeld app) plus every zone this provider has
    # itself explicitly (re)created. Entries are not removed except via `forget_zone`
    # (a failed connection proving the zone is actually dead): a zone that shrinks back
    # to one room after a member leaves is still the same real zone, and must not be
    # reactively destroyed and recreated by _reconcile - that stops whatever was still
    # playing on it for no reason.
    _confirmed_zones: set[str]
    # hardware model names keyed by physical renderer UDN - they never change, so each
    # description is fetched only once (see `get_device_model`)
    _device_models: dict[str, str]

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """
        Return the (options) config entries for this (existing) provider instance.

        The host/port are collected once via `setup_flow.py`; there are no further
        provider-level options.
        """
        return ()

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        self._players = {}
        self._watch_tasks = []
        self._reconcile_lock = asyncio.Lock()
        self._confirmed_zones = set()
        self._device_models = {}
        host = cast("str", self.get_setup_value(CONF_HOST))
        port = cast("int", self.get_setup_value(CONF_PORT))
        self.client = RaumfeldWebserviceClient(host, port, self.mass.http_session)
        if not await self.client.ping():
            raise SetupFailedError(f"Unable to reach a Raumfeld host webservice at {host}:{port}")
        self.requester = AiohttpSessionRequester(self.mass.http_session, with_sleep=True)
        self.upnp_factory = UpnpFactory(self.requester, non_strict=True)
        self.notify_server = RaumfeldNotifyServer(self.requester, self.mass)
        self.topology = await self.client.get_topology()

    async def loaded_in_mass(self) -> None:
        """Call after the provider has been loaded."""
        await self.discover_players()
        self._watch_tasks = [
            self.mass.create_task(self._watch_topology("/getZones")),
            self.mass.create_task(self._watch_topology("/listDevices")),
        ]

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/close of the provider."""
        for task in self._watch_tasks:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._watch_tasks = []
        for player in list(self._players.values()):
            await self.mass.players.unregister(player.player_id)
        self._players.clear()

    async def discover_players(self) -> None:
        """Register a player for every currently known Raumfeld room."""
        for room_udn, room in self.topology.rooms.items():
            if room_udn in self._players:
                continue
            player = TeufelRaumfeldPlayer(self, room_udn, room.name)
            self._players[room_udn] = player
            await player.update_room_info(room, self.topology)
            await player.connect(self.topology)
            await self.mass.players.register_or_update(player)

    async def refresh_topology(self) -> None:
        """Re-fetch the topology and reconcile players against it (see `_settle_and_reconcile`)."""
        async with self._reconcile_lock:
            await self._settle_and_reconcile()

    async def apply_zone_change(
        self,
        rooms_to_drop: list[str],
        rooms_to_group: list[str],
        target_zone_udn: str | None = None,
        prefer_leader: str | None = None,
    ) -> None:
        """
        Apply a zone membership change, then reconcile - atomically.

        Issuing the webservice command(s) and the settle-and-reconcile that follows run
        under `_reconcile_lock` as a single step, so a background topology watcher can
        never reconcile the (still incomplete or not-yet-preference-carrying) result of
        only some of this call's own webservice calls, and `prefer_leader` always applies
        to the very first reconcile pass that sees the finished change.

        :param rooms_to_drop: UDNs of rooms to drop from whatever zone each is in.
        :param rooms_to_group: UDNs of rooms that should end up combined into one zone
            together; only issued to the webservice if there is more than one.
        :param target_zone_udn: UDN of an existing zone to add `rooms_to_group` to,
            instead of forming a brand new zone. Pass the current (sync leader) room's
            own zone UDN when joining a room to an already-playing zone: the webservice
            call is destructive without this (it always creates a fresh zone UDN and
            tears down whatever renderer instance was there before), which otherwise
            silently stops playback on everyone already in that zone.
        :param prefer_leader: UDN of the room that should become the MA sync leader of
            whatever zone it ends up in, if any (see `_pick_leader`). Pass this from a
            command that just changed grouping to make the room the command was issued
            on the leader, rather than falling back to the deterministic tie-break.
        """
        async with self._reconcile_lock:
            for room_udn in rooms_to_drop:
                await self.client.drop_room(room_udn)
            if len(rooms_to_group) > 1:
                await self.client.connect_rooms_to_zone(rooms_to_group, zone_udn=target_zone_udn)
            await self._settle_and_reconcile(prefer_leader=prefer_leader)

    def is_zone_confirmed(self, zone_udn: str) -> bool:
        """Return whether `zone_udn` is known to be a real, audible speaker group."""
        return zone_udn in self._confirmed_zones

    def confirm_zone(self, zone_udn: str) -> None:
        """Mark `zone_udn` as a real, audible speaker group (see `_confirmed_zones`)."""
        self._confirmed_zones.add(zone_udn)

    def forget_zone(self, zone_udn: str) -> None:
        """
        Un-confirm `zone_udn` after a failed connection proves it is actually dead.

        The one exception to `_confirmed_zones` entries never being removed: a zone a
        room could not actually connect to is not a real, currently-usable endpoint
        regardless of what the topology still says, so the next playback attempt should
        try to create a fresh one instead of reactively retrying the same dead zone.
        """
        self._confirmed_zones.discard(zone_udn)

    async def get_device_model(self, renderer_udn: str, topology: RaumfeldTopology) -> str | None:
        """
        Return the hardware model name of a room's physical renderer, if it can be read.

        :param renderer_udn: UDN of the room's physical renderer (not its zone renderer).
        :param topology: The topology snapshot to resolve the renderer's location from.
        """
        if model := self._device_models.get(renderer_udn):
            return model
        if not (location := topology.location_for(renderer_udn)):
            return None
        if model := await get_device_model(self.mass.http_session, location):
            self._device_models[renderer_udn] = model
        return model

    async def get_diagnostics(self) -> dict[str, SerializableType]:
        """Return diagnostics info for this provider to include in diagnostics reports."""
        return {
            "host": self.client.host,
            "port": self.client.port,
            "rooms": {
                udn: {"name": room.name, "zone_udn": room.zone_udn, "power": room.power_state}
                for udn, room in self.topology.rooms.items()
            },
            "zones": {udn: zone.room_udns for udn, zone in self.topology.zones.items()},
            "models": dict(self._device_models),
        }

    async def _watch_topology(self, path: str) -> None:
        """Long-poll a topology endpoint forever, reconciling players on every change."""
        while True:
            try:
                async for _xml in self.client.long_poll(path):
                    async with self._reconcile_lock:
                        await self._settle_and_reconcile()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("Error while watching %s, retrying shortly", path)
                await asyncio.sleep(5)

    async def _settle_and_reconcile(self, prefer_leader: str | None = None) -> None:
        """
        Re-fetch the topology and reconcile players against it.

        Must be called with `_reconcile_lock` held. Retries briefly (see
        `_TOPOLOGY_SETTLE_ATTEMPTS`) until every currently known room resolves a device
        location, to ride out the brief lag `/listDevices` can have right after a zone
        change this same call chain (usually `group_rooms`/`drop_rooms`) just made.

        :param prefer_leader: See `group_rooms`.
        """
        topology = self.topology
        for attempt in range(_TOPOLOGY_SETTLE_ATTEMPTS):
            topology = await self.client.get_topology()
            if all(
                topology.location_for(topology.addressable_udn_for_room(room_udn))
                for room_udn in topology.rooms
            ):
                break
            if attempt + 1 < _TOPOLOGY_SETTLE_ATTEMPTS:
                await asyncio.sleep(_TOPOLOGY_SETTLE_DELAY)
        self.topology = topology
        await self._reconcile(prefer_leader=prefer_leader)

    async def _reconcile(self, prefer_leader: str | None = None) -> None:
        """
        Reconcile registered players and their group membership against `self.topology`.

        Must be called with `_reconcile_lock` held.

        :param prefer_leader: See `group_rooms`.
        """
        await self.discover_players()
        for zone_udn, known_zone in self.topology.zones.items():
            if len(known_zone.room_udns) > 1:
                self._confirmed_zones.add(zone_udn)
        for room_udn, player in list(self._players.items()):
            room = self.topology.rooms.get(room_udn)
            if room is None:
                await self.mass.players.unregister(player.player_id)
                del self._players[room_udn]
                continue
            zone = self.topology.zones.get(room.zone_udn) if room.zone_udn else None
            zone_room_udns = zone.room_udns if zone else []
            if len(zone_room_udns) > 1:
                leader_udn = self._pick_leader(zone_room_udns, prefer=prefer_leader)
                if room_udn == leader_udn:
                    others = [udn for udn in zone_room_udns if udn != room_udn]
                    player._attr_group_members = [room_udn, *others]
                else:
                    player._attr_group_members = []
            else:
                player._attr_group_members = []
            await player.refresh_from_topology(self.topology)

    def _pick_leader(self, room_udns: list[str], prefer: str | None = None) -> str:
        """
        Pick the MA sync-group leader for a zone's current room set.

        Prefers `prefer` if given and a member of this zone (the room a grouping
        command was just issued on). Otherwise prefers whichever room already claims
        leadership of exactly this room set, so a previously established leader stays
        stable across reconciliation passes that were not triggered by a grouping
        command. Falls back to the lexicographically first room UDN for zones formed
        outside Music Assistant (the Raumfeld app, a physical remote), so the pick is
        still stable and deterministic even without an established MA-side leader.

        :param room_udns: UDNs of every room currently combined into the zone.
        :param prefer: UDN of the room to prefer as leader, if it is a member of this zone.
        """
        room_set = set(room_udns)
        if prefer is not None and prefer in room_set:
            return prefer
        for udn in room_udns:
            if (player := self._players.get(udn)) and set(player._attr_group_members) == room_set:
                return udn
        return min(room_udns)
