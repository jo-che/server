"""Tests for how a Teufel Raumfeld room follows zone changes and handles playback commands."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from async_upnp_client.exceptions import UpnpError
from async_upnp_client.profiles.dlna import TransportState
from music_assistant_models.enums import MediaType, PlaybackState
from music_assistant_models.errors import PlayerUnavailableError
from music_assistant_models.player import PlayerMedia

from music_assistant.providers.teufel_raumfeld.player import TeufelRaumfeldPlayer
from music_assistant.providers.teufel_raumfeld.raumfeld_client import (
    RaumfeldDevice,
    RaumfeldRoom,
    RaumfeldTopology,
    RaumfeldZone,
)
from tests.common import MockProvider

if TYPE_CHECKING:
    from async_upnp_client.profiles.dlna import DmrDevice

ROOM = "uuid:Room-Living"
OTHER_ROOM = "uuid:Room-Kitchen"
ZONE_A = "uuid:Zone-A"
ZONE_B = "uuid:Zone-B"
MA_BASE_URL = "http://10.0.0.86:8097"

DMR_DEVICE = "music_assistant.providers.teufel_raumfeld.player.DmrDevice"


def _topology(zone_udn: str | None, *, zone_rooms: list[str] | None = None) -> RaumfeldTopology:
    """
    Return a topology in which the room is part of the given zone.

    :param zone_udn: UDN of the room's zone, or None while it is not part of any zone.
    :param zone_rooms: Rooms combined into that zone, defaults to just this room.
    """
    rooms = {ROOM: RaumfeldRoom(ROOM, "Living", "uuid:Renderer-Living", "ACTIVE", zone_udn)}
    zones: dict[str, RaumfeldZone] = {}
    devices: dict[str, RaumfeldDevice] = {}
    addressable = zone_udn or ROOM
    devices[addressable] = RaumfeldDevice(addressable, f"http://10.0.0.5/{addressable}.xml", "r")
    if zone_udn:
        zones[zone_udn] = RaumfeldZone(zone_udn, zone_rooms or [ROOM])
    return RaumfeldTopology(rooms=rooms, zones=zones, devices=devices)


def _provider() -> Any:
    """Return a provider stand-in with a mocked webservice client and UPnP factory."""
    provider: Any = MockProvider("teufel_raumfeld", instance_id="teufel_raumfeld_test")
    provider.client = MagicMock()
    provider.client.get_topology = AsyncMock()
    provider.client.connect_room_to_zone = AsyncMock()
    provider.client.leave_standby = AsyncMock()
    provider.upnp_factory = MagicMock()
    provider.upnp_factory.async_create_device = AsyncMock(return_value=MagicMock())
    provider.notify_server = MagicMock()
    provider.apply_zone_change = AsyncMock()
    confirmed: set[str] = set()
    provider.is_zone_confirmed = confirmed.__contains__
    provider.confirm_zone = confirmed.add
    provider.forget_zone = MagicMock()
    return provider


def _dmr_device() -> MagicMock:
    """Return a mocked UPnP renderer that accepts every command."""
    device = MagicMock()
    device.async_subscribe_services = AsyncMock()
    device.async_unsubscribe_services = AsyncMock()
    device.async_stop = AsyncMock()
    device.async_pause = AsyncMock()
    device.can_stop = True
    device.can_pause = True
    return device


def _player(provider: Any | None = None) -> TeufelRaumfeldPlayer:
    """
    Return a player for the room.

    :param provider: The provider stand-in to attach the player to.
    """
    return TeufelRaumfeldPlayer(provider or _provider(), ROOM, "Living")


async def test_connect_follows_the_rooms_zone() -> None:
    """The player controls the renderer of whatever zone its room is currently part of."""
    provider = _provider()
    player = _player(provider)

    with patch(DMR_DEVICE, return_value=_dmr_device()):
        await player.connect(_topology(ZONE_A))

    provider.upnp_factory.async_create_device.assert_awaited_once_with(
        f"http://10.0.0.5/{ZONE_A}.xml"
    )
    assert player.device is not None


async def test_connect_to_the_same_location_is_a_no_op() -> None:
    """Reconnecting to an unchanged zone keeps the existing device and subscription."""
    provider = _provider()
    player = _player(provider)

    with patch(DMR_DEVICE, return_value=_dmr_device()):
        await player.connect(_topology(ZONE_A))
        device = player.device
        await player.connect(_topology(ZONE_A))

    assert player.device is device
    provider.upnp_factory.async_create_device.assert_awaited_once()


async def test_zone_swap_resumes_the_queue_at_its_current_position() -> None:
    """
    A zone change while a queue plays resumes it through MA, not with the old stream URL.

    MA's stream server takes a repeated request for a flow URL as a reconnect and restarts
    the current track from its beginning; a queue resume starts at the current position.
    """
    provider = _provider()
    provider.mass.create_task = MagicMock()
    provider.mass.player_queues.resume = MagicMock(return_value="resume-coroutine")
    player = _player(provider)
    player._apply_transport_uri = AsyncMock()  # type: ignore[method-assign]
    media = MagicMock(source_id="queue-1")

    with patch(DMR_DEVICE, side_effect=[_dmr_device(), _dmr_device()]):
        await player.connect(_topology(ZONE_A))
        player._attr_playback_state = PlaybackState.PLAYING
        player._last_play_media = media
        player._last_play_url = "http://ma/flow/stream"
        await player.connect(_topology(ZONE_B))

    provider.mass.player_queues.resume.assert_called_once_with("queue-1")
    provider.mass.create_task.assert_called_once_with("resume-coroutine")
    player._apply_transport_uri.assert_not_awaited()
    assert provider.is_zone_confirmed(ZONE_B)


async def test_zone_swap_replays_media_without_a_queue() -> None:
    """Media that does not come from a queue is started again on the new zone directly."""
    provider = _provider()
    provider.mass.player_queues.get = MagicMock(return_value=None)
    player = _player(provider)
    player._apply_transport_uri = AsyncMock()  # type: ignore[method-assign]
    media = MagicMock(source_id=None)

    with patch(DMR_DEVICE, side_effect=[_dmr_device(), _dmr_device()]):
        await player.connect(_topology(ZONE_A))
        player._attr_playback_state = PlaybackState.PLAYING
        player._last_play_media = media
        player._last_play_url = "http://ma/stream"
        await player.connect(_topology(ZONE_B))

    player._apply_transport_uri.assert_awaited_once_with(media, "http://ma/stream")


async def test_play_media_without_a_reachable_zone_raises() -> None:
    """A room that cannot be connected reports the failure instead of silently not playing."""
    provider = _provider()
    provider.upnp_factory.async_create_device = AsyncMock(side_effect=UpnpError("gone"))
    provider.client.get_topology = AsyncMock(return_value=_topology(ZONE_A))
    provider.confirm_zone(ZONE_A)
    provider.mass.streams.resolve_stream_url = AsyncMock(return_value="http://ma/stream")
    player = _player(provider)

    with (
        patch("music_assistant.providers.teufel_raumfeld.player.asyncio.sleep"),
        pytest.raises(PlayerUnavailableError),
    ):
        await player.play_media(MagicMock())


async def test_connect_does_not_resume_after_an_explicit_stop() -> None:
    """A stopped room stays quiet when its zone changes afterwards."""
    player = _player()
    player._apply_transport_uri = AsyncMock()  # type: ignore[method-assign]

    with patch(DMR_DEVICE, side_effect=[_dmr_device(), _dmr_device()]):
        await player.connect(_topology(ZONE_A))
        player._attr_playback_state = PlaybackState.PLAYING
        player._last_play_media = MagicMock()
        player._last_play_url = "http://ma/stream"
        await player.stop()
        await player.connect(_topology(ZONE_B))

    player._apply_transport_uri.assert_not_awaited()


async def test_unreachable_zone_is_forgotten() -> None:
    """A zone that cannot be connected to is no longer treated as a real speaker group."""
    provider = _provider()
    provider.upnp_factory.async_create_device = AsyncMock(side_effect=UpnpError("gone"))
    player = _player(provider)

    await player.connect(_topology(ZONE_A))

    provider.forget_zone.assert_called_once_with(ZONE_A)
    assert player.device is None


async def test_real_solo_zone_is_created_for_an_unconfirmed_room() -> None:
    """A solo room gets an explicitly created zone, since its own renderer is silent."""
    provider = _provider()
    provider.client.get_topology = AsyncMock(return_value=_topology(ZONE_B))
    player = _player(provider)

    with patch("music_assistant.providers.teufel_raumfeld.player.asyncio.sleep"):
        udn, _topo = await player._ensure_real_solo_zone(ROOM, _topology(None))

    provider.client.connect_room_to_zone.assert_awaited_once_with(ROOM)
    assert udn == ZONE_B
    assert provider.is_zone_confirmed(ZONE_B)


@pytest.mark.parametrize(
    ("confirmed", "zone_rooms"),
    [(True, [ROOM]), (False, [ROOM, OTHER_ROOM])],
    ids=["confirmed_zone", "multi_room_zone"],
)
async def test_real_solo_zone_leaves_real_zones_alone(
    confirmed: bool, zone_rooms: list[str]
) -> None:
    """A confirmed or multi-room zone is never recreated, which would stop its playback."""
    provider = _provider()
    if confirmed:
        provider.confirm_zone(ZONE_A)
    player = _player(provider)

    udn, _topo = await player._ensure_real_solo_zone(
        ZONE_A, _topology(ZONE_A, zone_rooms=zone_rooms)
    )

    provider.client.connect_room_to_zone.assert_not_awaited()
    assert udn == ZONE_A


async def test_pause_stops_instead_of_pausing() -> None:
    """Pause stops the renderer, so resuming starts a fresh stream session."""
    player = _player()
    player.device = _dmr_device()
    player._last_play_media = MagicMock()

    await player.pause()

    player.device.async_stop.assert_awaited_once()
    player.device.async_pause.assert_not_awaited()
    assert player._last_play_media is None


async def test_set_members_extends_the_rooms_current_zone() -> None:
    """Grouping targets the leader's current zone and makes it the MA sync leader."""
    provider = _provider()
    provider.client.get_topology = AsyncMock(return_value=_topology(ZONE_A))
    player = _player(provider)

    await player.set_members(player_ids_to_add=[OTHER_ROOM])

    provider.apply_zone_change.assert_awaited_once_with(
        rooms_to_drop=[],
        rooms_to_group=sorted([ROOM, OTHER_ROOM]),
        target_zone_udn=ZONE_A,
        prefer_leader=ROOM,
    )


async def test_poll_of_a_standby_room_without_renderer_is_not_an_error() -> None:
    """A room in standby legitimately has no renderer to poll."""
    provider = _provider()
    provider.topology = RaumfeldTopology()
    player = _player(provider)
    player._attr_powered = False

    await player.poll()

    assert player.device is None


@pytest.mark.parametrize("powered", [True, None])
async def test_poll_of_an_awake_room_without_renderer_is_unavailable(
    powered: bool | None,
) -> None:
    """A room that should have a renderer but has none is reported unavailable."""
    provider = _provider()
    provider.topology = RaumfeldTopology()
    player = _player(provider)
    player._attr_powered = powered

    with pytest.raises(PlayerUnavailableError):
        await player.poll()


def _reporting_device(state: TransportState, reported_at: datetime) -> MagicMock:
    """
    Return a mocked renderer that reports position 0:00 as of the given time.

    :param state: The transport state the renderer reports.
    :param reported_at: When the renderer last stamped its position.
    """
    device = _dmr_device()
    device.transport_state = state
    device.current_track_uri = f"{MA_BASE_URL}/flow/session/queue/item/room.flac"
    device.media_title = device.media_artist = device.media_album_name = None
    device.media_image_url = None
    device.media_duration = None
    device.media_position = 0
    device.media_position_updated_at = reported_at
    return device


def test_position_from_before_a_resume_is_not_extrapolated() -> None:
    """
    A resume anchors the position to when playback restarted, not to its stale stamp.

    Mirrors a live recording: after pause (stop) and resume the renderer reported 0:00,
    stamped at the stop, until audio actually restarted seconds later.
    """
    stopped_at = datetime.now(UTC) - timedelta(seconds=9)
    player = _player()
    player.device = _reporting_device(TransportState.STOPPED, stopped_at)
    player._sync_from_device()

    player.device.transport_state = TransportState.PLAYING
    resumed = time.time()
    player._sync_from_device()

    assert player._attr_elapsed_time == 0
    assert player._attr_elapsed_time_last_updated is not None
    assert player._attr_elapsed_time_last_updated >= resumed


def test_position_does_not_run_while_the_stream_is_still_loading() -> None:
    """
    Playback is only counted from when audio starts, not from when loading began.

    Mirrors a live recording: STOPPED, then about 1-2s TRANSITIONING at 0:00 while the
    renderer loads the stream, then PLAYING at 0:00.
    """
    stopped_at = datetime.now(UTC) - timedelta(seconds=9)
    player = _player()
    player.device = _reporting_device(TransportState.STOPPED, stopped_at)
    player._sync_from_device()

    player.device.transport_state = TransportState.TRANSITIONING
    loading = time.time()
    player._sync_from_device()
    assert player._attr_elapsed_time_last_updated is not None
    assert player._attr_elapsed_time_last_updated >= loading

    player.device.transport_state = TransportState.PLAYING
    started = time.time()
    player._sync_from_device()
    assert player._attr_elapsed_time_last_updated >= started


def test_position_of_a_room_found_playing_keeps_its_own_stamp() -> None:
    """A room that was already playing when first seen has no resume to anchor to."""
    reported_at = datetime.now(UTC) - timedelta(seconds=5)
    player = _player()
    player.device = _reporting_device(TransportState.PLAYING, reported_at)

    player._sync_from_device()

    assert player._attr_elapsed_time_last_updated == reported_at.timestamp()


def _track() -> PlayerMedia:
    """Return the first track of a queue, the way MA hands it to the player."""
    return PlayerMedia(
        uri="library://track/1",
        media_type=MediaType.TRACK,
        title="First Track",
        duration=237,
        queue_item_id="item-1",
    )


async def _set_transport_uri(url: str) -> str:
    """
    Start the first track at the given stream URL and return the metadata sent along.

    :param url: The stream URL MA resolved for the track.
    """
    provider = _provider()
    provider.mass.streams.base_url = MA_BASE_URL
    player = _player(provider)
    player.device = _dmr_device()
    player.device.async_set_transport_uri = AsyncMock()
    player.device.async_wait_for_can_play = AsyncMock()
    player.device.async_play = AsyncMock()

    await player._apply_transport_uri(_track(), url)

    return str(player.device.async_set_transport_uri.call_args.args[2])


async def test_flow_stream_is_described_as_a_continuous_stream() -> None:
    """A flow stream is not described as its first track, which would expire in the app."""
    didl = await _set_transport_uri(f"{MA_BASE_URL}/flow/session/queue/item-1/room.flac")

    assert "audioBroadcast" in didl
    assert "First Track" not in didl
    assert "duration=" not in didl


async def test_position_is_stamped_once_play_was_sent() -> None:
    """The seconds it takes to load the stream do not count as time already played."""
    provider = _provider()
    provider.mass.streams.base_url = MA_BASE_URL
    player = _player(provider)
    player.device = _dmr_device()
    player.device.async_set_transport_uri = AsyncMock()
    player.device.async_wait_for_can_play = AsyncMock()
    play_sent: list[float] = []
    player.device.async_play = AsyncMock(side_effect=lambda: play_sent.append(time.time()))

    await player._apply_transport_uri(_track(), f"{MA_BASE_URL}/flow/s/q/item-1/room.flac")

    assert player._attr_elapsed_time == 0
    assert player._attr_elapsed_time_last_updated is not None
    assert player._attr_elapsed_time_last_updated >= play_sent[0]


async def test_single_track_stream_keeps_its_track_metadata() -> None:
    """A stream of a single track is still described as that track."""
    didl = await _set_transport_uri(f"{MA_BASE_URL}/single/session/queue/item-1/room.flac")

    assert "First Track" in didl


PLAYER = "music_assistant.providers.teufel_raumfeld.player"


async def test_play_media_stops_then_starts_the_stream_on_the_zone() -> None:
    """Playing connects to the room's zone, stops what plays there and starts the stream."""
    provider = _provider()
    provider.client.get_topology = AsyncMock(return_value=_topology(ZONE_A))
    provider.confirm_zone(ZONE_A)
    provider.mass.streams.resolve_stream_url = AsyncMock(return_value="http://ma/stream")
    player = _player(provider)
    player._apply_transport_uri = AsyncMock()  # type: ignore[method-assign]
    device = _dmr_device()
    order: list[str] = []
    device.async_stop = AsyncMock(side_effect=lambda: order.append("stop"))
    player._apply_transport_uri.side_effect = lambda *_: order.append("start")
    media = _track()

    with (
        patch(DMR_DEVICE, return_value=device),
        patch.object(TeufelRaumfeldPlayer, "update_state") as update_state,
    ):
        await player.play_media(media)

    assert order == ["stop", "start"]
    player._apply_transport_uri.assert_awaited_once_with(media, "http://ma/stream")
    assert player._last_play_media is media
    assert player._last_play_url == "http://ma/stream"
    update_state.assert_called_once()


async def test_volume_and_mute_target_this_room_within_its_zone() -> None:
    """Volume and mute go to this room through the zone, not to the whole zone."""
    player = _player()
    player.device = _dmr_device()

    with (
        patch(f"{PLAYER}.set_room_volume", AsyncMock()) as set_volume,
        patch(f"{PLAYER}.set_room_mute", AsyncMock()) as set_mute,
        patch.object(TeufelRaumfeldPlayer, "update_state"),
    ):
        await player.volume_set(30)
        await player.volume_mute(True)

    set_volume.assert_awaited_once_with(player.device.device, ROOM, 30)
    set_mute.assert_awaited_once_with(player.device.device, ROOM, True)
    assert player._attr_volume_level == 30
    assert player._attr_volume_muted is True


async def test_poll_reads_state_and_this_rooms_volume() -> None:
    """A poll refreshes the renderer's state and this room's own volume and mute."""
    player = _player()
    player.device = _reporting_device(TransportState.PLAYING, datetime.now(UTC))
    player.device.async_update = AsyncMock()

    with (
        patch(f"{PLAYER}.get_room_volume", AsyncMock(return_value=35)),
        patch(f"{PLAYER}.get_room_mute", AsyncMock(return_value=False)),
        patch.object(TeufelRaumfeldPlayer, "update_state") as update_state,
    ):
        await player.poll()

    player.device.async_update.assert_awaited_once()
    assert player._attr_playback_state == PlaybackState.PLAYING
    assert player._attr_volume_level == 35
    assert player._attr_volume_muted is False
    update_state.assert_called_once()


async def test_poll_of_a_vanished_renderer_reports_unavailable() -> None:
    """A renderer that stops answering is dropped and the player reported unavailable."""
    player = _player()
    device = _dmr_device()
    device.async_update = AsyncMock(side_effect=UpnpError("gone"))
    player.device = device

    with pytest.raises(PlayerUnavailableError):
        await player.poll()

    assert cast("DmrDevice | None", player.device) is None
    device.async_unsubscribe_services.assert_awaited_once()


def _event(service_id: str, name: str, value: Any) -> tuple[MagicMock, list[MagicMock]]:
    """
    Return a UPnP event of one changed state variable.

    :param service_id: The id of the service the event comes from.
    :param name: The name of the changed state variable.
    :param value: Its new value.
    """
    service = MagicMock(service_id=service_id)
    variable = MagicMock(value=value)
    variable.name = name
    return service, [variable]


@pytest.mark.parametrize("state", [TransportState.PLAYING, TransportState.PAUSED_PLAYBACK])
def test_playing_or_paused_event_triggers_a_poll(state: TransportState) -> None:
    """Starting or pausing is followed by a full poll, for position and track info."""
    provider = _provider()
    provider.mass.create_task = MagicMock()
    player = _player(provider)

    player._handle_event(*_event("urn:upnp-org:serviceId:AVTransport", "TransportState", state))

    assert player.force_poll is True
    provider.mass.create_task.assert_called_once()


def test_other_event_updates_without_a_poll() -> None:
    """Other changes are taken over from the event itself."""
    provider = _provider()
    provider.mass.create_task = MagicMock()
    player = _player(provider)

    player._handle_event(*_event("urn:upnp-org:serviceId:RenderingControl", "Volume", 30))

    assert player.force_poll is False
    provider.mass.create_task.assert_called_once()


@pytest.mark.parametrize("force_poll", [True, False])
async def test_update_after_event_polls_only_when_asked(force_poll: bool) -> None:
    """After an event, a poll runs only if the event asked for one."""
    player = _player()
    player.force_poll = force_poll
    player.poll = AsyncMock()  # type: ignore[method-assign]

    with patch.object(TeufelRaumfeldPlayer, "update_state") as update_state:
        await player._async_update_after_event()

    assert player.poll.await_count == (1 if force_poll else 0)
    assert update_state.call_count == (0 if force_poll else 1)


async def test_room_gone_from_the_topology_becomes_unavailable() -> None:
    """A room the host no longer knows is reported unavailable."""
    player = _player()

    with patch.object(TeufelRaumfeldPlayer, "update_state") as update_state:
        await player.refresh_from_topology(RaumfeldTopology())

    assert player._attr_available is False
    update_state.assert_called_once()
