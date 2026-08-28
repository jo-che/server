"""
Tests that the artwork the playlog captured follows the item when it changes.

A play stores the item's name and thumb on the playlog row, and 'Recently played' serves
those columns straight back. Picking different artwork therefore has to reach those rows,
or the tile keeps showing what the item looked like when it was played.
See music-assistant/support#6216.
"""

from __future__ import annotations

from typing import Any

from music_assistant_models.enums import ImageType, MediaType
from music_assistant_models.media_items import Album, MediaItemImage, UniqueList

from music_assistant.constants import DB_TABLE_PLAYLOG
from music_assistant.controllers.music.media.base import SUPPRESS_MEDIA_ITEM_UPDATES
from music_assistant.helpers.json import json_loads, serialize_to_json
from music_assistant.mass import MusicAssistant

from .helpers import create_album

ALBUM_PROVIDER = "filesystem_local--AbCd"
ALBUM_ID = "album-001"
OLD_ART = "https://theaudiodb.example/old.jpg"
NEW_ART = "/media/music/Album/folder.jpg"


async def test_new_artwork_reaches_the_playlog_row(mass: MusicAssistant) -> None:
    """Choosing different artwork updates what a recently played tile will show."""
    db_album = await _add_library_album(mass, OLD_ART, provider="theaudiodb")
    await _add_playlog_row(mass, "library", db_album.item_id, OLD_ART)

    await _set_thumb(mass, db_album, NEW_ART, provider=ALBUM_PROVIDER)

    assert await _playlog_image_path(mass, "library", db_album.item_id) == NEW_ART


async def test_a_row_logged_under_the_provider_is_refreshed_too(mass: MusicAssistant) -> None:
    """A play logged against the provider identity stores the same thumb, so it goes stale too."""
    db_album = await _add_library_album(mass, OLD_ART, provider="theaudiodb")
    await _add_playlog_row(mass, ALBUM_PROVIDER, ALBUM_ID, OLD_ART)

    await _set_thumb(mass, db_album, NEW_ART, provider=ALBUM_PROVIDER)

    assert await _playlog_image_path(mass, ALBUM_PROVIDER, ALBUM_ID) == NEW_ART


async def test_a_renamed_item_updates_its_playlog_row(mass: MusicAssistant) -> None:
    """The name is captured alongside the thumb, so it has to follow the item as well."""
    db_album = await _add_library_album(mass, OLD_ART, provider="theaudiodb")
    await _add_playlog_row(mass, "library", db_album.item_id, OLD_ART)

    db_album.name = "Renamed Album"
    await mass.music.albums.update_item_in_library(db_album.item_id, db_album, overwrite=True)

    row = await _playlog_row(mass, "library", db_album.item_id)
    assert row is not None
    assert row["name"] == "Renamed Album"


async def test_artwork_picked_up_by_a_sync_reaches_the_playlog_row(mass: MusicAssistant) -> None:
    """A provider sync finding better artwork has to reach the row as well."""
    db_album = await _add_library_album(mass, OLD_ART, provider="theaudiodb")
    await _add_playlog_row(mass, "library", db_album.item_id, OLD_ART)

    token = SUPPRESS_MEDIA_ITEM_UPDATES.set(True)
    try:
        await _set_thumb(mass, db_album, NEW_ART, provider=ALBUM_PROVIDER)
    finally:
        SUPPRESS_MEDIA_ITEM_UPDATES.reset(token)

    assert await _playlog_image_path(mass, "library", db_album.item_id) == NEW_ART


async def test_an_item_that_was_never_played_is_left_alone(mass: MusicAssistant) -> None:
    """Nothing was captured for it, so an update must not create a playlog row."""
    db_album = await _add_library_album(mass, OLD_ART, provider="theaudiodb")

    await _set_thumb(mass, db_album, NEW_ART, provider=ALBUM_PROVIDER)

    assert await _playlog_row(mass, "library", db_album.item_id) is None


async def _add_library_album(mass: MusicAssistant, path: str, provider: str) -> Album:
    """
    Add the album under test to the library carrying a single thumb.

    :param mass: The MusicAssistant instance to seed.
    :param path: The path or url of the thumb to store.
    :param provider: The provider the thumb is attributed to.
    """
    album = create_album(ALBUM_PROVIDER, ALBUM_ID)
    album.metadata.images = UniqueList(
        [
            MediaItemImage(
                type=ImageType.THUMB,
                path=path,
                provider=provider,
                remotely_accessible=path.startswith("http"),
            )
        ]
    )
    return await mass.music.albums.add_item_to_library(album)


async def _set_thumb(mass: MusicAssistant, album: Album, path: str, provider: str) -> None:
    """
    Store a different thumb for the album, the way choosing preferred artwork does.

    :param mass: The MusicAssistant instance holding the album.
    :param album: The library album to restore.
    :param path: The path or url of the thumb to store.
    :param provider: The provider the thumb is attributed to.
    """
    album.metadata.images = UniqueList(
        [
            MediaItemImage(
                type=ImageType.THUMB,
                path=path,
                provider=provider,
                remotely_accessible=path.startswith("http"),
            )
        ]
    )
    await mass.music.albums.update_item_in_library(album.item_id, album, overwrite=True)


async def _add_playlog_row(mass: MusicAssistant, provider: str, item_id: str, path: str) -> None:
    """
    Seed the playlog row a play of the album would have written.

    :param mass: The MusicAssistant instance to seed.
    :param provider: The provider the play is logged under.
    :param item_id: The item id the play is logged under.
    :param path: The path or url of the thumb captured with the play.
    """
    await mass.music.database.insert(
        DB_TABLE_PLAYLOG,
        {
            "item_id": item_id,
            "provider": provider,
            "media_type": MediaType.ALBUM.value,
            "name": "Test Album",
            "image": serialize_to_json(
                MediaItemImage(
                    type=ImageType.THUMB,
                    path=path,
                    provider="theaudiodb",
                    remotely_accessible=True,
                ).to_dict()
            ),
            "userid": "test-user",
            "seconds_played": 120,
            "fully_played": True,
            "timestamp": 1000,
        },
        allow_replace=True,
    )


async def _playlog_row(mass: MusicAssistant, provider: str, item_id: str) -> dict[str, Any] | None:
    """
    Return the playlog row for the given identity, or None when there is none.

    :param mass: The MusicAssistant instance to read from.
    :param provider: The provider the play was logged under.
    :param item_id: The item id the play was logged under.
    """
    rows = await mass.music.database.get_rows(
        DB_TABLE_PLAYLOG,
        {
            "media_type": MediaType.ALBUM.value,
            "provider": provider,
            "item_id": item_id,
        },
    )
    return dict(rows[0]) if rows else None


async def _playlog_image_path(mass: MusicAssistant, provider: str, item_id: str) -> str | None:
    """
    Return the path of the thumb the playlog captured for the given identity.

    :param mass: The MusicAssistant instance to read from.
    :param provider: The provider the play was logged under.
    :param item_id: The item id the play was logged under.
    """
    row = await _playlog_row(mass, provider, item_id)
    if not row or not row["image"]:
        return None
    return str(json_loads(row["image"])["path"])
