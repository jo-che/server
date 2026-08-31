"""Tests for resolving references to a provider instance that no longer exists."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from music_assistant_models.config_entries import ProviderConfig
from music_assistant_models.enums import MediaType, ProviderType
from music_assistant_models.errors import ProviderUnavailableError
from music_assistant_models.media_items import ProviderMapping, Track
from music_assistant_models.provider import ProviderManifest

from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from music_assistant.mass import MusicAssistant

FAKE_DOMAIN = "beatstream"
NEW_INSTANCE = "beatstream--EecMYDCu"
OLD_INSTANCE = "beatstream--aetxVqCU"
TRACK_ID = "track1"


class FakeStreamingProvider(MusicProvider):
    """Streaming-style provider that owns a single track."""

    async def sync_library(self, media_type: MediaType) -> None:
        """No-op sync implementation for tests."""

    async def get_track(self, prov_track_id: str) -> Track:
        """Return the track this provider owns."""
        return Track(
            item_id=prov_track_id,
            provider=NEW_INSTANCE,
            name="Provider Owned Track",
            provider_mappings={
                ProviderMapping(
                    item_id=prov_track_id,
                    provider_domain=FAKE_DOMAIN,
                    provider_instance=NEW_INSTANCE,
                )
            },
        )


def _mock_provider(instance_id: str, domain: str, is_streaming: bool) -> Mock:
    """Return a mocked, available music provider."""
    prov = Mock(spec=MusicProvider)
    prov.instance_id = instance_id
    prov.domain = domain
    prov.available = True
    prov.is_streaming_provider = is_streaming
    return prov


@pytest.fixture(name="stream_mass")
async def stream_mass_fixture(mass: MusicAssistant) -> AsyncGenerator[MusicAssistant]:
    """Return a booted instance with a fake streaming provider loaded."""
    provider = FakeStreamingProvider(
        mass,
        manifest=ProviderManifest(
            type=ProviderType.MUSIC,
            domain=FAKE_DOMAIN,
            name="Beatstream",
            description="Fake streaming provider",
            codeowners=["@music-assistant"],
        ),
        config=ProviderConfig(
            values={},
            type=ProviderType.MUSIC,
            domain=FAKE_DOMAIN,
            instance_id=NEW_INSTANCE,
            name="Beatstream",
        ),
    )
    provider.available = True
    mass._providers[NEW_INSTANCE] = provider
    try:
        yield mass
    finally:
        mass._providers.pop(NEW_INSTANCE, None)


async def test_loaded_instance_is_returned_unchanged(mass_minimal: MusicAssistant) -> None:
    """A reference to a loaded instance must keep pointing at that exact instance."""
    mass_minimal._providers[NEW_INSTANCE] = _mock_provider(NEW_INSTANCE, FAKE_DOMAIN, True)

    assert mass_minimal.resolve_provider_reference(NEW_INSTANCE) == NEW_INSTANCE


async def test_domain_is_returned_unchanged(mass_minimal: MusicAssistant) -> None:
    """A reference that is already a domain has nothing to resolve."""
    mass_minimal._providers[FAKE_DOMAIN] = _mock_provider(FAKE_DOMAIN, FAKE_DOMAIN, True)

    assert mass_minimal.resolve_provider_reference(FAKE_DOMAIN) == FAKE_DOMAIN


async def test_deleted_instance_resolves_to_domain(mass_minimal: MusicAssistant) -> None:
    """A reference to a deleted instance falls back to the domain of its replacement."""
    mass_minimal._providers[NEW_INSTANCE] = _mock_provider(NEW_INSTANCE, FAKE_DOMAIN, True)

    assert mass_minimal.resolve_provider_reference(OLD_INSTANCE) == FAKE_DOMAIN


async def test_the_redirect_is_reported_once(
    mass_minimal: MusicAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """The redirect is visible in the log, without repeating on every lookup."""
    mass_minimal._providers[NEW_INSTANCE] = _mock_provider(NEW_INSTANCE, FAKE_DOMAIN, True)

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            mass_minimal.resolve_provider_reference(OLD_INSTANCE)

    warnings = [record for record in caplog.records if OLD_INSTANCE in record.getMessage()]
    assert len(warnings) == 1
    assert NEW_INSTANCE in warnings[0].getMessage()


async def test_an_unchanged_reference_is_not_reported(
    mass_minimal: MusicAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing is reported when the reference is handed back as it came in."""
    mass_minimal._providers["filesystem_local--new"] = _mock_provider(
        "filesystem_local--new", "filesystem_local", False
    )

    with caplog.at_level(logging.WARNING):
        mass_minimal.resolve_provider_reference("filesystem_local--old")
        mass_minimal.resolve_provider_reference(NEW_INSTANCE)

    assert caplog.records == []


async def test_deleted_instance_of_local_provider_is_kept(mass_minimal: MusicAssistant) -> None:
    """Item ids of a non-streaming provider are instance specific, so no fallback."""
    mass_minimal._providers["filesystem_local--new"] = _mock_provider(
        "filesystem_local--new", "filesystem_local", False
    )

    assert (
        mass_minimal.resolve_provider_reference("filesystem_local--old") == "filesystem_local--old"
    )


async def test_deleted_instance_without_replacement_is_kept(mass_minimal: MusicAssistant) -> None:
    """Without a provider of the same domain there is nothing to fall back to."""
    assert mass_minimal.resolve_provider_reference(OLD_INSTANCE) == OLD_INSTANCE


async def test_get_item_resolves_deleted_instance(stream_mass: MusicAssistant) -> None:
    """An item id stored against a deleted instance still resolves to the item."""
    track = await stream_mass.music.get_item(MediaType.TRACK, TRACK_ID, OLD_INSTANCE)

    assert track.name == "Provider Owned Track"


async def test_get_item_reports_unknown_provider(stream_mass: MusicAssistant) -> None:
    """A reference to a provider that was never configured stays an error."""
    with pytest.raises(ProviderUnavailableError):
        await stream_mass.music.get_item(MediaType.TRACK, TRACK_ID, "gonestream--old")
