"""
Native Player Provider for Teufel Raumfeld multiroom speakers.

Talks directly to the Raumfeld system: the host webservice for room/zone topology and
grouping, and standard UPnP AVTransport/RenderingControl (plus one vendor per-room-volume
extension) directly against each room's resolved renderer for playback control. See
`raumfeld_client/webservice.py` for the protocol notes and this provider's design
references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from music_assistant_models.enums import ProviderFeature

from .provider import TeufelRaumfeldPlayerProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

SUPPORTED_FEATURES = {
    ProviderFeature.SYNC_PLAYERS,
}


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return TeufelRaumfeldPlayerProvider(mass, manifest, config, SUPPORTED_FEATURES)
