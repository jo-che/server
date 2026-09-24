"""mDNS discovery of Raumfeld host webservices."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zeroconf import ServiceStateChange
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo

from music_assistant.helpers.util import get_primary_ip_address_from_zeroconf

from .raumfeld_client import DEFAULT_PORT

if TYPE_CHECKING:
    from zeroconf import Zeroconf

# The host announces its webservice as a plain HTTP service named e.g. "RaumfeldControl
# on Teufel One S", without TXT records - the name prefix is the only marker (checked on
# a live system). Speakers that are not the host announce no such service.
MDNS_TYPE = "_http._tcp.local."
MDNS_NAME_PREFIX = "RaumfeldControl on "

# Hosts answer the same query at about the same time, so after the first answer only a
# short extra wait is needed to also catch any further host on the network.
_MORE_HOSTS_WAIT = 0.5
_RESOLVE_TIMEOUT_MS = 1000


@dataclass(frozen=True)
class DiscoveredHost:
    """A Raumfeld host webservice found on the network."""

    name: str  # the model of the speaker acting as host, e.g. "Teufel One S"
    address: str
    hostname: str | None  # its mDNS hostname, e.g. "one-s.local"
    port: int


async def discover_hosts(zeroconf: Zeroconf, timeout: float) -> list[DiscoveredHost]:
    """
    Return every Raumfeld host webservice announcing itself on the network.

    Returns shortly after a first host has answered, or empty-handed after `timeout`.

    :param zeroconf: The Zeroconf instance to browse with.
    :param timeout: Maximum time in seconds to wait for a host to answer.
    """
    names: set[str] = set()
    found = asyncio.Event()

    def on_service_state_change(
        zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange
    ) -> None:
        del zeroconf, service_type
        if state_change is not ServiceStateChange.Removed and name.lower().startswith(
            MDNS_NAME_PREFIX.lower()
        ):
            names.add(name)
            found.set()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    browser = AsyncServiceBrowser(zeroconf, [MDNS_TYPE], handlers=[on_service_state_change])
    try:
        await asyncio.wait_for(found.wait(), timeout)
        await asyncio.sleep(max(0.0, min(_MORE_HOSTS_WAIT, deadline - loop.time())))
    except TimeoutError:
        return []
    finally:
        await browser.async_cancel()
    hosts: list[DiscoveredHost] = []
    for name in sorted(names):
        info = AsyncServiceInfo(MDNS_TYPE, name)
        if not await info.async_request(zeroconf, _RESOLVE_TIMEOUT_MS):
            continue
        if address := get_primary_ip_address_from_zeroconf(info):
            hosts.append(
                DiscoveredHost(
                    name=name.removesuffix(f".{MDNS_TYPE}")[len(MDNS_NAME_PREFIX) :],
                    address=address,
                    hostname=(info.server or "").rstrip(".") or None,
                    port=info.port or DEFAULT_PORT,
                )
            )
    return hosts
