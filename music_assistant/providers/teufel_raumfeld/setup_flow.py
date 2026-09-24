"""Setup flow for the Teufel Raumfeld provider."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType

from music_assistant.models.setup_flow import SetupFlowError

from .const import CONF_HOST, CONF_PORT, DOMAIN
from .discovery import DiscoveredHost, discover_hosts
from .raumfeld_client import DEFAULT_PORT, RaumfeldWebserviceClient

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant
    from music_assistant.models.setup_flow import SetupSession

LOGGER = logging.getLogger(__name__)

# how long the form waits for a Raumfeld host to answer on mDNS before it is shown with
# an empty host; kept short so setup does not appear to hang on a network without one.
_DISCOVERY_TIMEOUT = 3.0
# how long resolving the hostname another instance is set up with may take
_RESOLVE_TIMEOUT = 2.0

CONF_DISCOVERED_HOST = "discovered_host"
MANUAL_ENTRY = "manual"

_ENTRIES = (
    ConfigEntry(
        key=CONF_HOST,
        type=ConfigEntryType.STRING,
        required=True,
    ),
    ConfigEntry(
        key=CONF_PORT,
        type=ConfigEntryType.INTEGER,
        required=True,
        default_value=DEFAULT_PORT,
    ),
)


async def run_setup(session: SetupSession) -> None:
    """Run the setup flow: collect the host webservice's host/port and validate it."""
    errors: dict[str, str] | None = None
    setup_data = dict(session.context.setup_data)
    if CONF_HOST not in setup_data and (
        hosts := await _hosts_to_offer(session.mass, session.context.instance_id)
    ):
        host = hosts[0] if len(hosts) == 1 else await _select_host(session, hosts)
        if host is not None:
            setup_data[CONF_HOST] = host.address
            setup_data[CONF_PORT] = host.port
    while True:
        entries = [
            replace(entry, value=setup_data.get(entry.key, entry.value)) for entry in _ENTRIES
        ]
        submitted = await session.form(entries, step_id="user", errors=errors, last_step=True)
        setup_data.update(submitted)
        host_name = cast("str", submitted[CONF_HOST]).strip()
        setup_data[CONF_HOST] = host_name
        port = cast("int", submitted[CONF_PORT])
        client = RaumfeldWebserviceClient(host_name, port, session.mass.http_session)
        if not await client.ping():
            errors = {"base": "cannot_connect"}
            continue
        try:
            await session.finish(setup_data)
            return
        except SetupFlowError as err:
            errors = {"base": err.translation_key or str(err)}


async def _select_host(session: SetupSession, hosts: list[DiscoveredHost]) -> DiscoveredHost | None:
    """
    Let the user pick one of several discovered hosts, or none to enter one manually.

    :param session: The running setup session.
    :param hosts: The discovered hosts to choose from.
    """
    options = [ConfigValueOption(title=_describe(host), value=host.address) for host in hosts]
    options.append(ConfigValueOption(value=MANUAL_ENTRY))
    entry = ConfigEntry(
        key=CONF_DISCOVERED_HOST,
        type=ConfigEntryType.STRING,
        required=True,
        options=options,
        default_value=hosts[0].address,
    )
    submitted = await session.form([entry], step_id="select_host")
    selected = submitted[CONF_DISCOVERED_HOST]
    return next((host for host in hosts if host.address == selected), None)


async def _hosts_to_offer(
    mass: MusicAssistant, own_instance_id: str | None
) -> list[DiscoveredHost]:
    """
    Return the discovered hosts no other instance of this provider is set up for yet.

    :param mass: The MusicAssistant instance.
    :param own_instance_id: The instance being (re)configured, excluded from the check.
    """
    hosts = await discover_hosts(mass.discovery.aiozc.zeroconf, _DISCOVERY_TIMEOUT)
    if not hosts:
        LOGGER.debug("No Raumfeld host discovered")
        return []
    claimed: set[str] = set()
    for instance_id, conf in mass.config.get("providers", {}).items():
        if conf.get("domain") != DOMAIN or instance_id == own_instance_id:
            continue
        if configured := mass.config.get_provider_setup_value(instance_id, CONF_HOST):
            claimed |= await _addresses_of(str(configured))
    offered = [host for host in hosts if not _identities(host) & claimed]
    LOGGER.debug(
        "Discovered Raumfeld hosts %s, offering %s",
        [host.address for host in hosts],
        [host.address for host in offered],
    )
    return offered


async def _addresses_of(configured_host: str) -> set[str]:
    """
    Return every form a configured host can be recognized by: itself and its addresses.

    A hostname (e.g. "one-s.fritz.box") is resolved so it still matches the discovered
    address of the same speaker; one that cannot be resolved only matches itself.

    :param configured_host: The host as entered during setup, a hostname or IP address.
    """
    identities = {configured_host.lower()}
    try:
        ipaddress.ip_address(configured_host)
    except ValueError:
        pass
    else:
        return identities
    try:
        infos = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(configured_host, None, type=socket.SOCK_STREAM),
            _RESOLVE_TIMEOUT,
        )
    except TimeoutError, OSError:
        return identities
    return identities | {str(info[4][0]) for info in infos}


def _identities(host: DiscoveredHost) -> set[str]:
    """Return every form a discovered host can be recognized by."""
    return {host.address.lower()} | ({host.hostname.lower()} if host.hostname else set())


def _describe(host: DiscoveredHost) -> str:
    """Return the label to show a discovered host with."""
    if host.hostname:
        return f"{host.name} - {host.hostname} ({host.address})"
    return f"{host.name} ({host.address})"
