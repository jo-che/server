"""Tests for finding the Raumfeld host on the network and choosing it during setup."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

from zeroconf import ServiceStateChange

from music_assistant.providers.teufel_raumfeld.const import CONF_HOST, CONF_PORT
from music_assistant.providers.teufel_raumfeld.discovery import (
    MDNS_TYPE,
    DiscoveredHost,
    discover_hosts,
)
from music_assistant.providers.teufel_raumfeld.setup_flow import (
    CONF_DISCOVERED_HOST,
    MANUAL_ENTRY,
    _addresses_of,
    _hosts_to_offer,
    run_setup,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry

HOST_SERVICE = "RaumfeldControl on Teufel One S._http._tcp.local."
SECOND_HOST_SERVICE = "RaumfeldControl on Raumfeld Connector._http._tcp.local."
PRINTER_SERVICE = "Brother MFC-J4410DW._http._tcp.local."
DISCOVERY = "music_assistant.providers.teufel_raumfeld.discovery"
SETUP_FLOW = "music_assistant.providers.teufel_raumfeld.setup_flow"

# what each announced service resolves to: address, mDNS hostname and port
RECORDS = {
    HOST_SERVICE: ("10.0.0.125", "one-s.local.", 47365),
    SECOND_HOST_SERVICE: ("10.0.0.27", "connector.local.", 47365),
    PRINTER_SERVICE: ("10.0.0.3", "brother.local.", 80),
}

ONE_S = DiscoveredHost("Teufel One S", "10.0.0.125", "one-s.local", 47365)
CONNECTOR = DiscoveredHost("Raumfeld Connector", "10.0.0.27", "connector.local", 47365)


def _browser(announced: list[str]) -> Callable[..., MagicMock]:
    """
    Return a stand-in for the mDNS browser that announces the given services at once.

    :param announced: Names of the services on the network.
    """

    def create(zeroconf: Any, types: list[str], handlers: list[Callable[..., None]]) -> MagicMock:
        for name in announced:
            for handler in handlers:
                handler(
                    zeroconf=zeroconf,
                    service_type=types[0],
                    name=name,
                    state_change=ServiceStateChange.Added,
                )
        browser = MagicMock()
        browser.async_cancel = AsyncMock()
        return browser

    return create


def _service_info(service_type: str, name: str) -> MagicMock:
    """
    Return a stand-in for a resolved mDNS record of the given service.

    :param service_type: The mDNS service type.
    :param name: The service's full mDNS name.
    """
    del service_type
    info = MagicMock()
    info.async_request = AsyncMock(return_value=name in RECORDS)
    info.address, info.server, info.port = RECORDS.get(name, (None, None, None))
    return info


async def _discover(announced: list[str]) -> list[DiscoveredHost]:
    """
    Run host discovery on a network announcing the given services.

    :param announced: Names of the services on the network.
    """
    with (
        patch(f"{DISCOVERY}.AsyncServiceBrowser", side_effect=_browser(announced)),
        patch(f"{DISCOVERY}.AsyncServiceInfo", side_effect=_service_info),
        patch(f"{DISCOVERY}._MORE_HOSTS_WAIT", 0),
        patch(
            f"{DISCOVERY}.get_primary_ip_address_from_zeroconf",
            side_effect=lambda info: info.address,
        ),
    ):
        return await discover_hosts(MagicMock(), timeout=0.1)


async def test_discover_hosts_finds_the_raumfeld_host() -> None:
    """Only the host's RaumfeldControl service counts, not other HTTP services."""
    assert await _discover([PRINTER_SERVICE, HOST_SERVICE]) == [ONE_S]


async def test_discover_hosts_finds_every_raumfeld_host() -> None:
    """Several Raumfeld systems on one network are all found."""
    hosts = await _discover([HOST_SERVICE, SECOND_HOST_SERVICE])

    assert set(hosts) == {ONE_S, CONNECTOR}


async def test_discover_hosts_without_a_host_returns_nothing() -> None:
    """A network without a Raumfeld host yields no hosts after the timeout."""
    assert await _discover([PRINTER_SERVICE]) == []


async def test_discover_hosts_browses_http_services() -> None:
    """The host is looked for among plain HTTP services, the only type it announces."""
    browser = MagicMock(side_effect=_browser([]))

    with patch(f"{DISCOVERY}.AsyncServiceBrowser", browser):
        await discover_hosts(MagicMock(), timeout=0.1)

    assert browser.call_args.args[1] == [MDNS_TYPE]


def _mass(configured_hosts: dict[str, str]) -> MagicMock:
    """
    Return a MusicAssistant stand-in with the given provider instances set up.

    :param configured_hosts: Host of each existing instance, keyed by instance id.
    """
    mass = MagicMock()
    mass.config.get.return_value = {
        instance_id: {"domain": "teufel_raumfeld"} for instance_id in configured_hosts
    } | {"other": {"domain": "dlna"}}
    mass.config.get_provider_setup_value.side_effect = lambda instance_id, _key: (
        configured_hosts.get(instance_id)
    )
    return mass


async def _offered(
    configured_hosts: dict[str, str],
    own_instance_id: str | None = None,
    resolved: set[str] | None = None,
) -> list[DiscoveredHost]:
    """
    Return the hosts offered on a network with both hosts, given the existing instances.

    :param configured_hosts: Host of each existing instance, keyed by instance id.
    :param own_instance_id: The instance being reconfigured, if any.
    :param resolved: Addresses a configured hostname resolves to.
    """

    async def addresses_of(configured: str) -> set[str]:
        return {configured.lower()} | (resolved or set())

    with (
        patch(f"{SETUP_FLOW}.discover_hosts", AsyncMock(return_value=[ONE_S, CONNECTOR])),
        patch(f"{SETUP_FLOW}._addresses_of", side_effect=addresses_of),
    ):
        return await _hosts_to_offer(_mass(configured_hosts), own_instance_id)


async def test_hosts_to_offer_offers_every_new_host() -> None:
    """Without existing instances, every discovered host is offered."""
    assert await _offered({}) == [ONE_S, CONNECTOR]


async def test_hosts_to_offer_skips_a_host_set_up_by_address() -> None:
    """A host another instance is set up for by its IP address is not offered again."""
    assert await _offered({"existing": "10.0.0.125"}) == [CONNECTOR]


async def test_hosts_to_offer_skips_a_host_set_up_by_mdns_hostname() -> None:
    """A host another instance is set up for by its mDNS hostname is not offered again."""
    assert await _offered({"existing": "One-S.local"}) == [CONNECTOR]


async def test_hosts_to_offer_skips_a_host_set_up_by_other_hostname() -> None:
    """A hostname that resolves to a discovered host's address also marks it as set up."""
    offered = await _offered({"existing": "one-s.fritz.box"}, resolved={"10.0.0.125"})

    assert offered == [CONNECTOR]


async def test_hosts_to_offer_keeps_the_host_of_the_instance_being_reconfigured() -> None:
    """Reconfiguring an instance still offers the host it is set up for itself."""
    assert await _offered({"existing": "10.0.0.125"}, own_instance_id="existing") == [
        ONE_S,
        CONNECTOR,
    ]


async def test_addresses_of_an_ip_address_is_itself() -> None:
    """An IP address is not resolved any further."""
    assert await _addresses_of("10.0.0.125") == {"10.0.0.125"}


async def test_addresses_of_a_hostname_includes_what_it_resolves_to() -> None:
    """A resolvable hostname is recognized by its name and its addresses."""
    assert await _addresses_of("localhost") >= {"localhost", "127.0.0.1"}


async def test_addresses_of_an_unresolvable_hostname_is_itself() -> None:
    """A hostname that cannot be resolved still matches itself."""
    assert await _addresses_of("no-such-host.invalid") == {"no-such-host.invalid"}


def _session(*submissions: dict[str, Any]) -> MagicMock:
    """
    Return a setup session stand-in that submits the given form values in order.

    :param submissions: The values submitted for each form shown, in order.
    """
    session = MagicMock()
    session.context.setup_data = {}
    session.context.instance_id = None
    session.form = AsyncMock(side_effect=list(submissions))
    session.finish = AsyncMock()
    return session


def _shown(session: MagicMock, step_id: str) -> list[ConfigEntry]:
    """
    Return the entries of the (first) form shown for the given step.

    :param session: The setup session stand-in.
    :param step_id: The step to return the form entries of.
    """
    for call in session.form.call_args_list:
        if call.kwargs.get("step_id") == step_id:
            return list(call.args[0])
    raise AssertionError(f"no {step_id} form was shown")


async def _run_setup(session: MagicMock, offered: list[DiscoveredHost]) -> MagicMock:
    """
    Run the setup flow with the given hosts offered and a reachable host webservice.

    :param session: The setup session stand-in.
    :param offered: The discovered hosts to offer.
    """
    client = MagicMock()
    client.ping = AsyncMock(return_value=True)
    with (
        patch(f"{SETUP_FLOW}._hosts_to_offer", AsyncMock(return_value=offered)),
        patch(f"{SETUP_FLOW}.RaumfeldWebserviceClient", return_value=client) as client_class,
    ):
        await run_setup(session)
    return client_class


async def test_setup_prefills_a_single_host_without_asking() -> None:
    """With one host found there is nothing to choose, so the form is prefilled directly."""
    session = _session({CONF_HOST: "10.0.0.125", CONF_PORT: 47365})

    await _run_setup(session, [ONE_S])

    assert session.form.await_count == 1
    host_entry = _shown(session, "user")[0]
    assert host_entry.value == "10.0.0.125"


async def test_setup_lets_the_user_choose_between_hosts() -> None:
    """With several hosts found, the chosen one prefills the form."""
    session = _session(
        {CONF_DISCOVERED_HOST: "10.0.0.27"}, {CONF_HOST: "10.0.0.27", CONF_PORT: 47365}
    )

    await _run_setup(session, [ONE_S, CONNECTOR])

    choice = _shown(session, "select_host")[0]
    assert [option.value for option in choice.options] == ["10.0.0.125", "10.0.0.27", MANUAL_ENTRY]
    assert _shown(session, "user")[0].value == "10.0.0.27"


async def test_setup_manual_entry_leaves_the_form_empty() -> None:
    """Choosing to enter a host manually shows the form without a discovered host."""
    session = _session({CONF_DISCOVERED_HOST: MANUAL_ENTRY}, {CONF_HOST: "x", CONF_PORT: 1})

    await _run_setup(session, [ONE_S, CONNECTOR])

    assert _shown(session, "user")[0].value is None


async def test_setup_accepts_a_hostname() -> None:
    """A hostname is used as entered, without surrounding whitespace."""
    session = _session({CONF_HOST: " one-s.fritz.box ", CONF_PORT: 47365})

    client_class = await _run_setup(session, [])

    client_class.assert_called_once_with("one-s.fritz.box", 47365, session.mass.http_session)
    session.finish.assert_awaited_once()
    assert session.finish.call_args.args[0][CONF_HOST] == "one-s.fritz.box"
