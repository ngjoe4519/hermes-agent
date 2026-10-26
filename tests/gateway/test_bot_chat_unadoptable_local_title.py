"""A local row that holds the canonical "Bot Chat" title but no native open can ever adopt does not
refuse every Bot Chat delivery for good (#141243).

A crash or reconnect can leave a CLI/TUI/Desktop row titled "Bot Chat" with a route column but no
local binding (or another profile's stamp). ``adopt_legacy_session`` refuses it for every caller, so
a cron ``deliver: bot-chat`` failed ``not_found`` on every beat. The door now moves it aside, as it
does an API or messaging row, and mints the profile's local Bot Chat; adoptable history stays put.
"""
from pathlib import Path

import pytest

from hermes_state_runtime import RuntimeStoreError

from tests.gateway import test_peer_dm_bot_chat_headless as _headless_gateway

# The headless gateway fixture: one real authority over a SessionDB in a temp HERMES_HOME.
headless = _headless_gateway.headless
_local_owner = _headless_gateway._local_owner


def _squatter(db, sid, *, column, value, source='tui'):
    db.create_session(sid, source)
    db.set_session_title(sid, 'Bot Chat')
    if column:
        db._execute_write(lambda conn: conn.execute(f'UPDATE sessions SET {column}=? WHERE id=?', (value, sid)))
    return sid


async def _deliver(authority, key):
    from gateway.session_authorities import served_profile_name
    from gateway.session_bot import deliver
    from tools.bot_live_delivery import _bot_chat_tip
    home = Path(authority.db.db_path).parent.resolve()
    # The cron producer's door: ``bot_relay.deliver`` on a local interactive connection, no author,
    # pinned to the Bot Chat it discovered first (``find_canonical_live_owner``).
    pinned = _bot_chat_tip(home)
    return await deliver(_local_owner(authority), dict(id=key * 32, profile=served_profile_name(home),
                                                       message='cron beat', **({'session_id': pinned} if pinned else {})))


@pytest.mark.parametrize('column, value', [
    ('session_key', 'agent:main:tui:dm:stale'),   # route left by a crash, never bound
    ('profile_name', 'another-profile'),          # stamped by another profile
])
@pytest.mark.asyncio
async def test_an_unadoptable_local_bot_chat_is_moved_aside_and_the_delivery_lands(headless, column, value):
    from tests.waiting import async_wait_until
    db = headless.authority.db
    old = _squatter(db, '20261018_071332_c8e3e4', column=column, value=value)
    first = await _deliver(headless.authority, 'a')  # pinned to the squatter: the beat still lands
    new = first['session_id']
    assert new != old and str(db.get_session(new)['chat_id']).startswith('local-'), first
    assert db.get_session_by_title('Bot Chat')['id'] == new
    assert db.get_session(old)['title'] == f'Bot Chat (tui {old})', 'the old row keeps its id under a visible name'
    # Every later beat finds the minted chat by title: no further move, no second Bot Chat.
    second = await _deliver(headless.authority, 'b')
    assert second['session_id'] == new
    await async_wait_until(lambda: len(headless.executed) == 2, timeout=10, interval=.05)
    assert [sid for sid, _, _ in headless.executed] == [new, new]
    assert len(db._read_all('SELECT id FROM sessions')) == 2


@pytest.mark.parametrize('stamp', ['github:alice', 'nous:alice@example.com'])
@pytest.mark.asyncio
async def test_an_unbound_stamped_bot_chat_is_moved_aside_and_the_delivery_lands(headless, stamp):
    # A ``user_id`` on a row with no binding at all names nobody the delivery can reach: the door's
    # principal is submit-only (no native transport) and refuses a foreign stamp, so the row held
    # the title and refused every delivery for good (#143715). Moving it loses nothing.
    from tests.waiting import async_wait_until
    db = headless.authority.db
    old = _squatter(db, '20260924_173840_909979', column='user_id', value=stamp)
    db.append_message(old, 'user', 'earlier')
    first = await _deliver(headless.authority, 'd')
    new = first['session_id']
    assert new != old and str(db.get_session(new)['chat_id']).startswith('local-'), first
    assert db.get_session_by_title('Bot Chat')['id'] == new
    assert db.get_session(old)['title'] == f'Bot Chat (tui {old})' and db.get_session(old)['user_id'] == stamp
    second = await _deliver(headless.authority, 'e')
    assert second['session_id'] == new
    await async_wait_until(lambda: len(headless.executed) == 2, timeout=10, interval=.05)
    assert len(db._read_all('SELECT id FROM sessions')) == 2


@pytest.mark.parametrize('source, bound', [
    ('tui', True),        # a binding names a route the row answers on
    ('desktop', False),   # a served login's own Desktop row (#140183)
])
@pytest.mark.asyncio
async def test_a_stamped_bot_chat_that_names_its_owner_is_never_moved_aside(headless, source, bound):
    db = headless.authority.db
    old = _squatter(db, 'tui_bot_chat', column='user_id', value='github:alice', source=source)
    if bound:
        db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET session_key='agent:main:tui:dm:bound' WHERE id=?", (old,)))
    db.append_message(old, 'user', 'earlier')
    with pytest.raises(RuntimeStoreError):
        await _deliver(headless.authority, 'f')
    assert db.get_session_by_title('Bot Chat')['id'] == old, 'an owned row keeps its title'
    assert len(db._read_all('SELECT id FROM sessions')) == 1


@pytest.mark.asyncio
async def test_kept_local_bot_chat_history_is_never_moved_aside(headless):
    # A plain pre-upgrade TUI Bot Chat is adopted in place, never renamed.
    db = headless.authority.db
    old = _squatter(db, 'tui_bot_chat', column=None, value=None)
    db.append_message(old, 'user', 'earlier')
    record = await _deliver(headless.authority, 'c')
    assert record['session_id'] == old
    assert db.get_session_by_title('Bot Chat')['id'] == old and len(db._read_all('SELECT id FROM sessions')) == 1
