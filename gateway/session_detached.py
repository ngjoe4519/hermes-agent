"""A messaging conversation its chat moved past stays usable from the apps (Teknium's ruling).

When a messaging chat names another conversation (``/new``, ``/resume``, ``/branch``, a handoff,
its next inbound naming another one), a Desktop / Webapp / mobile view still open on the previous
conversation keeps working: its turns, Stop and controls act on that conversation alone, and its
replies reach its viewers only, never the chat, which now gets only its new conversation's replies.

``detach`` moves the live session off the chat's route onto a key that names the conversation
(``agent:<ns>:detached:<session id>``) and pins that key on a copy of its source
(``SessionSource.detached_route``). From there the turn runs on the ordinary messaging path:

* ``build_session_key`` answers the pinned key, so the runner's per-route state (running slot,
  agent cache, busy guard, the agent a Stop reaches) is the conversation's own, never the chat's;
* ``SessionStore.get_or_create_session`` resolves the key to the conversation itself
  (``detached_entry``), and never creates or persists a route for it;
* delivery goes through the profile's ``LocalSessionAdapter`` (gateway/session_local.py), as a local
  session's does: its sends publish nothing, viewers read the canonical event stream.

The key carries no platform slot, so a reader that parses a chat out of a session key
(``gateway.run._parse_session_key``) finds none and delivers nowhere. The chat naming the
conversation again (``/resume``) moves it back onto the chat's route (``SessionAuthority.name_lane``).
"""
from __future__ import annotations

from datetime import datetime, timedelta

_SLOT = 'detached'


def detached_route_of(source) -> str | None:
    """*source*'s pinned detached key; None for any other source (duck-typed test sources too)."""
    route = getattr(source, 'detached_route', None)
    return route if isinstance(route, str) else None


def detached_key(route: str, session_id: str) -> str:
    """The detached key of *session_id*, in the profile namespace of its chat's *route*."""
    parts = route.split(':')
    namespace = ':'.join(parts[:2]) if len(parts) > 1 and parts[0] == 'agent' else 'agent:main'
    return f'{namespace}:{_SLOT}:{session_id}'


def detached_session_id(key: str) -> str | None:
    parts = key.split(':')
    return parts[3] if len(parts) == 4 and parts[0] == 'agent' and parts[2] == _SLOT else None


def detach(authority, live, session_id) -> None:
    """Move *live* (a messaging conversation its chat no longer names) off the chat's route. A
    history mount (no source) has no route to leave, and a detached one has left it already."""
    if live.source is None or detached_route_of(live.source):
        return
    from gateway.session_identity import replace_source
    from gateway.session_local_recovery import local_adapter
    local_adapter(authority)  # its delivery; a messaging-only gateway has none yet
    live.route = detached_key(live.route, session_id)
    live.source = replace_source(live.source, detached_route=live.route)


def recover_detached_sessions(authority, pending):
    """Boot: schedule the queued input of conversations their chats moved past (an app prompt
    accepted during a restart's after-turn wait). The routing index names only each chat's current
    conversation, so native recovery never binds these, and no chat message will wake them."""
    from gateway.session_contract import SessionRef
    from gateway.session_route_recovery import restore_routed_session
    scheduled = []
    from hermes_state_runtime import list_session_admissions
    for session_id in sorted(pending):
        if session_id in authority.sessions or not any(  # a lost (unknown) turn waits for /discard
                row['status'] == 'queued' for row in list_session_admissions(authority.db, session_id=session_id)):
            continue
        if restore_routed_session(authority, session_id) is None:
            continue
        if authority.sessions[session_id].detached:
            authority._schedule(SessionRef(authority.profile_id, session_id))
            scheduled.append(session_id)
    return scheduled


def detached_entry(store, source):
    """The routing entry a detached *source* resolves to: its own conversation, at the compression
    tip a turn writes to. Never stored: the chat's route stays the only one this conversation has."""
    from gateway.session import SessionEntry
    key = source.detached_route
    session_id = str(detached_session_id(key))
    db = store._db_for_key(key)
    row = db.get_session(session_id) if db is not None else None
    tip = (db.get_compression_tip(session_id) if db is not None else None) or session_id
    now = datetime.now()
    # Never a new conversation (that would fire session:start and first-contact notes).
    started = datetime.fromtimestamp(row['started_at']) if row and row.get('started_at') else now - timedelta(seconds=1)
    return SessionEntry(session_key=key, session_id=tip, created_at=started, updated_at=now, origin=source,
                        platform=source.platform, chat_type=source.chat_type)
