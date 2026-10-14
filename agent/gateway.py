"""agent/gateway.py — YOUR control plane. CONTRACTS.md section 4, exactly.

READ agent/README.md FIRST — it maps all five files in this directory to what
each is scored on. This file is the one CONTRACTS.md calls "the trusted
envelope's untrusted half": every single MCP / A2A / DISCOVER command your
agent's model wants to make passes through `Gateway.decide` before it is
allowed to happen.

WHY THERE IS NO `execute()` METHOD ON `GatewayContext` (read this before you
go looking for one — there isn't one, and that is not an oversight)
----------------------------------------------------------------------------
CONTRACTS.md section 4's trusted envelope, reproduced here because it is the
one diagram worth memorising:

    [ trusted ]   loop emits a raw action line
         v
    [ trusted ]   INTERCEPT + CANONICALISE -> Command        (kit/loop/agent.py)
         v
    [ UNTRUSTED ] Gateway.decide(cmd) -> Decision             <- THIS FILE
         v
    [ trusted ]   ENFORCE: honour the Decision, meter it,
                  apply the active mutation, execute the
                  ToolCall or refuse it                       (the arena)
         v
    [ trusted ]   RECORD the authoritative L1 event, then
                  RENDER the Observation                      (the arena)
         v
    [ trusted ]   the model sees the Observation

`decide()` returns a *decision*, never a *result*. You cannot reach a tool
server, a file, a socket, or a clock from in here — there is nothing to
call. Two things follow from that, and both matter more than they look:

  1. YOUR TRACE CANNOT BE FORGED. Every `command` / `decision` / `enforced`
     / `tool_call` / `tool_result` L1 event (CONTRACTS.md 5.2) is written by
     the arena, from what the arena itself actually did — never from
     anything you claimed happened. A student gateway that wanted to lie
     about having blocked an attack ("I totally denied that, trust me")
     simply has no channel to lie through: the only thing you ever hand
     back is this one small `Decision` value, and the arena is the one that
     turns it into history.
  2. NOBODY CAN ACCUSE YOU OF A CALL YOU DID NOT AUTHORISE, either. Because
     `decide()` is the ONLY door a command can walk through on its way to
     actually running, a prosecutor's `enforcement_failure` claim against
     you has exactly one thing to point at: the `Decision` you returned for
     that specific `cmd_id`. There is no ambiguity about "maybe the loop
     called the tool directly" — CONTRACTS.md 4.2 removed that path on
     purpose, and kit/loop/agent.py's own module docstring names the same
     invariant from the other side (the loop never imports this module,
     never sees a `Decision`, never executes anything itself).

The cost of that guarantee is that this file is PURE: synchronous, no I/O,
no threads, no `sleep`, 250 ms wall-clock deadline (RULES.md section 3).
Raising anything, returning something that is not a valid `Decision`, or
missing the deadline is treated by the arena as a DENIED command PLUS a 2
credit penalty PLUS an `integrity` event that hands the prosecutor a free
`enforcement_failure` — CONTRACTS.md 4.1's charging table, reproduced in
agent/README.md's own table. Getting this file to just plainly return valid
`Decision` values, every time, is worth more than getting it clever.

THE STARTER'S SHAPE (read this before you start editing `decide()`)
----------------------------------------------------------------------------
This starter FORWARDS ALMOST EVERYTHING AND DENIES NOTHING. That is not a
placeholder oversight — it is the honest zero-defence baseline you are
meant to beat: `bots/rookie` in the kit's own ladder does exactly the same
thing, and RULES.md's own words are "if you cannot beat Rookie you have a
bug, not a strategy." `decide()` below is structured as four named jobs —
ROUTE, ADMIT, AUTHORIZE, BUDGET — each with a one-line TODO naming what a
real implementation checks and why. None of the four currently rejects,
rewrites, or reroutes anything; they are seams, not solutions. Fill them in
using `agent/strategy.py` (routing/budget policy) and `agent/guardrails.py`
(the safety checks) — both already import cleanly from here.

ONE THING WORTH INTERNALISING BEFORE YOU WRITE YOUR FIRST REAL CHECK:
`verdict="deny"` costs the CALLER (your own team) **zero credits** —
CONTRACTS.md 4.1's charging table has exactly one $0 row, and it is this
one. Refusing to make a call you cannot justify is FREE. That makes
abstention a real strategy, not a luxury you can't afford: a `deny` you can
defend beats a `forward` you can't, every time a prosecutor is watching.

Stdlib only. No network, no randomness, no wall-clock reads, no sleeping —
none of that would even survive the kernel sandbox (CONTRACTS.md 12), but
the point is this file has no reason to want any of it in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import zlib
from typing import Any, Mapping, Protocol, runtime_checkable

# kit.mcp.types is a collaborator's file (workspace hard rule 2: import it,
# degrade gracefully). It is present as of this writing and is core, stable
# infrastructure (CONTRACTS.md 3.1) — but this module must still not fail to
# IMPORT if a concurrent edit ever breaks it transiently. When it is
# unavailable, `Decision.call` type-checking is skipped (not enforced), and
# `Gateway.decide` falls back to a minimal local dict-shaped stand-in so the
# rest of this file — everything that does not need a *real* ToolCall — still
# runs.
try:
    from kit.mcp.types import ToolCall
    _TOOLCALL_AVAILABLE = True
except ImportError:  # pragma: no cover - collaborator file
    ToolCall = Any  # type: ignore[assignment, misc]
    _TOOLCALL_AVAILABLE = False

# kit.loop.agent is also a collaborator's file, used only by this module's
# own __main__ demo (to build real Commands the same way the arena's trusted
# canonicaliser would) — never by decide() itself, which never touches the
# loop. Degraded the same way.
try:
    from kit.loop.agent import canonicalise_action as _canonicalise_action
except ImportError:  # pragma: no cover - collaborator file
    _canonicalise_action = None

from agent.telemetry import RecordingGatewayContext, Telemetry


from agent import guardrails, strategy

# ---------------------------------------------------------------------------
# Policy data — what the four jobs in `Gateway.decide` check against.
# ---------------------------------------------------------------------------

# Every tool that mutates state. Each one needs a write scope, an If-Match
# etag, and an idempotency key (CONTRACTS.md 3.2 / hard-mode mechanic 3).
WRITE_TOOLS: frozenset[tuple[str, str]] = frozenset({
    ("progress", "record_mastery"),
    ("content", "flag_stale_slide"),
    ("content", "file_content_bug"),
})
WRITE_SCOPES: Mapping[tuple[str, str], str] = {
    ("progress", "record_mastery"): "wiki.write:progress",
    ("content", "flag_stale_slide"): "wiki.write:content",
    ("content", "file_content_bug"): "wiki.write:content",
}

# Tools that only work with a live lease from a nearby query (mechanic 2).
LEASED_TOOLS: frozenset[tuple[str, str]] = frozenset({("slides", "get_frame")})

# The A2A peers. A call to one of these is a delegation and is held to the
# Agent Card checks (admission, declared skill, audience).
A2A_PEERS: frozenset[str] = frozenset({"curriculum-analyst", "citation-checker", "roster"})

# Argument keys that name WHOSE record a call touches — checked against `ctx.act`.
TARGET_KEYS: tuple[str, ...] = ("learner", "learner_id", "target", "subject", "act", "for_learner")

# Argument keys through which a route can be smuggled in the BODY. Routing is
# decided by the `Mcp-Replica` header only (the `header_spoof` lesson).
BODY_ROUTE_KEYS: tuple[str, ...] = ("route", "_route", "replica", "mcp-replica", "mcp_replica")

# Header values that mean "nobody vouches for this".
UNTRUSTED_MARKERS: frozenset[str] = frozenset({
    "unvouched", "invalid", "unverified", "forged", "tampered", "unknown", "false", "none", "",
})

# The cheap mask a catalog "punishment button" is narrowed to (strategy.py's arithmetic).
CATALOG_SAFE_MASK: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("registry", "list_servers"): ("name",),
    ("glossary", "list_terms"): ("term",),
}

# A single argument this large is a context-economics bomb, not a parameter.
BOMB_ARG_CHARS = 1024


def _header(headers: Mapping[str, Any], name: str) -> Any:
    """Case-insensitive header lookup — HTTP headers are not case-sensitive."""
    lname = name.lower()
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == lname:
            return v
    return None


def _same_learner(target: str, act: str) -> bool:
    """`learner:sv-0417` and `sv-0417` name the same learner; nothing else does."""
    def norm(s: str) -> str:
        s = s.strip().lower()
        return s.split(":", 1)[1] if s.startswith("learner:") else s
    return norm(target) == norm(act)


def _aud_matches(aud: str, server: str) -> bool:
    return aud in (server, f"mcp:{server}", f"a2a:{server}")


def _is_bomb(value: Any) -> bool:
    return isinstance(value, str) and len(value) > BOMB_ARG_CHARS


def _path_id_of(args: Mapping[str, Any]) -> str | None:
    """`Frame:053195a5/w/054` -> `053195a5`; anything else -> the explicit path_id, if any."""
    pid = args.get("path_id")
    if isinstance(pid, str) and pid:
        return pid
    anchor = args.get("anchor")
    if isinstance(anchor, str) and ":" in anchor:
        slug = anchor.split(":", 1)[1].split("/", 1)[0]
        if len(slug) == 8 and all(c in "0123456789abcdef" for c in slug):
            return slug
    return None


def _idempotency_key(cmd: "Command", act: str) -> str:
    """Deterministic for the same logical write, so a retry is recognised as one."""
    parts = [act, cmd.server, cmd.tool] + [f"{k}={cmd.args[k]}" for k in sorted(cmd.args)]
    return "idem:" + format(zlib.crc32("|".join(parts).encode("utf8")), "08x")


def _int_or(value: Any, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default

__all__ = [
    "COMMAND_KINDS",
    "DECISION_VERDICTS",
    "Command",
    "Decision",
    "GatewayContext",
    "Gateway",
]

# CONTRACTS.md 4.1: `Command.kind` — "mcp" | "a2a" | "discover". An "answer"
# action is NEVER a Command (kit/loop/agent.py's own module docstring: "an
# answer is not a tool call routed to a server, so it never becomes a
# Command at all") — it is handled entirely by the loop/arena and never
# reaches `Gateway.decide`.
COMMAND_KINDS: frozenset[str] = frozenset({"mcp", "a2a", "discover"})

# CONTRACTS.md 4.1: `Decision.verdict` — the closed three-member set.
DECISION_VERDICTS: frozenset[str] = frozenset({"forward", "deny", "rewrite"})


@dataclass(frozen=True, slots=True)
class Command:
    """CONTRACTS.md 4.1, field for field — "canonicalised by the arena
    BEFORE the student sees it". You never build one of these from your own
    agent's raw text; the arena's canonicaliser (kit/loop/agent.py's
    `canonicalise_action`, run inside the trusted envelope) already did that
    work and minted `cmd_id` by the time `decide()` sees it. The
    `from_action_dict` classmethod below exists only so this file's own demo
    (and your local tests, if you write any) can build a realistic `Command`
    without duplicating the arena's canonicalisation logic."""

    cmd_id: str
    kind: str  # "mcp" | "a2a" | "discover" — see COMMAND_KINDS
    raw: str
    server: str
    tool: str
    args: dict
    fields: tuple[str, ...]
    headers: dict
    lease_id: str | None
    call_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.cmd_id, str) or not self.cmd_id:
            raise ValueError(f"Command.cmd_id must be a non-empty str, got {self.cmd_id!r}")
        if self.kind not in COMMAND_KINDS:
            raise ValueError(f"Command.kind must be one of {sorted(COMMAND_KINDS)}, got {self.kind!r}")
        if not isinstance(self.server, str) or not self.server:
            raise ValueError(f"Command.server must be a non-empty str, got {self.server!r}")
        if not isinstance(self.tool, str) or not self.tool:
            raise ValueError(f"Command.tool must be a non-empty str, got {self.tool!r}")
        if not isinstance(self.args, dict):
            raise ValueError(f"Command.args must be a dict, got {type(self.args).__name__}")
        if not isinstance(self.headers, dict):
            raise ValueError(f"Command.headers must be a dict, got {type(self.headers).__name__}")
        if (
            not isinstance(self.call_index, int)
            or isinstance(self.call_index, bool)
            or self.call_index < 0
        ):
            raise ValueError(f"Command.call_index must be a non-negative int, got {self.call_index!r}")

    @classmethod
    def from_action_dict(cls, action: Mapping[str, Any], *, cmd_id: str) -> "Command":
        """Build a `Command` from the dict shape `kit.loop.agent.canonicalise_action`
        returns (`kind, raw, server, tool, args, fields, headers, lease_id,
        call_index` — everything except the arena-minted `cmd_id`, supplied
        here as a keyword). Raises `ValueError` if `action["kind"] ==
        "answer"` — an answer is never a Command (see the module docstring).
        This is a convenience for tests/demos, not something the real arena
        calls: the trusted envelope mints `cmd_id` itself and constructs the
        real `Command` on its own side of the boundary."""
        kind = action.get("kind")
        if kind == "answer":
            raise ValueError(
                "an 'answer' action never becomes a Command (kit/loop/agent.py: "
                "\"an answer is not a tool call routed to a server\") — do not "
                "route it through Gateway.decide at all"
            )
        return cls(
            cmd_id=cmd_id,
            kind=kind,
            raw=action["raw"],
            server=action["server"],
            tool=action["tool"],
            args=dict(action.get("args", {})),
            fields=tuple(action.get("fields", ())),
            headers=dict(action.get("headers", {})),
            lease_id=action.get("lease_id"),
            call_index=action.get("call_index", 0),
        )

    def to_dict(self) -> dict:
        return {
            "cmd_id": self.cmd_id,
            "kind": self.kind,
            "raw": self.raw,
            "server": self.server,
            "tool": self.tool,
            "args": dict(self.args),
            "fields": list(self.fields),
            "headers": dict(self.headers),
            "lease_id": self.lease_id,
            "call_index": self.call_index,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """CONTRACTS.md 4.1, field for field.

    Validated strictly (`__post_init__`) because a *structurally* invalid
    `Decision` is charged exactly like a raised exception — CONTRACTS.md
    4.1's charging table: "malformed Decision (schema-invalid) -> 2 cr
    penalty, command denied." Failing loudly HERE, in your own process
    during development, is strictly better than discovering it live in a
    duel as an unexplained penalty.

    `verdict == "deny"` requires a non-empty `reason` (CONTRACTS.md 4.1:
    "required when verdict == 'deny'; shown in the combat log") and
    forbids `call` — a real denial has nothing left to carry out.
    `verdict` in `("forward", "rewrite")` requires `call` to be set — the
    arena executes exactly that `ToolCall`, nothing else, per the trusted
    envelope's whole point (see the module docstring)."""

    verdict: str  # "forward" | "deny" | "rewrite" — see DECISION_VERDICTS
    reason: str | None = None
    call: "ToolCall | None" = None
    quarantine: bool = False
    note: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in DECISION_VERDICTS:
            raise ValueError(
                f"Decision.verdict must be one of {sorted(DECISION_VERDICTS)}, got {self.verdict!r}"
            )
        if self.verdict == "deny":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("Decision.verdict=='deny' requires a non-empty 'reason'")
            if self.call is not None:
                raise ValueError("Decision.verdict=='deny' must not carry a 'call' — there is nothing to run")
        else:  # forward | rewrite
            if self.call is None:
                raise ValueError(f"Decision.verdict=={self.verdict!r} requires 'call' to be set")
            if _TOOLCALL_AVAILABLE and not isinstance(self.call, ToolCall):
                raise ValueError(
                    f"Decision.call must be a kit.mcp.types.ToolCall instance, got {type(self.call).__name__}"
                )
        if not isinstance(self.quarantine, bool):
            raise ValueError(f"Decision.quarantine must be a bool, got {self.quarantine!r}")
        if self.note is not None and not isinstance(self.note, str):
            raise ValueError(f"Decision.note must be a str or None, got {self.note!r}")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "call": self.call.to_dict() if self.call is not None and hasattr(self.call, "to_dict") else self.call,
            "quarantine": self.quarantine,
            "note": self.note,
        }


@runtime_checkable
class GatewayContext(Protocol):
    """CONTRACTS.md 4.2 — "read-only, arena-provided". Note what this is
    NOT: unlike `Command`/`Decision` above, CONTRACTS.md writes this as a
    plain `class`, not a `@dataclass(frozen=True, ...)`. That is a real
    signal, not an inconsistency — a `GatewayContext` is a LIVE VIEW the
    arena owns and updates as the duel actually progresses (credits get
    spent, rounds advance, leases mint and expire), not a frozen snapshot
    you receive once. This file therefore models it as a structural
    `Protocol`: the arena hands your `Gateway.__init__` some object shaped
    like this (see `agent/telemetry.py`'s `RecordingGatewayContext` for a
    concrete example your own tests/demos can build), and you read
    `self.ctx.<field>` fresh every time `decide()` runs — NEVER cache a
    context field into a local variable across calls; if you need memory
    that persists across the duel (a cache, a budget tracker, drift notes),
    keep it on YOUR OWN `Gateway` instance attributes instead (CONTRACTS.md
    4.3: "Gateway is instantiated once per duel... persists across the 10
    rounds").

    There is deliberately no `execute` method here — see the module
    docstring's "WHY THERE IS NO execute() METHOD" section. `emit` is the
    ONE thing you can do that has any side effect at all, and it writes to
    `own_telemetry` only: your own side, never scored, never shown to the
    opponent (`agent/telemetry.py` wraps it)."""

    act: str  # "learner:sv-0417" — WHOM YOU SERVE. Authority derives from this.
    sub: str  # "agent:vlearn-tutor" — what you are. NOT authority.
    scopes: frozenset[str]  # {"wiki.read"}; writes need "wiki.write:progress" etc.
    credits: int  # remaining this duel
    round: int
    call_index: int
    leases: tuple[str, ...]  # live lease ids, arena-tracked
    history: tuple[Mapping[str, Any], ...]  # YOUR OWN prior (Command, Decision, outcome) triples this duel

    def emit(self, name: str, **payload: Any) -> None: ...


class Gateway:
    """The control plane. One instance per duel (CONTRACTS.md 4.3) — built
    once at duel start with a `GatewayContext`, then asked to `decide()` on
    every MCP/A2A/DISCOVER command either side of the duel makes for all 10
    rounds. See the module docstring for the trusted-envelope diagram and
    why there is no `execute()` to call instead.

    Instance attributes below are this starter's ENTIRE per-duel memory —
    all currently unused by `decide()`'s naive body, but declared here
    (rather than invented ad hoc later) so the four TODO jobs below have
    somewhere obvious to keep state once you implement them. `agent/
    strategy.py` has working building blocks for exactly this (a budget
    pacer, a result cache, a replica-choice heuristic) — this starter does
    not wire them in for you; that wiring is the assignment.
    """

    def __init__(self, ctx: GatewayContext) -> None:
        self.ctx = ctx
        self._telemetry = Telemetry(ctx)

        # --- per-duel memory, unused by the naive starter below ---------
        # A cache of anchor -> body-ish data you have already paid for this
        # duel (agent/strategy.py's ResultCache is a ready-made version of
        # this). Populating it needs the *result* of a call, which decide()
        # never sees (it only sees the outgoing Command) — you would fill
        # this from whatever the arena hands back to your agent loop AFTER
        # a call executes, then consult it here on the NEXT decide() call
        # for the same anchor.
        self._seen_anchors: dict[str, Any] = {}
        # Credits you have personally authorised so far this duel — your
        # own running total, independent of (and a cross-check against)
        # `ctx.credits`, which the arena maintains authoritatively.
        self._credits_authorised: int = 0
        # Command ids you have already denied, in case a later job wants to
        # know "have I already said no to this once".
        self._denied_cmd_ids: set[str] = set()
        # Peer Agent Cards the registry vouched for (fed by `note_card`).
        self._cards: dict[str, dict] = {}
        # anchor -> etag from the latest `registry.provenance` (fed by `note_provenance`).
        self._etags: dict[str, str] = {}
        # path_id -> replica we have pinned it to (fed by `note_provenance`).
        self._pinned: dict[str, str] = {}
        # Idempotency keys of writes already let through — exactly-once.
        self._idempotency_keys: set[str] = set()

    def decide(self, cmd: Command) -> Decision:
        """SYNCHRONOUS. PURE. NO I/O. 250 ms wall (RULES.md section 3).
        Raising anything, or returning a `Decision` `__post_init__` rejects,
        is treated by the arena exactly like an explicit deny PLUS a 2
        credit penalty PLUS a scored `integrity` event (CONTRACTS.md 4.1's
        charging table) — so the one thing this method must never do is
        blow up or wander off into I/O, no matter how tempting a "quick
        check" against something external looks. Everything you need to
        decide is already sitting in `cmd` and `self.ctx`.

        Every command runs through the four jobs in order — AUTHORIZE first
        (identity is cheapest to check and the heaviest to get wrong), then
        ADMIT, ROUTE, BUDGET. The first job that refuses wins; a job that
        only needs to narrow the call rewrites it instead. Any exception in
        here fails CLOSED as a plain deny (0 credits), never as a raise
        (2 credits + an `integrity` event)."""
        try:
            self._telemetry.decision_seen(cmd)
        except Exception:  # telemetry must never cost us the decision
            pass
        try:
            decision = self._decide(cmd)
        except Exception as exc:  # fail closed: a deny is free, a raise is not
            decision = Decision(verdict="deny", reason=f"gateway fail-closed: {type(exc).__name__}")
            self._denied_cmd_ids.add(cmd.cmd_id)
        try:
            self._telemetry.decision_made(cmd, decision)
        except Exception:
            pass
        return decision

    # ------------------------------------------------------------------
    # the four jobs
    # ------------------------------------------------------------------
    def _decide(self, cmd: Command) -> Decision:
        key = (cmd.server, cmd.tool)

        # ------------------------------------------------------------------
        # JOB 3 — AUTHORIZE: does this command belong to WHOM YOU SERVE?
        # Checked first: it is the weight-10 class, and none of it costs a
        # thing to evaluate. Authority derives from `ctx.act`, never `ctx.sub`.
        reason = self._authorize(cmd)
        if reason:
            return self._refuse(cmd, reason)

        # ------------------------------------------------------------------
        # JOB 2 — ADMIT: is this call worth letting through AT ALL?
        reason = self._admit(cmd)
        if reason:
            return self._refuse(cmd, reason)

        # ------------------------------------------------------------------
        # JOB 1 — ROUTE: right server/tool, right replica, routed on the HEADER.
        server, tool = strategy.successor_of(*key) or key
        args = {k: v for k, v in cmd.args.items() if not _is_bomb(v)}
        headers = self._route_headers(cmd)

        # Writes: exactly once, with a precondition, or not at all.
        if (server, tool) in WRITE_TOOLS:
            write_headers = self._write_preconditions(cmd, headers)
            if isinstance(write_headers, str):
                return self._refuse(cmd, write_headers)
            headers = write_headers

        # ------------------------------------------------------------------
        # JOB 4 — BUDGET: can the whole duel afford this call as written?
        fields = tuple(cmd.fields)
        if strategy.is_catalog_trap(server, tool, fields):
            fields = CATALOG_SAFE_MASK.get((server, tool), ("name",))
        cost = self._cost(server, tool, fields)
        credits = _int_or(getattr(self.ctx, "credits", None), 100)
        if cost > credits:
            return self._refuse(cmd, f"insufficient credits: {server}.{tool} costs {cost}, {credits} left")

        call = ToolCall(
            server=server, tool=tool, args=args, fields=fields, headers=headers,
            lease_id=cmd.lease_id, call_index=cmd.call_index,
        ) if _TOOLCALL_AVAILABLE else {
            "server": server, "tool": tool, "args": args, "fields": fields,
            "headers": headers, "lease_id": cmd.lease_id, "call_index": cmd.call_index,
        }
        self._credits_authorised += cost
        changed = (
            (server, tool) != key
            or args != cmd.args
            or tuple(sorted(set(fields))) != tuple(sorted(set(cmd.fields)))
            or any(h.lower() in ("if-match", "idempotency-key") and h not in cmd.headers for h in headers)
        )
        note = None
        if (server, tool) != key:
            note = f"{cmd.server}.{cmd.tool} is deprecated; rewritten to {server}.{tool}"
        return Decision(verdict="rewrite" if changed else "forward", call=call, note=note)

    def _authorize(self, cmd: Command) -> str | None:
        act = str(getattr(self.ctx, "act", "") or "")
        # The confused deputy: a target that is not the learner we serve.
        for k in TARGET_KEYS:
            target = cmd.args.get(k)
            if target not in (None, "") and act and not _same_learner(str(target), act):
                return f"target {target!r} is not owned by act {act!r} (authority derives from act, not sub)"
        # A delegation minted for some other audience is not ours to carry.
        aud = _header(cmd.headers, "aud")
        if aud is not None and not _aud_matches(str(aud), cmd.server):
            return f"delegation aud {aud!r} does not match the server called ({cmd.server!r})"
        # A counterfeit server or a tampered Agent Card.
        fp = _header(cmd.headers, "x-server-fingerprint")
        if fp is not None and str(fp).lower() in UNTRUSTED_MARKERS:
            return f"server {cmd.server!r} presented an unvouched fingerprint ({fp!r})"
        sig = _header(cmd.headers, "x-card-signature")
        if sig is not None and str(sig).lower() in UNTRUSTED_MARKERS:
            return f"agent card for {cmd.server!r} failed signature verification"
        if cmd.kind == "a2a" or cmd.server in A2A_PEERS:
            card = self._cards.get(cmd.server)
            if card is not None:
                if not card.get("verified"):
                    return f"peer {cmd.server!r} has no registry-verified agent card"
                skills = set(card.get("skills") or ())
                if skills and cmd.tool not in skills:
                    return f"skill {cmd.tool!r} is not declared on {cmd.server!r}'s agent card"
        # Writes need a write scope this credential was actually granted.
        if (cmd.server, cmd.tool) in WRITE_TOOLS:
            needed = WRITE_SCOPES.get((cmd.server, cmd.tool), "wiki.write")
            scopes = set(getattr(self.ctx, "scopes", ()) or ())
            if needed not in scopes and "wiki.write" not in scopes:
                return f"write {cmd.server}.{cmd.tool} needs scope {needed!r}; granted {sorted(scopes)}"
        return None

    def _admit(self, cmd: Command) -> str | None:
        # A route smuggled through the BODY is the header_spoof / replica_flip tell:
        # routing is decided by the header or refused, never read out of the args.
        for k in BODY_ROUTE_KEYS:
            if cmd.args.get(k) not in (None, ""):
                return f"route declared in the request body ({k}={cmd.args.get(k)!r}); routing is header-only"
        # Instructions riding inside the arguments are injected content, not a request.
        blob = " ".join(str(v) for v in cmd.args.values() if isinstance(v, (str, int, float)))
        # The learner we serve is not "another learner": drop their own id before
        # scanning, so a request about their own record is never read as an attack.
        act_id = str(getattr(self.ctx, "act", "") or "").split(":", 1)[-1]
        if act_id:
            blob = re.sub(rf"(?i)(learner:)?{re.escape(act_id)}\b", " ", blob)
        scan = guardrails.scan_for_injected_instructions(blob)
        if scan.suspicious:
            return f"instruction-shaped content in arguments ({', '.join(scan.matched_patterns[:3])})"
        # A peer answer the arena itself flags as unverified must be cross-checked, not trusted.
        if cmd.args.get("peer_unverified") is True:
            return f"peer {cmd.server!r} answer is unverified; refusing until cross-checked"
        # get_frame without a live lease is doomed (protocol_misuse) — refuse it for free.
        if (cmd.server, cmd.tool) in LEASED_TOOLS:
            if not cmd.lease_id:
                return f"{cmd.server}.{cmd.tool} needs a live lease from a recent query; none presented"
            live = tuple(getattr(self.ctx, "leases", ()) or ())
            if live and cmd.lease_id not in live:
                return f"lease {cmd.lease_id!r} is not live (live: {len(live)})"
        return None

    def _route_headers(self, cmd: Command) -> dict:
        headers = {k: v for k, v in cmd.headers.items()
                   if k.lower() not in ("mcp-replica", "x-mcp-body-route")}
        replica = _header(cmd.headers, "mcp-replica")
        if replica not in ("w", "c"):
            path_id = _path_id_of(cmd.args)
            replica = self._pinned.get(path_id) if path_id else None
            if replica not in ("w", "c"):
                replica = strategy.pick_replica(path_id=path_id, known_drifting=False).replica
        headers["Mcp-Replica"] = replica
        return headers

    def _write_preconditions(self, cmd: Command, headers: dict) -> dict | str:
        anchor = str(cmd.args.get("anchor") or cmd.args.get("target_anchor") or "")
        etag = _header(cmd.headers, "if-match") or self._etags.get(anchor)
        if not etag:
            return "write without an If-Match etag; read registry.provenance first"
        idem = _header(cmd.headers, "idempotency-key") or _idempotency_key(cmd, str(getattr(self.ctx, "act", "")))
        if idem in self._idempotency_keys:
            return "write already committed this duel (exactly-once)"
        self._idempotency_keys.add(idem)
        out = {k: v for k, v in headers.items() if k.lower() not in ("if-match", "idempotency-key")}
        out["If-Match"] = etag
        out["Idempotency-Key"] = idem
        return out

    def _cost(self, server: str, tool: str, fields: tuple[str, ...]) -> int:
        try:
            return int(strategy._spec_cost(server, tool, fields=fields))
        except Exception:
            return 2 + 2 * len(fields)

    def _refuse(self, cmd: Command, reason: str) -> Decision:
        self._denied_cmd_ids.add(cmd.cmd_id)
        return Decision(verdict="deny", reason=reason)

    # ------------------------------------------------------------------
    # fed by the loop/harness after a call returns (optional hooks)
    # ------------------------------------------------------------------
    def note_card(self, server: str, card: Mapping[str, Any]) -> None:
        """Record a peer's Agent Card as the registry vouched for it."""
        self._cards[str(server)] = dict(card or {})

    def note_provenance(self, anchor: str, etag: str, replica: str | None = None) -> None:
        """Record a fresh etag (and, optionally, the replica it was read from)."""
        if anchor and etag:
            self._etags[str(anchor)] = str(etag)
        path_id = _path_id_of({"anchor": anchor})
        if path_id and replica in ("w", "c"):
            self._pinned[path_id] = replica

    def deny(self, cmd: Command, reason: str) -> Decision:
        """Not called anywhere in this starter's `decide()` — a ready-made
        helper for when you fill in JOB 2 / JOB 3 above, so denying doesn't
        mean hand-building a `Decision` inline at every call site. Kept as
        a real method (not a stub) because the shape of a correct denial —
        no `call`, a non-empty `reason` — is exactly the thing worth
        getting right by construction rather than by convention."""
        self._denied_cmd_ids.add(cmd.cmd_id)
        decision = Decision(verdict="deny", reason=reason)
        self._telemetry.decision_made(cmd, decision)
        return decision

    def _to_tool_call(self, cmd: Command) -> "ToolCall":
        """`Command` -> the `ToolCall` (CONTRACTS.md 3.1) the arena will
        actually execute on a `forward`/`rewrite` verdict. When
        `kit.mcp.types` is unavailable (see the module-level import guard),
        falls back to a plain dict carrying the identical fields — `Decision`
        accepts it either way (the `ToolCall` isinstance check inside
        `Decision.__post_init__` only runs when the real class loaded)."""
        fields = {
            "server": cmd.server,
            "tool": cmd.tool,
            "args": dict(cmd.args),
            "fields": cmd.fields,
            "headers": dict(cmd.headers),
            "lease_id": cmd.lease_id,
            "call_index": cmd.call_index,
        }
        if _TOOLCALL_AVAILABLE:
            return ToolCall(**fields)
        return fields  # type: ignore[return-value]


if __name__ == "__main__":
    print("=== agent.gateway: Command / Decision validation ===\n")

    good_cmd = Command(
        cmd_id="cmd:0000",
        kind="mcp",
        raw="MCP slides.get_frame anchor=Frame:3f2a9c11/w/041 fields=title,body lease=lse_7f21",
        server="slides",
        tool="get_frame",
        args={"anchor": "Frame:3f2a9c11/w/041"},
        fields=("body", "title"),
        headers={},
        lease_id="lse_7f21",
        call_index=0,
    )
    print(f"  Command constructed: {good_cmd}")
    assert good_cmd.kind == "mcp"

    print("\n  Rejection demo (each must raise ValueError):")

    def _expect_value_error(label: str, fn) -> None:
        try:
            fn()
        except ValueError as exc:
            print(f"    [{label:38}] -> ValueError: {exc}")
        else:
            raise AssertionError(f"expected ValueError for case {label!r}")

    _expect_value_error("Command.kind == 'answer'", lambda: Command(
        cmd_id="cmd:0001", kind="answer", raw="x", server="slides", tool="get_frame",
        args={}, fields=(), headers={}, lease_id=None, call_index=0,
    ))
    _expect_value_error("Decision verdict='deny' with no reason", lambda: Decision(verdict="deny"))
    _expect_value_error(
        "Decision verdict='forward' with no call", lambda: Decision(verdict="forward")
    )
    _expect_value_error(
        "Decision verdict='deny' carrying a call",
        lambda: Decision(verdict="deny", reason="nope", call={"server": "x", "tool": "y"}),
    )
    _expect_value_error("Decision verdict='?' unknown", lambda: Decision(verdict="???"))

    print("\n=== Command.from_action_dict — real canonicaliser integration ===\n")
    if _canonicalise_action is None:
        print("  kit.loop.agent not importable yet — skipping the live canonicaliser demo")
        demo_commands: list[Command] = [good_cmd]
    else:
        raw_actions = [
            "MCP registry.provenance anchor=Frame:3f2a9c11/w/041 fields=etag",
            'MCP slides.query q="streamable http replaces http+sse" fields=title,body',
            "A2A curriculum-analyst.which_days_cover concept=Concept:streamable-http fields=anchor,course_day,track",
            "DISCOVER registry.list_servers fields=name",
        ]
        demo_commands = []
        for i, raw in enumerate(raw_actions):
            action = _canonicalise_action(raw, call_index=i)
            cmd = Command.from_action_dict(action, cmd_id=f"cmd:{i:04d}")
            print(f"  {raw!r}\n    -> {cmd.kind}: {cmd.server}.{cmd.tool} fields={cmd.fields}")
            demo_commands.append(cmd)
        assert {c.kind for c in demo_commands} == {"mcp", "a2a", "discover"}

        answer_action = _canonicalise_action(
            'ANSWER {"text": "day 26, track P2T2"}', call_index=None
        )
        try:
            Command.from_action_dict(answer_action, cmd_id="cmd:9999")
        except ValueError as exc:
            print(f"\n  an 'answer' action correctly refuses to become a Command: {exc}")
        else:
            raise AssertionError("expected ValueError for an 'answer' action")

    print("\n=== Gateway.decide — the naive starter forwards everything ===\n")
    ctx = RecordingGatewayContext(
        act="learner:sv-0401",
        sub="agent:demo-team",
        scopes=frozenset({"wiki.read"}),
        credits=100,
        round=1,
        call_index=0,
        leases=(),
        history=(),
    )
    assert isinstance(ctx, GatewayContext), "RecordingGatewayContext must structurally satisfy GatewayContext"
    gw = Gateway(ctx)
    for cmd in demo_commands:
        decision = gw.decide(cmd)
        print(f"  decide({cmd.server}.{cmd.tool}) -> verdict={decision.verdict!r} quarantine={decision.quarantine}")
        assert decision.verdict == "forward"
        assert decision.call is not None
        call_dict = decision.call.to_dict() if hasattr(decision.call, "to_dict") else decision.call
        assert call_dict["server"] == cmd.server
        assert call_dict["tool"] == cmd.tool
        assert tuple(call_dict["fields"]) == cmd.fields

    print(f"\n=== Gateway.deny — the unused-by-default free-abstention path ===\n")
    denial = gw.deny(demo_commands[0], reason="demo: withholding pending a fresher registry.provenance read")
    print(f"  gw.deny(...) -> verdict={denial.verdict!r} reason={denial.reason!r} call={denial.call!r}")
    assert denial.verdict == "deny"
    assert denial.call is None
    assert demo_commands[0].cmd_id in gw._denied_cmd_ids

    print(f"\n=== own_telemetry — recorded on YOUR side only, never shown to the opponent ===\n")
    print(f"  {len(ctx.events)} events recorded on this ctx this run:")
    for ev in ctx.events:
        print(f"    {ev['name']}: {sorted(ev['payload'].keys())}")
    assert len(ctx.events) >= len(demo_commands) * 2 + 1  # decision_seen + decision_made per call, plus the deny

    print("\nAll agent/gateway.py demos passed.")
