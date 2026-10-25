"""Persistent daily budgets with atomic call reservations (ARCHIE-300).

A reservation holds a call's cost ceiling before the request starts. Completion
charges the actual cost and releases unused capacity. Expired reservations have
unknown outcomes: count their full ceiling until a late settlement arrives.
This keeps a worker restart or a provider timeout from reopening spent capacity.
"""

import logging
import math
import re
import uuid
from datetime import UTC, date, datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from zoneinfo import ZoneInfo
from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis


logger = logging.getLogger(__name__)

_MICRODOLLARS = 1_000_000
_MAX_MICRODOLLARS = 2**53 - 1
_RETENTION_SECONDS = 8 * 24 * 3600

# One hash and sorted set share the Redis Cluster hash tag. All amounts are
# integer microdollars; no concurrent caller can reserve the same capacity.
_EXPIRE_RESERVATIONS = """
local function expire_reservations()
    local clock = redis.call('TIME')
    local now = clock[1] * 1000 + math.floor(clock[2] / 1000)
    local expired = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', now)
    for _, id in ipairs(expired) do
        local field = 'r:' .. id
        local record = redis.call('HGET', KEYS[1], field)
        if record and string.sub(record, 1, 2) == 'R:' then
            local amount = tonumber(string.sub(record, 3))
            redis.call('HINCRBY', KEYS[1], 'held', -amount)
            redis.call('HINCRBY', KEYS[1], 'spent', amount)
            redis.call('HSET', KEYS[1], field, 'E:' .. amount)
        end
        redis.call('ZREM', KEYS[2], id)
    end
    return now
end
"""

_RESERVE = (
    _EXPIRE_RESERVATIONS
    + """
local now = expire_reservations()
local spent = tonumber(redis.call('HGET', KEYS[1], 'spent') or '0')
local held = tonumber(redis.call('HGET', KEYS[1], 'held') or '0')
local amount = tonumber(ARGV[2])
if spent + held + amount > tonumber(ARGV[3]) then
    return {0, spent + held, 0}
end
local expires_at = now + tonumber(ARGV[4])
redis.call('HSET', KEYS[1], 'r:' .. ARGV[1], 'R:' .. amount)
redis.call('HINCRBY', KEYS[1], 'held', amount)
redis.call('ZADD', KEYS[2], expires_at, ARGV[1])
redis.call('EXPIRE', KEYS[1], ARGV[5])
redis.call('EXPIRE', KEYS[2], ARGV[5])
return {1, spent + held + amount, expires_at}
"""
)

_SETTLE = (
    _EXPIRE_RESERVATIONS
    + """
expire_reservations()
local field = 'r:' .. ARGV[1]
local record = redis.call('HGET', KEYS[1], field)
if not record then return {-1, 0} end
local state = string.sub(record, 1, 2)
local amount = tonumber(string.sub(record, 3))
local actual = tonumber(ARGV[2])
if state == 'S:' then
    if amount ~= actual then return {-2, amount} end
else
    if state == 'R:' then
        redis.call('HINCRBY', KEYS[1], 'held', -amount)
        redis.call('HINCRBY', KEYS[1], 'spent', actual)
    else
        redis.call('HINCRBY', KEYS[1], 'spent', actual - amount)
    end
    redis.call('HSET', KEYS[1], field, 'S:' .. actual)
    redis.call('ZREM', KEYS[2], ARGV[1])
end
redis.call('EXPIRE', KEYS[1], ARGV[3])
return {1, tonumber(redis.call('HGET', KEYS[1], 'spent') or '0')}
"""
)

_SPENT = (
    _EXPIRE_RESERVATIONS
    + """
expire_reservations()
return tonumber(redis.call('HGET', KEYS[1], 'spent') or '0')
"""
)


def _microdollars(value: float, *, round_up: bool = True) -> int:
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("A cost must be finite and nonnegative")
    units = int(
        (amount * _MICRODOLLARS).to_integral_value(
            rounding=ROUND_CEILING if round_up else ROUND_FLOOR
        )
    )
    if units > _MAX_MICRODOLLARS:
        raise ValueError("The cost exceeds the supported microdollar range")
    return units


class DailyBudgetReservation(BaseModel):
    """One durable reservation; settle it on its original local day."""

    model_config = ConfigDict(frozen=True)

    reservation_id: str = Field(description="Unique call reservation")
    kind: str = Field(description="Agent kind charged for this call")
    day: date = Field(description="Local calendar day at reservation time")
    reserved_microdollars: int = Field(ge=0, description="Reserved call ceiling")
    expires_at: datetime = Field(description="UTC expiry of the pending call")


class DailyBudgetExceeded(RuntimeError):
    """The day's spent and reserved costs leave insufficient capacity."""


class DailyBudget:
    """A per-kind budget shared by processes and retained across restarts.

    Redis failures propagate. Callers must not start their paid request unless
    reserve() succeeds. A call must be bounded by its reserved cost and ttl.
    """

    def __init__(
        self, redis: Redis, kind: str, timezone: str, limit_usd: float
    ) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_.-]*", kind) is None:
            raise ValueError("The agent kind must be a simple lowercase name")
        self.redis = redis
        self.kind = kind
        self.timezone = ZoneInfo(timezone)
        # Capacity rounds down; incurred costs round up. Never create capacity
        # merely because the configured limit has fractional microdollars.
        self.limit_microdollars = _microdollars(limit_usd, round_up=False)

    def _day(self) -> date:
        return datetime.now(self.timezone).date()

    def _keys(self, day: date) -> tuple[str, str]:
        key = f"archie:runtime:budget:{{{self.kind}:{day.isoformat()}}}"
        return key, f"{key}:pending"

    async def reserve(
        self, max_cost: float, ttl: float = 300
    ) -> DailyBudgetReservation:
        amount = _microdollars(max_cost)
        if not math.isfinite(ttl) or ttl <= 0 or ttl >= _RETENTION_SECONDS:
            raise ValueError("Reservation ttl must be positive and less than retention")
        day = self._day()
        reservation_id = uuid.uuid4().hex
        result = await self.redis.eval(
            _RESERVE,
            2,
            *self._keys(day),
            reservation_id,
            amount,
            self.limit_microdollars,
            math.ceil(ttl * 1000),
            _RETENTION_SECONDS,
        )
        accepted, used, expires_at = result
        if not accepted:
            logger.warning(
                f"daily_budget_warning_001: {self.kind} capacity exhausted: "
                f"{used} + {amount} > {self.limit_microdollars} microdollars"
            )
            raise DailyBudgetExceeded(f"Daily {self.kind} budget has no capacity")
        return DailyBudgetReservation(
            reservation_id=reservation_id,
            kind=self.kind,
            day=day,
            reserved_microdollars=amount,
            expires_at=datetime.fromtimestamp(expires_at / 1000, UTC),
        )

    async def settle(self, reservation: DailyBudgetReservation, actual: float) -> None:
        """Charge once, refund unused capacity, and retain every incurred cost."""
        if reservation.kind != self.kind:
            raise ValueError("A reservation belongs to a different agent kind")
        amount = _microdollars(actual)
        status, _ = await self.redis.eval(
            _SETTLE,
            2,
            *self._keys(reservation.day),
            reservation.reservation_id,
            amount,
            _RETENTION_SECONDS,
        )
        if status == -1:
            raise ValueError("The reservation does not exist or passed retention")
        if status == -2:
            raise ValueError("The reservation was already settled with another cost")
        if amount > reservation.reserved_microdollars:
            logger.warning(
                f"daily_budget_warning_002: {self.kind} call cost {amount} exceeds "
                f"reserved {reservation.reserved_microdollars} microdollars"
            )

    async def spent(self) -> float:
        """Actual plus conservative unknown costs today, excluding pending holds."""
        amount = await self.redis.eval(_SPENT, 2, *self._keys(self._day()))
        return amount / _MICRODOLLARS
