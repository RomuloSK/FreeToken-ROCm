from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    return f".pid={os.getpid()}"


def _zmq_addr(slot: int, suffix: str) -> str:
    """Return an IPC endpoint on Unix and a stable loopback TCP endpoint on Windows."""

    if os.name != "nt":
        return f"ipc:///tmp/freetoken_{slot}{suffix}"
    digest = hashlib.blake2s(suffix.encode("utf-8"), digest_size=4).digest()
    # Keep five adjacent ports per server and stay inside the user/dynamic
    # range.  The PID suffix makes concurrent server processes independent.
    # Reserve five adjacent ports while keeping the highest endpoint below the
    # 16-bit socket limit (65535).  The previous 8,000-slot range could produce
    # ports in the 70,000s for otherwise valid scheduler instances.
    base = 30000 + (int.from_bytes(digest, "little") % 5000) * 5
    return f"tcp://127.0.0.1:{base + slot}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return _zmq_addr(0, self._unique_suffix)

    @property
    def zmq_detokenizer_addr(self) -> str:
        return _zmq_addr(1, self._unique_suffix)

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return _zmq_addr(2, self._unique_suffix)

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
