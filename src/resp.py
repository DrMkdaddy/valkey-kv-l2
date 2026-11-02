"""Minimal RESP client for Valkey. No third-party dependency.

Speaks just enough of the protocol to move binary KV blocks: PING, SET (with
EX), GET, DEL, EXISTS, DBSIZE, INFO memory. Values are bytes in, bytes out.
"""
from __future__ import annotations

import socket


class ValkeyError(RuntimeError):
    pass


class Valkey:
    def __init__(self, host: str = "127.0.0.1", port: int = 6390, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._buf = b""

    # ---- connection ---------------------------------------------------------
    def connect(self) -> None:
        if self._sock is None:
            self._sock = socket.create_connection((self.host, self.port), self.timeout)
            self._sock.settimeout(self.timeout)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
            self._buf = b""

    def __enter__(self) -> "Valkey":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- wire format --------------------------------------------------------
    def _encode(self, *parts: bytes) -> bytes:
        out = [b"*%d\r\n" % len(parts)]
        for p in parts:
            out.append(b"$%d\r\n%s\r\n" % (len(p), p))
        return b"".join(out)

    def _read_line(self) -> bytes:
        while b"\r\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ValkeyError("connection closed")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\r\n", 1)
        return line

    def _read_exact(self, n: int) -> bytes:
        while len(self._buf) < n + 2:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ValkeyError("connection closed")
            self._buf += chunk
        data, self._buf = self._buf[:n], self._buf[n + 2:]
        return data

    def _read_reply(self):
        line = self._read_line()
        tag, rest = line[:1], line[1:]
        if tag == b"+":
            return rest.decode()
        if tag == b"-":
            raise ValkeyError(rest.decode())
        if tag == b":":
            return int(rest)
        if tag == b"$":
            n = int(rest)
            return None if n == -1 else self._read_exact(n)
        if tag == b"*":
            n = int(rest)
            return None if n == -1 else [self._read_reply() for _ in range(n)]
        raise ValkeyError(f"unknown reply tag {tag!r}")

    def cmd(self, *parts):
        self.connect()
        self._sock.sendall(self._encode(*(str(p).encode() if not isinstance(p, bytes) else p for p in parts)))
        return self._read_reply()

    # ---- convenience --------------------------------------------------------
    def ping(self) -> bool:
        return self.cmd("PING") == "PONG"

    def set(self, key: str, value: bytes, ex: int | None = None) -> bool:
        args = ["SET", key, value] + (["EX", ex] if ex else [])
        return self.cmd(*args) == "OK"

    def get(self, key: str) -> bytes | None:
        return self.cmd("GET", key)

    def delete(self, key: str) -> int:
        return int(self.cmd("DEL", key))

    def exists(self, key: str) -> bool:
        return int(self.cmd("EXISTS", key)) == 1

    def dbsize(self) -> int:
        return int(self.cmd("DBSIZE"))

    def info(self, section: str = "memory") -> str:
        raw = self.cmd("INFO", section)
        return raw.decode() if isinstance(raw, bytes) else str(raw)
