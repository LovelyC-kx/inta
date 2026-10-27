"""Framing for RHX's semicolon commands and delimiter-free responses.

RHX 3.5.0 TCP documentation; upstream GUI/Widgets/tcpdisplay.cpp
parseCommands() and Engine/Processing/commandparser.cpp getRunModeCommand().
Never interpret a recv() boundary or a 'Return:' prefix as a complete reply.
"""

import re
import socket
import time


class RHXCommandError(RuntimeError):
    """RHX rejected a command, or reported a warning in its command stream."""


RUN_MODE_REPLY = re.compile(
    rb"Return: RunMode (?:Stop|Run|Record|Trigger)(?=$|\s|Return:|Error|Warning)"
)


def validate_command(command):
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a nonempty string")
    if any(c in command for c in (";", "\r", "\n")):
        raise ValueError("Send one command at a time, without semicolons or newlines")
    return command.strip()


def receive_fenced_reply(sock, buffer, *, timeout, run_mode_query=False):
    """Consume target reply plus an ordered `Get RunMode` barrier.

    send_command holds the connection lock from write through this read. For a
    RunMode query, both its own reply and the barrier must arrive. A timeout or
    EOF invalidates the command connection; retry only after reconnecting.
    """
    required = 2 if run_mode_query else 1
    deadline = time.monotonic() + timeout
    while True:
        matches = list(RUN_MODE_REPLY.finditer(buffer))
        if len(matches) >= required:
            fence = matches[required - 1]
            body = bytes(buffer[:fence.start()]).decode("ascii", errors="strict").strip()
            del buffer[:fence.end()]
            if re.search(r"(?:Error|Warning)\s*[:-]", body, re.IGNORECASE):
                raise RHXCommandError(body)
            if not body:
                raise RHXCommandError("No response to the command before RHX barrier")
            return body
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Incomplete RHX command response before deadline")
        sock.settimeout(remaining)
        try:
            data = sock.recv(4096)
        except socket.timeout as exc:
            raise TimeoutError("Incomplete RHX command response before deadline") from exc
        if not data:
            raise ConnectionError("RHX command socket closed before response barrier")
        buffer.extend(data)
        if len(buffer) > 1024 * 1024:
            raise RHXCommandError("RHX response exceeds 1 MiB; refusing an unframed stream")
