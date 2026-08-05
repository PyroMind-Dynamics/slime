"""Sandbox backends for agent rollouts.

The public sandbox contract is intentionally small: async context management,
command execution, and file read/write. Agent examples can build task-specific
setup, runner, and evaluator logic on top of this without depending directly on
one sandbox provider.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import random
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from pyromind_sdk import PyroMindAsyncAPIClient
from pyromind_sdk.client.models import SandboxResponse

logger = logging.getLogger(__name__)


ExecResult = tuple[int, str, str]
FileContent = str | bytes | Path


@runtime_checkable
class Sandbox(Protocol):
    """Minimal async sandbox interface used by agent rollouts.

    ``write_file`` accepts either in-memory content (``str``/``bytes``) or a
    host ``Path`` to stream into the sandbox.

    Retry/idempotency is deliberately *not* part of this contract: whether a
    severed RPC is safe to re-send is a backend transport concern (see
    ``E2BSandbox._rpc_retry``), not something abstraction consumers reason about.
    """

    sandbox_id: str

    async def __aenter__(self) -> Sandbox: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult: ...

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None: ...

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str: ...


EXIT_TIME_BUDGET_EXCEEDED = -1


async def _await_done_marker(sb: Sandbox, done_file: str, *, user: str, time_budget_sec: int) -> int:
    """Poll a detached command's exit-code marker until it appears, returning the
    exit code (or ``EXIT_TIME_BUDGET_EXCEEDED`` if the budget runs out first).

    The 5s ``test -f && cat`` polls are deliberately short, idempotent RPCs --
    they keep the sandbox alive against idle GC while the detached command runs
    over a stream the gateway can't sever.
    """
    deadline = time.time() + time_budget_sec
    while time.time() < deadline:
        await asyncio.sleep(5)
        ec, out, _ = await sb.exec(f"test -f {done_file} && cat {done_file}", user=user, timeout=15, check=False)
        if ec == 0 and (out or "").strip():
            return int(out.strip())
    return EXIT_TIME_BUDGET_EXCEEDED


async def exec_and_wait(
    sb: Sandbox,
    *,
    cmd: str,
    time_budget_sec: int,
    tag: str,
    user: str = "root",
    env: dict[str, str] | None = None,
    workdir: str | None = None,
    out_file: str | None = None,
    want_output: bool = False,
) -> tuple[int, str]:
    """Run ``cmd`` to completion detached, returning ``(exit_code, output)``.

    A plain ``sb.exec`` keeps an HTTP/2 stream open for the command's whole
    runtime, so a long-running command (build, test suite) outlives what the
    E2B gateway will hold a single response stream open for: the stream gets
    severed mid-run and we lose the exit code with no safe way to retry a
    non-idempotent command. Instead we ``setsid`` the command fully detached,
    redirect its output to a file, and have it drop its exit code into a marker
    file. The caller side then becomes a sequence of short, idempotent RPCs --
    write the launcher, fire-and-forget the spawn, then poll for the marker (see
    ``_await_done_marker``) -- none of which depend on a stream staying alive,
    and the polling doubles as an idle-GC keepalive while the command runs.
    """
    out_file = out_file or f"/tmp/.{tag}.out"
    done_file = f"/tmp/.{tag}.done"
    launcher = f"/tmp/.{tag}.sh"
    lock_dir = f"/tmp/.{tag}.spawned"
    prefix = f"cd {workdir}\nexport HOME=/home/{user}\n" if workdir else ""
    launcher_body = f"#!/bin/bash\n{prefix}{cmd}\necho $? > {done_file}\n"
    await sb.write_file(launcher, launcher_body, user=user)

    await sb.exec(
        f"chmod +x {launcher}; "
        f"mkdir {lock_dir} 2>/dev/null || exit 0; "
        f"rm -f {out_file} {done_file}; "
        f"setsid bash {launcher} < /dev/null > {out_file} 2>&1 &",
        user=user,
        env=env,
        timeout=30,
        check=True,
        idempotent=True,
    )
    exit_code = await _await_done_marker(sb, done_file, user=user, time_budget_sec=time_budget_sec)
    if exit_code == 0 and not want_output:
        return exit_code, ""
    if want_output:
        return exit_code, await sb.read_file(out_file, user=user)
    _, tail, _ = await sb.exec(f"tail -c 512 {out_file} 2>/dev/null", user=user, timeout=15, check=False)
    return exit_code, tail or ""


def _getenv(*names: str, default: str = "") -> str:
    """First non-empty environment value among ``names`` (else ``default``).

    Lets a setting carry a primary name plus legacy aliases: list the canonical
    ``SLIME_AGENT_*`` name first, older names after."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return default


class E2BSandbox:
    """Async context manager around e2b.AsyncSandbox."""

    image_metadata_key_env = ("SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY", "SWE_SANDBOX_IMAGE_METADATA_KEY")
    lifetime_sec_env = ("SLIME_AGENT_SANDBOX_LIFETIME_SEC", "SWE_SANDBOX_LIFETIME_SEC")
    rpc_retries_env = ("SLIME_AGENT_SANDBOX_RPC_RETRIES", "SWE_RPC_RETRIES")
    size_env = ("SLIME_AGENT_E2B_SANDBOX_SIZE", "SWE_E2B_SANDBOX_SIZE")

    default_lifetime_sec = 3600
    default_rpc_retries = 6
    default_size = "md"
    rpc_backoff_base_sec = 1.0
    rpc_backoff_cap_sec = 32.0

    def __init__(
        self,
        image: str,
        *,
        timeout: int | None = None,
        image_metadata_key: str | None = None,
        rpc_retries: int | None = None,
        size: str | None = None,
    ) -> None:
        self.image = image
        self.timeout = timeout if timeout is not None else self._lifetime_sec_from_env()
        self.image_metadata_key = image_metadata_key or self._image_metadata_key_from_env()
        self.rpc_retries = rpc_retries if rpc_retries is not None else self._rpc_retries_from_env()
        self.size = size if size is not None else self._size_from_env()
        self._sb = None
        self.sandbox_id = ""

    @classmethod
    def _image_metadata_key_from_env(cls) -> str | None:
        return _getenv(*cls.image_metadata_key_env) or None

    @classmethod
    def _lifetime_sec_from_env(cls) -> int:
        return int(_getenv(*cls.lifetime_sec_env, default=str(cls.default_lifetime_sec)))

    @classmethod
    def _rpc_retries_from_env(cls) -> int:
        return int(_getenv(*cls.rpc_retries_env, default=str(cls.default_rpc_retries)))

    @classmethod
    def _size_from_env(cls) -> str:
        return _getenv(*cls.size_env, default=cls.default_size)

    # Transient client-side failures safe to retry.
    _TRANSIENT_RPC_ERRORS = frozenset(
        {
            "ProtocolError",
            "LocalProtocolError",
            "WriteError",
            "ReadError",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "RemoteProtocolError",
            "SSLError",
        }
    )

    @classmethod
    def _is_transient_rpc_error(cls, e: BaseException) -> bool:
        """True if e is a transient E2B client-side failure safe to retry."""
        name = type(e).__name__
        if name in cls._TRANSIENT_RPC_ERRORS:
            return True
        msg = str(e)
        if name == "SandboxException":
            if "does not exist" in msg or "STOPPED state" in msg:
                return False
            return True
        return False

    async def _rpc_retry(self, op_name: str, coro_factory, *, idempotent: bool = True):
        """Run coro_factory() with retries for transient E2B RPC failures.

        :param idempotent: When False, a transient failure is re-raised instead
            of retried: re-running a non-idempotent op (e.g. a process-spawning
            exec) after a severed response could double-execute it. Idempotent
            ops (the default: create / read_file / write_file / short read-only
            execs) retry as before.
        """
        last_err = None
        for attempt in range(self.rpc_retries):
            try:
                return await coro_factory()
            except Exception as e:
                if not self._is_transient_rpc_error(e):
                    raise
                if not idempotent:
                    raise
                last_err = e
                if attempt + 1 < self.rpc_retries:
                    await self._reset_conn_pool()
                    ceiling = min(self.rpc_backoff_cap_sec, self.rpc_backoff_base_sec * (2**attempt))
                    backoff = random.uniform(0.0, ceiling)
                    logger.debug(
                        "[agent.sandbox] %s transient %s, retry %d/%d in %.1fs: %s",
                        op_name,
                        type(e).__name__,
                        attempt + 1,
                        self.rpc_retries,
                        backoff,
                        str(e)[:120],
                    )
                    await asyncio.sleep(backoff)
        assert last_err is not None
        raise last_err

    async def _reset_conn_pool(self) -> None:
        """Tear down the sandbox's httpcore pool so the next RPC reconnects."""
        try:
            pool = self._sb._transport.pool  # httpcore.AsyncConnectionPool
            await pool.aclose()
        except Exception as e:
            logger.debug("[agent.sandbox] conn-pool reset skipped: %s", e)

    async def __aenter__(self) -> E2BSandbox:
        if self.image_metadata_key is None:
            raise RuntimeError(
                "SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY is not set. Export it "
                "to the metadata key your E2B gateway uses for image routing. "
                "The legacy SWE_SANDBOX_IMAGE_METADATA_KEY name is also "
                "accepted for coding-agent examples."
            )
        from e2b import AsyncSandbox  # type: ignore

        md = {self.image_metadata_key: self.image}

        if self.size:
            prefix = self.image_metadata_key.rsplit("/", 1)[0] if "/" in self.image_metadata_key else ""
            size_key = f"{prefix}/size" if prefix else "size"
            md[size_key] = self.size

        self._sb = await self._rpc_retry("create", lambda: AsyncSandbox.create(timeout=self.timeout, metadata=md))
        self.sandbox_id = self._sb.sandbox_id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._sb is not None:
                await self._sb.kill()
        except Exception as e:
            logger.warning("[agent.sandbox] kill %s failed: %s", self.sandbox_id[:8], e)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        from e2b.sandbox.commands.command_handle import CommandExitException

        try:
            res = await self._rpc_retry(
                f"exec({cmd[:60]!r})",
                lambda: self._sb.commands.run(
                    cmd,
                    user=user,
                    envs=env,
                    timeout=timeout,
                    on_stdout=lambda s: None,
                    on_stderr=lambda s: None,
                ),
                idempotent=idempotent,
            )
            return res.exit_code, res.stdout or "", res.stderr or ""
        except CommandExitException as e:
            if check:
                raise RuntimeError(
                    f"e2b exec failed (exit={e.exit_code}): {cmd[:120]}\n{(e.stderr or '')[:400]}"
                ) from None
            return e.exit_code, e.stdout or "", e.stderr or ""

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, Path):
            host_path = content

            async def _do_path():
                with open(host_path, "rb") as fp:
                    await self._sb.files.write(
                        sandbox_path,
                        fp,
                        user=user,
                        gzip=False,
                        use_octet_stream=True,
                        request_timeout=600,
                    )

            await self._rpc_retry(f"write_file({sandbox_path} <- {host_path.name})", _do_path)
            return

        if isinstance(content, bytes):

            async def _do_bytes():
                await self._sb.files.write(
                    sandbox_path,
                    io.BytesIO(content),
                    user=user,
                    gzip=False,
                    use_octet_stream=True,
                    request_timeout=600,
                )

            await self._rpc_retry(f"write_file({sandbox_path}, bytes={len(content)})", _do_bytes)
            return

        await self._rpc_retry(
            f"write_file({sandbox_path})",
            lambda: self._sb.files.write(sandbox_path, content, user=user),
        )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        try:
            return await self._rpc_retry(
                f"read_file({sandbox_path})",
                lambda: self._sb.files.read(sandbox_path, user=user),
            )
        except Exception:
            return ""


async def ensure_agent_user(sb: Sandbox, workdir: str) -> None:
    """Create the unprivileged 'agent' user that owns workdir + can git diff."""
    await sb.exec(
        f"id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent && "
        f"chown -R agent:agent /home/agent {workdir} && "
        f"git config --system --add safe.directory '*' && id agent",
        user="root",
        check=True,
        timeout=60,
    )


class PyromindSandbox:
    """Async context manager around PyroMind Platform Async Sandbox API.

    Implements the slime Sandbox Protocol using pyromind-sdk.
    """

    default_cpu = "4"
    default_memory = "8Gi"
    default_gpu = "0"
    default_gpu_card = None

    def __init__(
        self,
        image: str,
        *,
        timeout: int | None = None,
        cpu: str | None = None,
        memory: str | None = None,
        gpu: str | None = None,
        gpu_card: str | None = None,
        sandbox_type: str = "custom",
        name: str | None = None,
        volume_mounts: list | None = None,
        port_mappings: list | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.image = image
        self.timeout = timeout
        self.cpu = cpu or os.environ.get("PYROMIND_SANDBOX_CPU", self.default_cpu)
        self.memory = memory or os.environ.get("PYROMIND_SANDBOX_MEMORY", self.default_memory)
        self.gpu = gpu if gpu is not None else os.environ.get("PYROMIND_SANDBOX_GPU", self.default_gpu)
        self.gpu_card = gpu_card or os.environ.get("PYROMIND_SANDBOX_GPU_CARD", self.default_gpu_card)
        self.sandbox_type = sandbox_type
        self.name = name or f"slime-sandbox-{random.randint(0, 0xffffff):06x}"
        self.volume_mounts = volume_mounts
        self.port_mappings = port_mappings
        self.api_key = api_key
        self.base_url = base_url
        self._client: PyroMindAsyncAPIClient | None = None
        self._sandbox: SandboxResponse | None = None
        self.sandbox_id: str = ""

    async def __aenter__(self) -> PyromindSandbox:
        from pyromind_sdk import PyroMindAsyncAPIClient
        from pyromind_sdk.client.models import PortMapping, ResourceConfig, SandboxRequest, SandboxType, VolumeMount

        self._client = PyroMindAsyncAPIClient(api_key=self.api_key, base_url=self.base_url)

        mounts = None
        if self.volume_mounts:
            mounts = [VolumeMount(**m) if isinstance(m, dict) else m for m in self.volume_mounts]

        ports = None
        if self.port_mappings:
            ports = [PortMapping(**p) if isinstance(p, dict) else p for p in self.port_mappings]

        request = SandboxRequest(
            sandbox_type=SandboxType(self.sandbox_type),
            name=self.name,
            image=self.image,
            resources=ResourceConfig(
                cpu=self.cpu,
                memory=self.memory,
                gpu=self.gpu,
                gpu_card=self.gpu_card,
            ),
            volume_mounts=mounts,
            port_mappings=ports,
        )

        self._sandbox = await self._client.sandboxes.create_and_wait(
            request,
            target_status="running",
            timeout=self.timeout or 300,
        )
        self.sandbox_id = self._sandbox.id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._client is not None and self.sandbox_id:
                await self._client.sandboxes.pause(self.sandbox_id)
                await self._client.sandboxes.wait_for_sandbox_status(self.sandbox_id, "paused", timeout=60)
                await self._client.sandboxes.delete(self.sandbox_id)
        except Exception as e:
            logger.warning("[agent.sandbox.pyromind] kill %s failed: %s", self.sandbox_id[:8], e)
        finally:
            if self._client is not None:
                try:
                    await self._client.close()
                except Exception:
                    pass

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        """Execute a shell command in the sandbox.

        Note: user/env parameters are accepted but ignored by Pyromind backend
        since Pyromind sandboxes execute as the container default user.
        """
        from pyromind_sdk import PyroMindAsyncAPIError

        try:
            # Prepend environment variables if provided
            full_cmd = cmd
            if env:
                env_prefix = " ".join(f"{k}={v}" for k, v in env.items()) + " "
                full_cmd = env_prefix + full_cmd

            res = await self._client.sandboxes.exec_command(
                self.sandbox_id,
                command=full_cmd,
                timeout=timeout,
            )
            exit_code = res.returncode
            output = res.output or ""
            stderr = res.exception_info or ""
            if check and exit_code != 0:
                raise RuntimeError(
                    f"Pyromind sandbox exec failed (exit={exit_code}): {cmd[:120]}\n{(stderr or output)[:400]}"
                )
            return exit_code, output, stderr
        except PyroMindAsyncAPIError as e:
            if check:
                raise RuntimeError(f"Pyromind API error: {e.message}") from None
            return -1, "", str(e)

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        """Write a file to the sandbox via the SDK's native chunked upload."""
        from pyromind_sdk import PyroMindAsyncAPIError

        if isinstance(content, Path):
            with open(content, "rb") as fp:
                content_bytes = fp.read()
        elif isinstance(content, bytes):
            content_bytes = content
        else:
            content_bytes = content.encode("utf-8")

        # User parameter is noted but execution permissions handled at container level
        try:
            await self._client.sandboxes.write_file(
                self.sandbox_id,
                sandbox_path,
                content_bytes,
            )
        except PyroMindAsyncAPIError as e:
            raise RuntimeError(f"Failed to write file {sandbox_path}: {e.message}") from None

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        """Read a file from the sandbox via the SDK's native byte stream."""
        from pyromind_sdk import PyroMindAsyncAPIError

        try:
            data = await self._client.sandboxes.read_file(self.sandbox_id, sandbox_path)
        except PyroMindAsyncAPIError as e:
            # Match the legacy "missing file -> empty string" contract.
            if getattr(e, "status_code", None) == 404:
                return ""
            raise RuntimeError(f"Failed to read file {sandbox_path}: {e.message}") from None
        if not data:
            return ""
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")


_SANDBOX_BACKENDS: dict[str, type] = {
    "e2b": E2BSandbox,
    "pyromind": PyromindSandbox,
}


def _parse_json_env(key: str) -> list | None:
    """Parse a JSON list from an env var, returning None if unset or empty."""
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    import json

    return json.loads(raw)


def create_sandbox(image: str) -> PyromindSandbox | E2BSandbox:
    """Create a sandbox instance configured by environment variables.

    ``SANDBOX_BACKEND`` selects the backend (``e2b`` or ``pyromind``,
    default ``e2b``).  For the Pyromind backend the following env vars
    are also honoured:

    ==========================  =====================================
    ``PYROMIND_SANDBOX_CPU``       CPU cores (default ``"4"``)
    ``PYROMIND_SANDBOX_MEMORY``    Memory limit (default ``"8Gi"``)
    ``PYROMIND_SANDBOX_GPU``       GPU count (default ``"0"``)
    ``PYROMIND_SANDBOX_GPU_CARD``  GPU card type
    ``PYROMIND_SANDBOX_VOLUMES``   JSON list of volume mounts
    ``PYROMIND_SANDBOX_PORTS``     JSON list of port mappings
    ==========================  =====================================
    """
    backend = os.environ.get("SANDBOX_BACKEND", "e2b")
    if backend not in _SANDBOX_BACKENDS:
        raise ValueError(f"SANDBOX_BACKEND={backend!r} not supported; " f"choose from {sorted(_SANDBOX_BACKENDS)}")

    if backend == "pyromind":
        return PyromindSandbox(
            image,
            volume_mounts=_parse_json_env("PYROMIND_SANDBOX_VOLUMES"),
            port_mappings=_parse_json_env("PYROMIND_SANDBOX_PORTS"),
        )

    return E2BSandbox(image)
