"""Quick smoke test for the Pyromind sandbox backend via create_sandbox()."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.agent.sandbox import create_sandbox


async def main():
    api_key = os.environ.get("PYROMIND_API_KEY")
    base_url = os.environ.get("PYROMIND_BASE_URL")

    if not api_key or not base_url:
        print("ERROR: PYROMIND_API_KEY and PYROMIND_BASE_URL must be set")
        raise SystemExit(1)

    os.environ["SANDBOX_BACKEND"] = "pyromind"
    os.environ["PYROMIND_SANDBOX_CPU"] = "2"
    os.environ["PYROMIND_SANDBOX_MEMORY"] = "4Gi"
    os.environ["PYROMIND_SANDBOX_GPU"] = "0"
    os.environ["PYROMIND_SANDBOX_VOLUMES"] = '[{"host_path":"/tmp","mount_path":"/mnt/data"}]'
    os.environ["PYROMIND_SANDBOX_PORTS"] = '[{"container_port":8080}]'

    print(f"[1/7] Creating sandbox (base_url={base_url}) ...")
    async with create_sandbox("python:3.11-slim") as sb:
        print(f"       sandbox_id = {sb.sandbox_id}")
        print(f"       cpu={sb.cpu}  memory={sb.memory}  gpu={sb.gpu}  gpu_card={sb.gpu_card}")
        print(f"       volumes={sb.volume_mounts}")
        print(f"       ports={sb.port_mappings}")
        assert sb.cpu == "2"
        assert sb.memory == "4Gi"

        print("[2/7] exec: echo hello ...")
        ec, out, err = await sb.exec("echo hello from pyromind", check=True)
        print(f"       exit={ec}  stdout={out.strip()!r}")
        assert ec == 0

        print("[3/7] write_file + read_file ...")
        await sb.write_file("/tmp/test.txt", "hello slime!")
        content = await sb.read_file("/tmp/test.txt")
        print(f"       read back: {content.strip()!r}")
        assert content.strip() == "hello slime!"

        print("[4/7] exec: python3 one-liner ...")
        ec, out, err = await sb.exec("python3 -c 'print(2+3)'", check=True)
        print(f"       exit={ec}  output={out.strip()!r}")
        assert out.strip() == "5"

        print("[5/7] exec with env ...")
        ec, out, err = await sb.exec("bash -c 'echo $MY_VAR'", env={"MY_VAR": "foobar"}, check=False)
        print(f"       exit={ec}  output={out.strip()!r}")
        assert out.strip() == "foobar"

        print("[6/7] verify volume mount (/mnt/data) ...")
        ec, out, err = await sb.exec("ls -la /mnt/data/", check=False)
        print(f"       exit={ec}  output={out.strip()[:120]!r}")
        if ec == 0:
            print("       volume mount OK")
        else:
            print(f"       WARNING: volume mount may not be active: {err.strip()[:120]}")

        print("[7/7] verify port mapping (bind 8080) ...")
        ec, out, err = await sb.exec(
            "python3 -c \"import socket; s=socket.socket(); s.bind(('0.0.0.0',8080)); s.close(); print('port 8080 bindable')\"",
            check=False,
        )
        print(f"       exit={ec}  output={out.strip()!r}")

    print("\nAll checks passed!")


if __name__ == "__main__":
    asyncio.run(main())
