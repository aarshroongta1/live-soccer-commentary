"""Day 1 gate: prove that whatever is on screen reaches Python.

uv run python -m commentary            # 10 seconds of capture stats
"""

from __future__ import annotations

import asyncio
import sys

from commentary.capture import DelayBuffer, ScreenCapture
from commentary.config import CAPTURE


async def main(seconds: float = 10.0) -> int:
    buf = DelayBuffer(CAPTURE.fps, CAPTURE.delay_s, CAPTURE.history_s)
    count = 0
    async with ScreenCapture() as cap:
        async for frame in cap.frames():
            buf.append(frame)
            count += 1
            if count % CAPTURE.fps == 0:
                cursor = buf.cursor_ts
                print(
                    f"{frame.ts:6.1f}s  frames={count:5d}  "
                    f"fps={count / max(frame.ts, 1e-9):5.1f}  "
                    f"buffered={len(buf):4d}  "
                    f"cursor={'-' if cursor is None else f'{cursor:6.1f}s'}  "
                    f"ready={buf.ready}",
                    flush=True,
                )
            if frame.ts >= seconds:
                break
    if count == 0:
        print("no frames: check AVFOUNDATION_DEVICE, see scripts/list_devices.sh")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 10.0)))
