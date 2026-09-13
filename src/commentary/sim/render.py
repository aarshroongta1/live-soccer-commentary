"""Turning a :class:`SimState` into something that looks like television.

The renderer only has to be convincing to a vision model reading a wide shot,
so it aims at the handful of things the system actually keys on: which way
the two kits are pointing, where the ball is, whether the score bug is up,
and whether this is a replay. Everything else is set dressing.

The one thing here that is not set dressing is the timestamp strip. Every
frame carries its own video time burned into the top-left corner as black and
white blocks, which is what lets the offline oracle answer honestly: it gets
content blocks like any other backend, and the only way it learns which
moment it is looking at is by reading the picture. The strip is sized as a
fraction of the frame so that it fits inside the score-bug crop at any capture
resolution, which matters because the board reader sends that crop and nothing
else. It survives the JPEG round-trip too.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from commentary.config import SETTINGS
from commentary.schemas import Event, KnowledgePack, Player, Scene, Side
from commentary.sim.match import PITCH_L, PITCH_W, Dot, SimState

#: The pitch drawn flat into a wide virtual plane, before the camera crops it.
_FAR_Y = 120.0
_NEAR_Y = 940.0
_VIEW_CORNERS: tuple[tuple[float, float], ...] = (
    (360.0, _FAR_Y),
    (2040.0, _FAR_Y),
    (2400.0, _NEAR_Y),
    (0.0, _NEAR_Y),
)
_WORLD_CORNERS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (PITCH_L, 0.0),
    (PITCH_L, PITCH_W),
    (0.0, PITCH_W),
)

#: How much of the virtual plane a zoom of 1.0 fills the frame with.
BASE_WINDOW = 1500.0

GRASS = (52, 118, 58)
GRASS_LIGHT = (58, 130, 64)
LINE = (238, 240, 238)
STAND = (44, 40, 46)
BALL = (250, 250, 250)

_PITCH_LINES: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = (
    ((0.0, 0.0), (PITCH_L, 0.0)),
    ((PITCH_L, 0.0), (PITCH_L, PITCH_W)),
    ((PITCH_L, PITCH_W), (0.0, PITCH_W)),
    ((0.0, PITCH_W), (0.0, 0.0)),
    ((PITCH_L / 2, 0.0), (PITCH_L / 2, PITCH_W)),
    ((0.0, 13.85), (16.5, 13.85)),
    ((16.5, 13.85), (16.5, 54.15)),
    ((16.5, 54.15), (0.0, 54.15)),
    ((0.0, 24.84), (5.5, 24.84)),
    ((5.5, 24.84), (5.5, 43.16)),
    ((5.5, 43.16), (0.0, 43.16)),
    ((PITCH_L, 13.85), (88.5, 13.85)),
    ((88.5, 13.85), (88.5, 54.15)),
    ((88.5, 54.15), (PITCH_L, 54.15)),
    ((PITCH_L, 24.84), (99.5, 24.84)),
    ((99.5, 24.84), (99.5, 43.16)),
    ((99.5, 43.16), (PITCH_L, 43.16)),
    ((0.0, 30.34), (-2.0, 30.34)),
    ((-2.0, 30.34), (-2.0, 37.66)),
    ((-2.0, 37.66), (0.0, 37.66)),
    ((PITCH_L, 30.34), (107.0, 30.34)),
    ((107.0, 30.34), (107.0, 37.66)),
    ((107.0, 37.66), (PITCH_L, 37.66)),
)

#: Titles a broadcast would actually put on a full-screen graphic.
_GRAPHIC_TITLES: dict[Event, str] = {
    Event.CARD: "YELLOW CARD",
    Event.SUBSTITUTION: "SUBSTITUTION",
    Event.GOAL: "GOAL",
}

_KIT_BGR: dict[str, tuple[int, int, int]] = {
    "red": (48, 48, 214),
    "blue": (200, 92, 40),
    "white": (238, 238, 238),
    "black": (38, 38, 38),
    "yellow": (62, 212, 236),
    "green": (74, 158, 72),
    "orange": (40, 142, 240),
    "purple": (158, 62, 112),
    "sky": (226, 182, 92),
    "claret": (62, 42, 122),
    "navy": (112, 62, 32),
    "amber": (42, 178, 236),
    "maroon": (54, 40, 96),
    "pink": (168, 140, 240),
}


def kit_colour(kit: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    """The first colour word in a team sheet's kit description, as BGR.

    The kit string is written for the caller to read, so this reads it too
    rather than keeping a second, silently diverging table of colours.
    """
    for word in kit.lower().replace(",", " ").split():
        if word in _KIT_BGR:
            return _KIT_BGR[word]
    return fallback


# ------------------------------------------------------------- timestamps

TS_DATA_BITS = 24
TS_CHECK_BITS = 8
#: Two reference blocks (one white, one black) then data then checksum.
TS_BLOCKS = 2 + TS_DATA_BITS + TS_CHECK_BITS
_TS_MODULUS = 1 << TS_DATA_BITS

#: The narrowest thing any agent is ever handed is the board reader's crop of
#: the top-left corner, and the oracle has to read the time out of that too.
#: So the strip is sized to fit inside the crop rather than to a fixed pixel
#: width, with a little margin left so rounding cannot push the last block
#: over the edge. This assumes the crop starts at the left edge, which every
#: preset does, because that is where broadcasters put the bug.
TS_STRIP_FRACTION = min(0.40, (SETTINGS.board.crop[2] - SETTINGS.board.crop[0]) * 0.95)
_TS_CROP_FRACTION = SETTINGS.board.crop[2] - SETTINGS.board.crop[0]


def _block_width(width: float) -> float:
    """Block edge for a frame this wide, unrounded.

    The strip is a fixed fraction of the frame rather than a fixed number of
    pixels, so a uniform resize scales the blocks by exactly the same factor
    and the decoder can recompute their size from the width it is handed. Round
    this to whole pixels too early and the error compounds across thirty-four
    blocks until the far end of the strip is read a block out.
    """
    return TS_STRIP_FRACTION * width / TS_BLOCKS


def ts_block_size(width: int) -> int:
    """Block edge for a frame this wide, in whole pixels."""
    return int(_block_width(width))


#: Below this a block is too small to read back after JPEG.
TS_MIN_BLOCK = 3


def _checksum(value: int) -> int:
    return ((value >> 16) ^ (value >> 8) ^ value ^ 0xA5) & 0xFF


def encode_ts(image: np.ndarray, ts: float) -> None:
    """Burn ``ts`` into the top-left corner of ``image``, in place.

    Milliseconds in twenty-four bits wraps after four and a half hours, which
    no match reaches. The two reference blocks let the decoder recover the
    black and white levels after compression instead of trusting a constant.
    """
    height, width = image.shape[:2]
    block = ts_block_size(int(width))
    if block < TS_MIN_BLOCK or height < block:
        raise ValueError(f"a {width}x{height} frame is too small to carry a timestamp strip")
    value = int(round(ts * 1000.0)) % _TS_MODULUS
    bits = [1, 0]
    bits += [(value >> i) & 1 for i in range(TS_DATA_BITS - 1, -1, -1)]
    check = _checksum(value)
    bits += [(check >> i) & 1 for i in range(TS_CHECK_BITS - 1, -1, -1)]
    edge = _block_width(width)
    for i, bit in enumerate(bits):
        x0, x1 = int(round(i * edge)), int(round((i + 1) * edge))
        image[0:block, x0:x1] = 255 if bit else 0


def _read_strip(gray: np.ndarray, block: float) -> float | None:
    height, width = gray.shape[:2]
    if block < 3.0 or TS_BLOCKS * block > width + 0.5 or block > height:
        return None
    y0 = int(round(block * 0.3))
    y1 = max(y0 + 1, int(round(block * 0.7)))
    if y1 > height:
        return None
    levels: list[float] = []
    for i in range(TS_BLOCKS):
        x0 = int(round(i * block + block * 0.3))
        x1 = max(x0 + 1, int(round(i * block + block * 0.7)))
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        levels.append(float(patch.mean()))
    white, black = levels[0], levels[1]
    if white < 170.0 or black > 85.0 or white - black < 60.0:
        return None
    threshold = (white + black) / 2.0
    bits = [1 if v > threshold else 0 for v in levels[2:]]
    value = 0
    for bit in bits[:TS_DATA_BITS]:
        value = (value << 1) | bit
    check = 0
    for bit in bits[TS_DATA_BITS:]:
        check = (check << 1) | bit
    if check != _checksum(value):
        return None
    return value / 1000.0


def _candidate_blocks(width: int) -> list[float]:
    """Block sizes to try, most likely first.

    Two of these are arithmetic rather than guesswork. An image this wide is
    either a whole frame, in which case the block size follows from its width,
    or the board reader's crop of one, in which case the original frame was
    ``width / crop fraction`` across and the blocks inside the crop are still
    the size that frame gave them. The sweep after them only matters for an
    image that has been resized by something other than the two paths we
    control, and the checksum is what keeps it from inventing an answer.
    """
    exact = [
        _block_width(width),
        _block_width(width / _TS_CROP_FRACTION),
    ]
    guesses = [*exact, *(3.0 + 0.25 * k for k in range(117))]
    seen: list[float] = []
    for g in guesses:
        if g >= TS_MIN_BLOCK and all(abs(g - s) > 1e-6 for s in seen):
            seen.append(g)
    return seen


def decode_ts(image: np.ndarray) -> float | None:
    """Recover the video time from a frame, or ``None`` if the strip is gone.

    The frame may arrive resized, JPEG-compressed, or cropped to the score
    bug. The block size follows from the image width in the first two cases,
    so those are tried first and the common path costs one pass; the strip
    carries a checksum so anything else can be swept for without the decoder
    ever inventing a time.
    """
    gray = image.mean(axis=2) if image.ndim == 3 else image.astype(np.float64)
    for block in _candidate_blocks(int(gray.shape[1])):
        value = _read_strip(gray, block)
        if value is not None:
            return value
    return None


# ---------------------------------------------------------------- drawing


def _homography() -> np.ndarray:
    src = np.array(_WORLD_CORNERS, dtype=np.float32)
    dst = np.array(_VIEW_CORNERS, dtype=np.float32)
    matrix: np.ndarray = np.asarray(cv2.getPerspectiveTransform(src, dst), dtype=np.float64)
    return matrix


#: A dot smaller than this gets no number printed on it, because at that size
#: the digits would be two or three pixels of mush. The simulator's tracker
#: reads the same constant: what is unreadable in the picture must be
#: unreadable to anything claiming to read the picture.
NUMBER_LEGIBLE_RADIUS = 11


def _circle_points(cx: float, cy: float, radius: float, n: int = 48) -> list[tuple[float, float]]:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return [(cx + radius * float(np.cos(a)), cy + radius * float(np.sin(a))) for a in angles]


def _text(
    image: np.ndarray,
    s: str,
    org: tuple[int, int],
    scale: float,
    colour: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(image, s, org, cv2.FONT_HERSHEY_DUPLEX, scale, colour, thickness, cv2.LINE_AA)


class BroadcastRenderer:
    """Draws frames for one match. Reused across frames for the precomputation.

    Holding the homography, the kit colours and the crowd speckle on the
    instance keeps a frame down to a few hundred OpenCV calls, which is what
    makes a 90 minute render tractable.
    """

    def __init__(
        self,
        pack: KnowledgePack,
        *,
        width: int = 1280,
        height: int = 720,
        seed: int = 3,
    ) -> None:
        self.pack = pack
        self.width = width
        self.height = height
        self._h = _homography()
        #: Overlays are laid out at 720p and scaled from there.
        self._scale = width / 1280.0
        self.home_colour = kit_colour(pack.home.kit, (48, 48, 214))
        self.away_colour = kit_colour(pack.away.kit, (200, 92, 40))

        # Filling a 720p array from a colour tuple is slower than copying one
        # that already exists, and this runs once per frame for ninety minutes.
        self._blank = np.full((height, width, 3), GRASS, dtype=np.uint8)

        rng = np.random.default_rng(seed)
        n = 2200
        self._crowd_pts = np.stack(
            [rng.uniform(-500.0, 2900.0, n), rng.uniform(-820.0, _FAR_Y - 6.0, n)],
            axis=1,
        )
        self._crowd_cols = rng.integers(70, 205, size=(n, 3)).astype(np.int32)

    # -- geometry ---------------------------------------------------------

    def to_view(self, pts: Sequence[tuple[float, float]]) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        homo = np.concatenate([arr, np.ones((len(arr), 1))], axis=1) @ self._h.T
        return np.asarray(homo[:, :2] / homo[:, 2:3])

    def camera(self, state: SimState) -> tuple[float, float, float]:
        """Where the camera is looking, as (view x, view y, pixels per view unit)."""
        focus = self.to_view([state.focus])[0]
        if state.scene is Scene.CLOSE_UP:
            centre_y = float(focus[1])
        else:
            centre_y = 480.0 + 0.22 * (float(focus[1]) - 530.0)
        window = BASE_WINDOW / max(0.2, state.zoom)
        return float(focus[0]), centre_y, self.width / window

    def _screen(
        self, pts: Sequence[tuple[float, float]], cam: tuple[float, float, float]
    ) -> np.ndarray:
        view = self.to_view(pts)
        return self._view_to_screen(view, cam)

    def _view_to_screen(self, view: np.ndarray, cam: tuple[float, float, float]) -> np.ndarray:
        cx, cy, scale = cam
        out = np.empty_like(view)
        out[:, 0] = (view[:, 0] - cx) * scale + self.width / 2.0
        out[:, 1] = (view[:, 1] - cy) * scale + self.height / 2.0
        return np.asarray(np.rint(out).astype(np.int32))

    # -- layers -----------------------------------------------------------

    def _draw_stand(self, image: np.ndarray, cam: tuple[float, float, float]) -> None:
        far = self._view_to_screen(np.array(_VIEW_CORNERS[:2], dtype=np.float64), cam)
        (x0, y0), (x1, y1) = far[0], far[1]
        if y0 < 0 and y1 < 0:
            return
        span = max(1, int(x1 - x0))
        left_y = int(y0 - (x0 + 3000) * (y1 - y0) / span)
        right_y = int(y1 + (self.width + 3000 - x1) * (y1 - y0) / span)
        far_right = self.width + 3000
        poly = np.array(
            [[-3000, left_y], [far_right, right_y], [far_right, -4000], [-3000, -4000]],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [poly], STAND)
        pts = self._view_to_screen(self._crowd_pts, cam)
        on = (
            (pts[:, 0] >= -4)
            & (pts[:, 0] < self.width + 4)
            & (pts[:, 1] >= -4)
            & (pts[:, 1] < self.height + 4)
        )
        radius = max(1, int(round(cam[2] * 2.4)))
        for (px, py), col in zip(pts[on], self._crowd_cols[on], strict=True):
            shade = (int(col[0]), int(col[1]), int(col[2]))
            cv2.circle(image, (int(px), int(py)), radius, shade, -1)

    def _draw_pitch(self, image: np.ndarray, cam: tuple[float, float, float]) -> None:
        bands = 10
        for i in range(bands):
            x0 = PITCH_L * i / bands
            x1 = PITCH_L * (i + 1) / bands
            quad = self._screen([(x0, 0.0), (x1, 0.0), (x1, PITCH_W), (x0, PITCH_W)], cam)
            cv2.fillPoly(image, [quad], GRASS_LIGHT if i % 2 else GRASS)

        thickness = max(1, int(round(cam[2] * 2.2)))
        for a, b in _PITCH_LINES:
            seg = self._screen([a, b], cam)
            cv2.line(image, tuple(seg[0]), tuple(seg[1]), LINE, thickness, cv2.LINE_AA)
        circle = self._screen(_circle_points(PITCH_L / 2, PITCH_W / 2, 9.15), cam)
        cv2.polylines(image, [circle], True, LINE, thickness, cv2.LINE_AA)
        spots = ((PITCH_L / 2, PITCH_W / 2), (11.0, PITCH_W / 2), (PITCH_L - 11.0, PITCH_W / 2))
        for spot in spots:
            mark = self._screen([spot], cam)[0]
            cv2.circle(image, (int(mark[0]), int(mark[1])), thickness + 1, LINE, -1, cv2.LINE_AA)

    def dots_on_screen(
        self, state: SimState, cam: tuple[float, float, float]
    ) -> list[tuple[Dot, tuple[int, int], int]]:
        """Every player the camera can see, as ``(dot, centre, radius)``.

        Legibility is decided by this projection and this radius, so anything
        that needs to know what the picture shows asks here rather than
        keeping a second copy that would drift.
        """
        world = [(d.x, d.y) for d in state.players]
        view = self.to_view(world)
        screen = self._view_to_screen(view, cam)
        visible: list[tuple[Dot, tuple[int, int], int]] = []
        for dot, vpt, spt in zip(state.players, view, screen, strict=True):
            depth = 0.55 + 0.85 * float((vpt[1] - _FAR_Y) / (_NEAR_Y - _FAR_Y))
            radius = int(round(13.0 * depth * cam[2]))
            on_screen = -60 < spt[0] < self.width + 60 and -60 < spt[1] < self.height + 60
            if radius < 2 or not on_screen:
                continue
            visible.append((dot, (int(spt[0]), int(spt[1])), radius))
        return visible

    def _draw_players(
        self, image: np.ndarray, state: SimState, cam: tuple[float, float, float]
    ) -> None:
        for dot, spt, radius in self.dots_on_screen(state, cam):
            colour = self.home_colour if dot.side is Side.HOME else self.away_colour
            cv2.circle(image, (int(spt[0]), int(spt[1])), radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(image, (int(spt[0]), int(spt[1])), radius, colour, -1, cv2.LINE_AA)
            if radius >= NUMBER_LEGIBLE_RADIUS:
                label = str(dot.number)
                scale = radius / 20.0
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
                _text(
                    image,
                    label,
                    (int(spt[0]) - tw // 2, int(spt[1]) + th // 2),
                    scale,
                    (250, 250, 250),
                    1,
                )

    def _draw_ball(
        self, image: np.ndarray, state: SimState, cam: tuple[float, float, float]
    ) -> None:
        view = self.to_view([state.ball])
        spt = self._view_to_screen(view, cam)[0]
        depth = 0.55 + 0.85 * float((view[0][1] - _FAR_Y) / (_NEAR_Y - _FAR_Y))
        radius = max(3, int(round(6.5 * depth * cam[2])))
        cv2.circle(image, (int(spt[0]), int(spt[1])), radius + 2, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(image, (int(spt[0]), int(spt[1])), radius, BALL, -1, cv2.LINE_AA)

    # -- overlays ---------------------------------------------------------

    def _px(self, value: float) -> int:
        """A length designed at 720p, in pixels for the frame we are drawing.

        Overlay geometry has to scale with the frame or the score bug falls
        out of the board reader's crop the moment someone captures at 640x360,
        which the runtime tests do.
        """
        return int(round(value * self._scale))

    def _draw_score_bug(self, image: np.ndarray, state: SimState) -> None:
        """The score bug, kept inside ``BoardConfig.crop`` on purpose.

        The board reader crops the top-left corner and sends only that, so
        anything the bug needs to say has to live in the same box, at any
        capture resolution.
        """
        p = self._px
        x0, y0, x1, y1 = p(18), p(22), p(500), p(102)
        font, small = 0.95 * self._scale, 0.85 * self._scale
        weight = max(1, self._px(2))
        cv2.rectangle(image, (x0, y0), (x1, y1), (26, 24, 26), -1)
        cv2.rectangle(image, (x0, y0), (x1, y1), (200, 200, 200), 1)
        cv2.rectangle(image, (x0 + p(6), y0 + p(8)), (x0 + p(20), y1 - p(8)), self.home_colour, -1)
        cv2.rectangle(
            image, (x0 + p(232), y0 + p(8)), (x0 + p(246), y1 - p(8)), self.away_colour, -1
        )
        _text(image, self.pack.home.short, (x0 + p(30), y0 + p(56)), font, (240, 240, 240), weight)
        _text(
            image,
            f"{state.home_score} - {state.away_score}",
            (x0 + p(128), y0 + p(56)),
            font,
            (240, 240, 240),
            weight,
        )
        _text(image, self.pack.away.short, (x0 + p(256), y0 + p(56)), font, (240, 240, 240), weight)
        cv2.rectangle(image, (x0 + p(344), y0 + p(6)), (x1 - p(6), y1 - p(6)), (54, 50, 54), -1)
        _text(image, state.clock, (x0 + p(356), y0 + p(56)), small, (240, 240, 240), weight)

    def _draw_replay_badge(self, image: np.ndarray) -> None:
        p = self._px
        x1 = self.width - p(26)
        x0 = x1 - p(214)
        y0, y1 = p(24), p(86)
        weight = max(1, self._px(2))
        cv2.rectangle(image, (x0, y0), (x1, y1), (36, 28, 158), -1)
        cv2.rectangle(image, (x0, y0), (x1, y1), (230, 230, 230), weight)
        _text(image, "REPLAY", (x0 + p(22), p(68)), 1.0 * self._scale, (245, 245, 245), weight)

    def _draw_lower_third(self, image: np.ndarray, player: Player, side: Side) -> None:
        p = self._px
        colour = self.home_colour if side is Side.HOME else self.away_colour
        x0, y0 = p(96), self.height - p(176)
        x1, y1 = p(812), self.height - p(96)
        weight = max(1, self._px(2))
        cv2.rectangle(image, (x0, y0), (x1, y1), (22, 22, 24), -1)
        cv2.rectangle(image, (x0, y1), (x1, y1 + p(8)), colour, -1)
        cv2.rectangle(image, (x0 + p(10), y0 + p(10)), (x0 + p(78), y1 - p(10)), colour, -1)
        number = str(player.number) if player.number is not None else "-"
        _text(image, number, (x0 + p(24), y0 + p(58)), 1.1 * self._scale, (245, 245, 245), weight)
        _text(
            image,
            player.name.upper(),
            (x0 + p(100), y0 + p(44)),
            0.95 * self._scale,
            (242, 242, 242),
            weight,
        )
        if player.position:
            _text(
                image,
                player.position,
                (x0 + p(100), y1 - p(14)),
                0.55 * self._scale,
                (176, 176, 180),
                1,
            )

    def _draw_graphic_panel(self, image: np.ndarray, state: SimState) -> None:
        p = self._px
        x0, y0 = p(180), p(150)
        x1, y1 = self.width - p(180), self.height - p(170)
        weight = max(1, self._px(2))
        panel = image[y0:y1, x0:x1]
        tint = np.full_like(panel, (24, 22, 28))
        image[y0:y1, x0:x1] = cv2.addWeighted(panel, 0.18, tint, 0.82, 0)
        cv2.rectangle(image, (x0, y0), (x1, y1), (210, 210, 214), weight)
        title = _GRAPHIC_TITLES.get(state.event, state.event.value.replace("_", " ").upper())
        _text(
            image,
            title,
            (x0 + p(44), y0 + p(92)),
            1.5 * self._scale,
            (245, 245, 245),
            weight + 1,
        )
        if state.graphic is not None:
            _text(
                image,
                state.graphic.name,
                (x0 + p(44), y0 + p(160)),
                1.0 * self._scale,
                (226, 226, 230),
                weight,
            )
        team = self.pack.team(state.possession)
        if team is not None:
            _text(
                image,
                team.name,
                (x0 + p(44), y0 + p(216)),
                0.8 * self._scale,
                (186, 186, 192),
                weight,
            )

    # -- the frame --------------------------------------------------------

    def frame(self, state: SimState) -> np.ndarray:
        """One 1280x720 BGR frame of fake broadcast."""
        image: np.ndarray = self._blank.copy()
        cam = self.camera(state)
        self._draw_stand(image, cam)
        self._draw_pitch(image, cam)
        self._draw_players(image, state, cam)
        self._draw_ball(image, state, cam)

        if state.scene is Scene.REPLAY:
            # Broadcasters wash the colour out of a replay and pull the bug.
            # The board reader is supposed to notice, so make it noticeable.
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            washed: np.ndarray = cv2.addWeighted(
                image, 0.35, cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), 0.65, -12
            )
            image = washed
            self._draw_replay_badge(image)
        else:
            self._draw_score_bug(image, state)

        if state.scene is Scene.GRAPHIC:
            self._draw_graphic_panel(image, state)
        elif state.graphic is not None:
            self._draw_lower_third(image, state.graphic, state.graphic_side)

        encode_ts(image, state.ts)
        return image
