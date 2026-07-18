#!/usr/bin/env python3
"""Generate test videos: rotating cube with rainbow gradient faces on dark background.

Usage:
    python tests/samples/generate.py            # full render (numpy + PIL + ffmpeg)
    python tests/samples/generate.py --fast     # quick ffmpeg testsrc2 pattern
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

VIDEOS = [
    ("720p.mp4", 1280, 720),
    ("1080p.mp4", 1920, 1080),
    ("4k.mp4", 3840, 2160),
    ("8k.mp4", 7680, 4320),
]

FPS = 60
DURATION = 5
NUM_FRAMES = FPS * DURATION
BG = (5, 5, 20)
TEX_SIZE = 256
CAMERA_DIST = 6.0       # perspective distance (larger = wider field of view)
PROJECTION_SCALE = 0.25  # fraction of min(width, height); tuned to keep cube in frame

# Cube geometry: 8 vertices
VERTICES = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
]

# Faces: (vertex indices in quad order, outward normal)
FACES = [
    ([4, 5, 6, 7], (0, 0, 1)),    # front
    ([1, 0, 3, 2], (0, 0, -1)),   # back
    ([5, 1, 2, 6], (1, 0, 0)),    # right
    ([0, 4, 7, 3], (-1, 0, 0)),   # left
    ([7, 6, 2, 3], (0, 1, 0)),    # top
    ([0, 1, 5, 4], (0, -1, 0)),   # bottom
]


def check_dependencies():
    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        print("Required: numpy and Pillow. Install with:")
        print("  pip install numpy Pillow")
        sys.exit(1)
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    if r.returncode != 0:
        print("Required: ffmpeg must be on PATH.")
        sys.exit(1)


def hsv_to_rgb(h, s, v):
    """Vectorized HSV to RGB. h [0,360], s/v [0,1]. Returns uint8 array."""
    import numpy as np

    h60 = (h / 60.0) % 6
    i = np.floor(h60).astype(int)
    f = h60 - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    conds = [i == k for k in range(6)]
    r = np.select(conds, [v, q, p, p, t, v])
    g = np.select(conds, [t, v, v, q, p, p])
    b = np.select(conds, [p, p, t, v, v, q])

    return np.clip(np.stack([r, g, b], axis=-1) * 255, 0, 255).astype(np.uint8)


def make_face_textures():
    """Create 6 rainbow gradient face textures."""
    import numpy as np
    from PIL import Image

    y, x = np.mgrid[0:TEX_SIZE, 0:TEX_SIZE]
    diag = (x.astype(np.float64) + y) / (2 * TEX_SIZE)

    hue_bases = [0, 60, 120, 180, 240, 300]
    textures = []
    for base in hue_bases:
        h = (base + diag * 60) % 360
        s = np.full_like(h, 0.85)
        v = 0.5 + 0.5 * diag
        textures.append(Image.fromarray(hsv_to_rgb(h, s, v)))
    return textures


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def mat_mul(a, b):
    """Multiply two 3x3 matrices."""
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def mat_vec(m, v):
    """Multiply 3x3 matrix by 3-vector."""
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def find_coeffs(src_quad, dst_quad):
    """Find PIL perspective transform coefficients (dst -> src mapping)."""
    import numpy as np

    matrix = []
    for (xs, ys), (xd, yd) in zip(src_quad, dst_quad):
        matrix.append([xd, yd, 1, 0, 0, 0, -xs * xd, -xs * yd])
        matrix.append([0, 0, 0, xd, yd, 1, -ys * xd, -ys * yd])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([c for pt in src_quad for c in pt], dtype=np.float64)
    return tuple(np.linalg.solve(A, B))


def render_frame(t, width, height, textures):
    """Render one frame of the rotating cube. t in [0, 1]."""
    from PIL import Image, ImageDraw

    angle_y = t * 2 * math.pi * 2      # 2 full rotations over duration
    angle_x = math.sin(t * 2 * math.pi) * 0.3  # gentle wobble

    rot = mat_mul(rot_y(angle_y), rot_x(angle_x))

    # Transform vertices
    xformed = [mat_vec(rot, v) for v in VERTICES]

    # Perspective projection (cube centered at origin, projected to frame center)
    dist = CAMERA_DIST
    scale = min(width, height) * PROJECTION_SCALE
    screen = []
    for x, y, z in xformed:
        f = dist / (dist - z)
        screen.append((x * f * scale + width / 2, -y * f * scale + height / 2))

    # Determine visible faces and sort by depth (painter's algorithm)
    face_order = []
    for face_idx, (indices, normal) in enumerate(FACES):
        rn = mat_vec(rot, normal)
        if rn[2] > 0:  # facing camera
            center_z = sum(xformed[i][2] for i in indices) / 4
            face_order.append((center_z, face_idx, indices))
    face_order.sort(key=lambda x: x[0])  # farthest first

    img = Image.new("RGB", (width, height), BG)
    sz = TEX_SIZE - 1
    tex_quad = [(0, 0), (sz, 0), (sz, sz), (0, sz)]

    for _, face_idx, indices in face_order:
        dst_quad = [screen[i] for i in indices]
        try:
            coeffs = find_coeffs(tex_quad, dst_quad)
        except Exception:
            continue
        warped = textures[face_idx].transform(
            (width, height), Image.PERSPECTIVE, coeffs, Image.BILINEAR,
        )
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).polygon(dst_quad, fill=255)
        img.paste(warped, mask=mask)

    return img


def generate_rendered(name, width, height):
    """Generate video by rendering cube frames and piping to ffmpeg."""
    import numpy as np

    output = OUTPUT_DIR / name
    textures = make_face_textures()

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for i in range(NUM_FRAMES):
        t = i / NUM_FRAMES
        img = render_frame(t, width, height, textures)
        proc.stdin.write(np.asarray(img).tobytes())
        if (i + 1) % FPS == 0:
            print(f"  {name}: {i + 1}/{NUM_FRAMES} frames", flush=True)

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        print(f"  ffmpeg error for {name}:", proc.stderr.read().decode()[-500:])
        return False

    print(f"  {name}: done ({output.stat().st_size / 1024:.0f} KB)")
    return True


def generate_fast(name, width, height):
    """Generate video using ffmpeg's built-in testsrc2 pattern."""
    output = OUTPUT_DIR / name
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"testsrc2=size={width}x{height}:rate={FPS}:duration={DURATION}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print(f"  ffmpeg error for {name}:", r.stderr.decode()[-500:])
        return False
    print(f"  {name}: done ({output.stat().st_size / 1024:.0f} KB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate test videos")
    parser.add_argument(
        "--fast", action="store_true",
        help="Use ffmpeg testsrc2 instead of rendered cube",
    )
    args = parser.parse_args()

    if not args.fast:
        check_dependencies()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate = generate_fast if args.fast else generate_rendered

    for name, w, h in VIDEOS:
        print(f"Generating {name} ({w}x{h})...")
        if not generate(name, w, h):
            sys.exit(1)

    print("\nAll videos generated.")


if __name__ == "__main__":
    main()
