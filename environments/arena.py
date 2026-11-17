"""Flight arena (the safety net), shared by the Isaac env and RotorPy.

Pure Python, no Isaac imports. [TODO] set these to the INSIDE of your net.
The arena is centred on (0, 0) in x/y; z = 0 is the floor.
"""
ARENA_SIZE_XY = (3.0, 3.0)  # inner length x width of the net (m)
ARENA_HEIGHT = 2.5          # net ceiling height (m); <= 0 means open top
NET_MARGIN = 0.30           # hard keep-out band inside the net (m)
NET_WARN_BAND = 0.30        # soft-penalty band just inside the hard margin (m)
CRASH_HEIGHT = 0.10         # below this counts as hitting the floor (m)


def keep_in_box(floor_z: float = 0.0):
    """(xmin, xmax, ymin, ymax, zmin, zmax) the drone must stay inside, with the
    floor at `floor_z` in the caller's coordinates."""
    hx = ARENA_SIZE_XY[0] / 2.0 - NET_MARGIN
    hy = ARENA_SIZE_XY[1] / 2.0 - NET_MARGIN
    zmax = (ARENA_HEIGHT - NET_MARGIN) if ARENA_HEIGHT > 0 else 50.0
    return (-hx, hx, -hy, hy, floor_z + CRASH_HEIGHT, floor_z + zmax)
