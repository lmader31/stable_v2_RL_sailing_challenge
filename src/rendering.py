"""
Shared rendering utilities for the sailing environment.

All visual styling is centralized here so that env_sailing._render_frame()
and visualization.py (race view / GIF) produce a consistent look.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import binary_dilation, binary_erosion


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
OCEAN_DEEP = np.array([0.06, 0.15, 0.30])    # dark navy at edges
OCEAN_MID  = np.array([0.18, 0.42, 0.68])     # mid-blue
OCEAN_LIGHT = np.array([0.30, 0.58, 0.82])    # lighter center

ISLAND_SAND  = np.array([222, 200, 140, 255], dtype=np.uint8)
ISLAND_GREEN = np.array([74, 139, 63, 255], dtype=np.uint8)
ISLAND_DARK_GREEN = np.array([50, 100, 45, 255], dtype=np.uint8)
ISLAND_SHORE = np.array([240, 225, 180, 255], dtype=np.uint8)

BOAT_RED = '#C0392B'
SAIL_WHITE = '#F5F5F0'
GOAL_GREEN_INNER = '#27AE60'
GOAL_GREEN_OUTER = '#2ECC71'


# ---------------------------------------------------------------------------
# Ocean background
# ---------------------------------------------------------------------------
def _build_ocean_layer(grid_w, grid_h):
    """Return an (H, W, 3) float array with a radial-gradient ocean."""
    cy, cx = grid_h / 2.0, grid_w / 2.0
    max_r = np.sqrt(cx**2 + cy**2)
    Y, X = np.meshgrid(np.linspace(0, grid_h, grid_h),
                        np.linspace(0, grid_w, grid_w), indexing='ij')
    r = np.sqrt((X - cx)**2 + (Y - cy)**2) / max_r  # 0 at center, 1 at corners
    r = np.clip(r, 0, 1)

    ocean = (OCEAN_LIGHT[None, None, :] * (1 - r[:, :, None])
             + OCEAN_MID[None, None, :] * r[:, :, None])

    # faint wave pattern
    freq = 0.12
    wave = 0.03 * np.sin(2 * np.pi * freq * X + 0.7 * Y)
    wave += 0.02 * np.sin(2 * np.pi * freq * 0.7 * Y - 0.3 * X)
    ocean += wave[:, :, None]
    return np.clip(ocean, 0, 1)


# ---------------------------------------------------------------------------
# Island layer (precomputed RGBA)
# ---------------------------------------------------------------------------
def build_island_layer(world_map):
    """
    Build an RGBA island image with beach rim, green interior, and outline.

    Parameters
    ----------
    world_map : ndarray (H, W), 1 = island, 0 = water

    Returns
    -------
    layer : ndarray (H, W, 4) uint8  (RGBA, pre-multiplied alpha)
    """
    H, W = world_map.shape
    layer = np.zeros((H, W, 4), dtype=np.uint8)
    island_mask = world_map == 1

    struct = np.ones((5, 5), dtype=bool)
    eroded = binary_erosion(island_mask, structure=struct, iterations=1)
    deep_interior = binary_erosion(island_mask, structure=struct, iterations=2)

    shore_ring = island_mask & ~eroded
    sand_ring = eroded & ~deep_interior

    layer[deep_interior] = ISLAND_DARK_GREEN
    layer[sand_ring] = ISLAND_GREEN
    layer[shore_ring] = ISLAND_SHORE

    outline = binary_dilation(island_mask, structure=np.ones((3, 3), dtype=bool)) & ~island_mask
    layer[outline] = np.array([40, 60, 30, 180], dtype=np.uint8)

    return layer


# ---------------------------------------------------------------------------
# Draw helpers (all operate on an existing Axes)
# ---------------------------------------------------------------------------

def draw_ocean(ax, grid_w, grid_h):
    """Fill *ax* with the gradient ocean background."""
    ocean_img = _build_ocean_layer(grid_w, grid_h)
    ax.imshow(ocean_img, extent=(0, grid_w, 0, grid_h), origin='lower',
              aspect='auto', zorder=0)


def draw_island(ax, island_layer, grid_w, grid_h):
    """Overlay the precomputed RGBA island layer."""
    ax.imshow(island_layer, extent=(0, grid_w, 0, grid_h),
              origin='lower', zorder=2)


def draw_wind(ax, wind_field, grid_w, grid_h, density=32, arrow_scale=360):
    """Draw wind arrows colored by direction."""
    step = max(1, grid_w // density)
    x = np.arange(0, grid_w, step)
    y = np.arange(0, grid_h, step)
    X, Y = np.meshgrid(x, y)
    U = wind_field[::step, ::step, 0]
    V = wind_field[::step, ::step, 1]

    angles = np.arctan2(V, U)
    hue = (angles + np.pi) / (2 * np.pi)
    hsv = np.stack([hue, np.full_like(hue, 0.35), np.full_like(hue, 1.0)], axis=-1)
    colors = hsv_to_rgb(hsv).reshape(-1, 3)

    ax.quiver(X, Y, U, V, color=colors, alpha=0.50, scale=arrow_scale,
              width=0.0025, headwidth=3.5, headlength=4, zorder=3)


def draw_goal(ax, goal_pos):
    """Draw goal marker: concentric glowing rings + flag."""
    gx, gy = goal_pos[0], goal_pos[1]
    ax.add_patch(plt.Circle((gx, gy), 6.0, color=GOAL_GREEN_OUTER, alpha=0.12, zorder=4))
    ax.add_patch(plt.Circle((gx, gy), 4.5, color=GOAL_GREEN_OUTER, alpha=0.22, zorder=4))
    ax.add_patch(plt.Circle((gx, gy), 3.0, color=GOAL_GREEN_INNER, alpha=0.70, zorder=4))

    pole_base = np.array([gx, gy])
    pole_top = pole_base + np.array([0, 6.0])
    ax.plot([pole_base[0], pole_top[0]], [pole_base[1], pole_top[1]],
            color='white', linewidth=1.8, zorder=5)
    flag_verts = np.array([pole_top,
                           pole_top + np.array([4.0, -1.5]),
                           pole_top + np.array([0, -3.0])])
    ax.add_patch(Polygon(flag_verts, closed=True,
                         facecolor=GOAL_GREEN_INNER, edgecolor='white',
                         linewidth=0.8, alpha=0.9, zorder=5))


def draw_boat(ax, position, velocity, color=BOAT_RED, boat_length=9.0,
              boat_width=5.0, draw_velocity=True):
    """Draw boat hull + sail + optional velocity arrow.  Returns None."""
    direction = velocity.copy()
    if np.linalg.norm(direction) < 0.1:
        direction = np.array([0.0, 1.0])
    else:
        direction = direction / np.linalg.norm(direction)

    bow = position + direction * boat_length * 0.6
    stern_center = position - direction * boat_length * 0.4
    perp = np.array([-direction[1], direction[0]])
    port = stern_center + perp * boat_width * 0.5
    starboard = stern_center - perp * boat_width * 0.5

    hull = Polygon(np.array([bow, port, stern_center, starboard]),
                   closed=True, facecolor=color, edgecolor='#1a1a1a',
                   linewidth=1.2, alpha=0.92, zorder=7)
    ax.add_patch(hull)

    sail_base = position + direction * boat_length * 0.05
    sail_tip = position + direction * boat_length * 0.45
    sail_offset = perp * boat_width * 0.28
    sail_verts = np.array([sail_base - sail_offset,
                           sail_base + sail_offset,
                           sail_tip])
    sail = Polygon(sail_verts, closed=True, facecolor=SAIL_WHITE,
                   edgecolor='#999999', linewidth=0.8, alpha=0.85, zorder=8)
    ax.add_patch(sail)

    if draw_velocity and np.linalg.norm(velocity) > 0.2:
        ax.arrow(bow[0], bow[1], velocity[0] * 1.5, velocity[1] * 1.5,
                 head_width=1.5, head_length=1.0,
                 fc='#F1C40F', ec='#F39C12', alpha=0.8, zorder=9)


def draw_trajectory(ax, positions, color='#F1C40F', full=True):
    """Draw a fading trajectory line with dot markers."""
    if len(positions) < 2:
        return
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    n = len(xs)
    for i in range(n - 1):
        a = 0.25 + 0.55 * (i / n)
        ax.plot(xs[i:i+2], ys[i:i+2], color=color, linewidth=2, alpha=a, zorder=5)
    interval = max(1, n // 10)
    for i in range(0, n - 1, interval):
        ax.scatter(xs[i], ys[i], s=25, color=color, alpha=0.55,
                   zorder=6, edgecolors='#E67E22', linewidths=0.5)


# ---------------------------------------------------------------------------
# Convenience: draw the full scene background (ocean + island + wind + goal)
# ---------------------------------------------------------------------------

def draw_scene(ax, grid_size, island_layer, wind_field, goal_position,
               wind_density=32, wind_arrow_scale=360):
    """
    Draw the complete static scene (no boats) onto *ax*.

    Call this from both ``SailingEnv._render_frame`` and the race visualiser
    to guarantee identical styling.
    """
    grid_w, grid_h = grid_size
    ax.set_xlim(-1, grid_w + 1)
    ax.set_ylim(-1, grid_h + 1)

    draw_ocean(ax, grid_w, grid_h)
    draw_island(ax, island_layer, grid_w, grid_h)
    draw_wind(ax, wind_field, grid_w, grid_h,
              density=wind_density, arrow_scale=wind_arrow_scale)
    draw_goal(ax, goal_position)

    ax.set_xticks([])
    ax.set_yticks([])
