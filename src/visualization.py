"""
Visualization utilities for sailing agents.

This module provides functions for visualizing agent trajectories and races.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from typing import Dict, Any, List, Optional
from ipywidgets import interact, IntSlider
from env_sailing import SailingEnv
from wind_scenarios import get_wind_scenario
from rendering import (build_island_layer, draw_scene, draw_boat,
                       draw_trajectory, BOAT_RED)
import io
from PIL import Image


def visualize_race(race_results: List[Dict[str, Any]], 
                   windfield_name: str, 
                   seed: int,
                   show_full_trajectories: bool = False) -> None:
    """
    Visualize multiple agents racing on the same windfield with an interactive slider.
    
    Args:
        race_results: List of dictionaries containing race results for each agent
                     Each dict should have: name, color, positions, actions, reward, steps, success
        windfield_name: Name of the windfield to visualize
        seed: Seed used for the race
        show_full_trajectories: If True, show full trajectory for each agent (default: False)
    """
    wind_scenario = get_wind_scenario(windfield_name)
    env_params = wind_scenario.get('env_params', {}).copy()

    env = SailingEnv(
        wind_init_params=wind_scenario['wind_init_params'],
        wind_evol_params=wind_scenario['wind_evol_params'],
        **env_params
    )
    env.seed(seed)
    env.reset(seed=seed)
    
    max_race_steps = max(len(result['positions']) for result in race_results)
    
    def render_race_frame(step: int) -> None:
        """Render a single frame of the race at given step."""
        fig, ax = plt.subplots(figsize=(12, 12))

        # Evolve wind to this step
        env.seed(seed)
        env.reset(seed=seed)
        for _ in range(step):
            env.step(0)

        draw_scene(ax, env.grid_size, env.island_layer, env.wind_field,
                   env.goal_position, wind_density=env.wind_grid_density,
                   wind_arrow_scale=env.wind_arrow_scale)

        legend_elements = []
        for result in race_results:
            if step < len(result['positions']):
                position = result['positions'][step]

                if step > 0:
                    velocity = position - result['positions'][step - 1]
                else:
                    velocity = np.array([0.0, 1.0])

                # Trajectory trail
                if step > 0:
                    if show_full_trajectories:
                        draw_trajectory(ax, result['positions'][:step + 1],
                                        color=result['color'])
                    else:
                        trail_len = min(10, step)
                        trail = result['positions'][max(0, step - trail_len):step + 1]
                        xs = [p[0] for p in trail]
                        ys = [p[1] for p in trail]
                        ax.plot(xs, ys, color=result['color'], alpha=0.3,
                                linewidth=5, linestyle='--', zorder=5)

                draw_boat(ax, position, velocity, color=result['color'],
                          boat_length=7.0, boat_width=4.0, draw_velocity=False)

                legend_elements.append(
                    plt.Line2D([0], [0], marker='^', color='w',
                               markerfacecolor=result['color'],
                               label=f"{result['name']}: Step {step}/{len(result['positions'])-1}",
                               markersize=12, markeredgecolor='black', markeredgewidth=1.5)
                )
            else:
                final_pos = result['positions'][-1]

                if show_full_trajectories and len(result['positions']) > 1:
                    draw_trajectory(ax, result['positions'], color=result['color'])

                if result['success']:
                    ax.scatter(final_pos[0], final_pos[1], s=300, marker='*',
                               color=result['color'], edgecolors='gold', linewidths=3,
                               zorder=10)
                else:
                    ax.scatter(final_pos[0], final_pos[1], s=200, marker='x',
                               color=result['color'], linewidths=3, zorder=10)

                status = 'FINISHED' if result['success'] else 'TIMEOUT'
                legend_elements.append(
                    plt.Line2D([0], [0],
                               marker='*' if result['success'] else 'x',
                               color='w', markerfacecolor=result['color'],
                               label=f"{result['name']}: {status}",
                               markersize=12,
                               markeredgecolor='gold' if result['success'] else 'black',
                               markeredgewidth=2)
                )

        ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
                  facecolor='#1a2a3a', labelcolor='white',
                  framealpha=0.85, edgecolor='#555555')

        ax.set_title(f"Race Visualization - Step {step}/{max_race_steps-1}\n"
                      f"Windfield: {windfield_name} | Seed: {seed}",
                      fontsize=14, color='white',
                      bbox=dict(facecolor='#1a2a3a', alpha=0.8,
                                boxstyle='round,pad=0.5'))

        info_lines = [f"Race Step: {step}"]
        for result in race_results:
            if step < len(result['positions']):
                pos = result['positions'][step]
                dist = np.linalg.norm(pos - env.goal_position)
                info_lines.append(f"{result['name']}: {dist:.1f} from goal")
            else:
                info_lines.append(
                    f"{result['name']}: Finished at step {len(result['positions'])-1}")

        ax.text(0.02, 0.02, "\n".join(info_lines), fontsize=10,
                color='white', transform=ax.transAxes, verticalalignment='bottom',
                bbox=dict(facecolor='#1a2a3a', alpha=0.75,
                          boxstyle='round,pad=0.5'))

        plt.tight_layout()
        plt.show()

    try:
        widget = interact(
            render_race_frame,
            step=IntSlider(min=0, max=max_race_steps - 1, step=1, value=0,
                           description='Race Step:')
        )
        if widget is None:
            raise RuntimeError("Widget not displayed")
    except Exception as e:
        print(f"Interactive slider not available ({type(e).__name__}). "
              "Showing static frames instead.")
        for s in [0, max_race_steps // 2, max_race_steps - 1]:
            print(f"\n--- Step {s} ---")
            render_race_frame(s)


def print_race_summary(race_results: List[Dict[str, Any]]) -> None:
    """
    Print a formatted summary of race results.
    
    Args:
        race_results: List of dictionaries containing race results for each agent
    """
    import pandas as pd
    from IPython.display import display
    
    summary_data = []
    for result in race_results:
        summary_data.append({
            'Agent': result['name'],
            'Color': result['color'],
            'Steps': result['steps'],
            'Reward': f"{result['reward']:.2f}",
            'Success': 'Yes' if result['success'] else 'No'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Steps')
    
    print("\n" + "=" * 50)
    print("RACE RESULTS")
    print("=" * 50)
    display(summary_df)
    
    successful_agents = [r for r in race_results if r['success']]
    if successful_agents:
        winner = min(successful_agents, key=lambda x: x['steps'])
        print(f"\nWINNER: {winner['name']} (completed in {winner['steps']} steps!)")
    else:
        print("\nNo agent reached the goal.")


def create_race_gif(race_results: List[Dict[str, Any]], 
                    windfield_name: str, 
                    seed: int, 
                    output_path: str,
                    fps: int = 10,
                    step_interval: int = 1,
                    figsize: tuple = (10, 10),
                    show_full_trajectories: bool = False) -> None:
    """
    Create a GIF animation of a race between multiple agents.
    
    Args:
        race_results: List of dictionaries containing race results for each agent
        windfield_name: Name of the windfield to visualize
        seed: Seed used for the race
        output_path: Path where the GIF will be saved (e.g., "race.gif")
        fps: Frames per second for the GIF (default: 10)
        step_interval: Interval between frames (1 = every step, 2 = every other step, etc.)
        figsize: Size of the figure (default: (10, 10))
        show_full_trajectories: If True, show full trajectory for each agent (default: False)
    """
    try:
        import imageio
    except ImportError:
        print("Error: 'imageio' library is required to create GIFs.")
        print("Install it with: pip install imageio")
        return
    
    print(f"Creating race GIF...")
    
    wind_scenario = get_wind_scenario(windfield_name)
    env_params = wind_scenario.get('env_params', {}).copy()
    env = SailingEnv(
        wind_init_params=wind_scenario['wind_init_params'],
        wind_evol_params=wind_scenario['wind_evol_params'],
        **env_params
    )
    env.seed(seed)
    env.reset(seed=seed)
    
    max_race_steps = max(len(result['positions']) for result in race_results)
    
    frames = []
    steps_to_render = range(0, max_race_steps, step_interval)
    
    for step in steps_to_render:
        fig, ax = plt.subplots(figsize=figsize)

        env.seed(seed)
        env.reset(seed=seed)
        for _ in range(step):
            env.step(0)

        draw_scene(ax, env.grid_size, env.island_layer, env.wind_field,
                   env.goal_position, wind_density=env.wind_grid_density,
                   wind_arrow_scale=env.wind_arrow_scale)

        legend_elements = []
        for result in race_results:
            if step < len(result['positions']):
                position = result['positions'][step]

                if step > 0:
                    velocity = position - result['positions'][step - 1]
                else:
                    velocity = np.array([0.0, 1.0])

                if step > 0:
                    if show_full_trajectories:
                        draw_trajectory(ax, result['positions'][:step + 1],
                                        color=result['color'])
                    else:
                        trail_len = min(10, step)
                        trail = result['positions'][max(0, step - trail_len):step + 1]
                        xs = [p[0] for p in trail]
                        ys = [p[1] for p in trail]
                        ax.plot(xs, ys, color=result['color'], alpha=0.3,
                                linewidth=5, linestyle='--', zorder=5)

                draw_boat(ax, position, velocity, color=result['color'],
                          boat_length=7.0, boat_width=4.0, draw_velocity=False)

                legend_elements.append(
                    plt.Line2D([0], [0], marker='^', color='w',
                               markerfacecolor=result['color'],
                               label=f"{result['name']}",
                               markersize=10, markeredgecolor='black',
                               markeredgewidth=1.5)
                )
            else:
                final_pos = result['positions'][-1]

                if show_full_trajectories and len(result['positions']) > 1:
                    draw_trajectory(ax, result['positions'],
                                    color=result['color'])

                if result['success']:
                    ax.scatter(final_pos[0], final_pos[1], s=250, marker='*',
                               color=result['color'], edgecolors='gold',
                               linewidths=2.5, zorder=10)
                    legend_elements.append(
                        plt.Line2D([0], [0], marker='*', color='w',
                                   markerfacecolor=result['color'],
                                   label=f"{result['name']} done",
                                   markersize=10, markeredgecolor='gold',
                                   markeredgewidth=1.5)
                    )
                else:
                    ax.scatter(final_pos[0], final_pos[1], s=150, marker='x',
                               color=result['color'], linewidths=2.5, zorder=10)
                    legend_elements.append(
                        plt.Line2D([0], [0], marker='x', color='w',
                                   markerfacecolor=result['color'],
                                   label=f"{result['name']} timeout",
                                   markersize=10, markeredgecolor='black',
                                   markeredgewidth=1.5)
                    )

        ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
                  facecolor='#1a2a3a', labelcolor='white',
                  framealpha=0.85, edgecolor='#555555')

        ax.set_title(f"Step {step}/{max_race_steps-1} | {windfield_name}",
                      fontsize=12, color='white', pad=10,
                      bbox=dict(facecolor='#1a2a3a', alpha=0.8,
                                boxstyle='round,pad=0.5'))

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(np.array(img))

        plt.close(fig)
        buf.close()

        if (len(frames) % 10 == 0) or (step == max_race_steps - 1):
            print(f"  Processed {len(frames)}/{len(steps_to_render)} frames...",
                  end='\r')
    
    print(f"\nSaving GIF to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF created: {output_path}")
    print(f"   Total frames: {len(frames)} | Duration: ~{len(frames)/fps:.1f}s")
