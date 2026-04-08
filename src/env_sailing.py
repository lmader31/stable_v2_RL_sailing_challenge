from scipy.ndimage import zoom
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.patches import Circle, Arrow # type: ignore
from typing import Tuple, Dict, Any, Optional # type: ignore
from src.sailing_physics import calculate_sailing_efficiency # type: ignore
from src.rendering import (build_island_layer, draw_scene, draw_boat, # type: ignore
                           draw_trajectory)
import gymnasium as gym # type: ignore

class SailingEnv(gym.Env): # type: ignore
    """
    A sailing navigation environment where an agent must navigate from
    a starting point to a destination while accounting for wind.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Default wind initialization parameters (NE wind with minimal variations)
    DEFAULT_WIND_INIT_PARAMS = {
            'base_speed': 10.0,
            'base_max_rotation_angle_degree': 30,
            'pattern' : (
                ((0, -1),  (1, -1)),
                ((1, -1),  (-1, -1)),
                ((-1, -1),  (-1, 1)),
            )
    }

    DEFAULT_WIND_EVOL_PARAMS = {
        'mean_rotation_angle_degree': 0.5,
        'std_rotation_angle_degree': 0.01,
    }

    def __init__(self,
                 grid_size=(128, 128),
                 wind_init_params=None,
                 wind_evol_params=None,
                 static_wind=False,
                 wind_grid_density=32,
                 wind_arrow_scale=360,
                 render_mode=None,
                 boat_performance=0.4,
                 max_speed=8.0,
                 inertia_factor=0.3,
                 reward_discount_factor=0.995,
                 max_horizon=500,
                 show_full_trajectory=False):
        """
        Initialize the sailing environment.

        Args:
            grid_size: Tuple of (width, height) for the grid
            wind_init_params: Dictionary of wind initialization parameters
            wind_evol_params: Dictionary of wind evolution parameters
            static_wind: If True, wind will remain static regardless of evolution parameters
            wind_grid_density: Number of wind arrows to display (default: 25)
            wind_arrow_scale: Scale factor for wind arrow visualization (default: 100)
            render_mode: How to render the environment
            boat_performance: How well the boat converts wind to movement
            max_speed: Maximum boat speed
            inertia_factor: How much velocity is preserved (0-1)
            reward_discount_factor: Discount factor for future rewards
            show_full_trajectory: If True, render will show the full trajectory trace (default: False)
        """
        super().__init__()

        # Store parameters
        self.grid_size = grid_size
        self.wind_init_params = wind_init_params or self.DEFAULT_WIND_INIT_PARAMS.copy()
        self.wind_evol_params = wind_evol_params or self.DEFAULT_WIND_EVOL_PARAMS.copy()
        self.static_wind = static_wind
        self.wind_grid_density = wind_grid_density
        self.wind_arrow_scale = wind_arrow_scale
        self.render_mode = render_mode
        self.boat_performance = boat_performance
        self.max_speed = max_speed
        self.inertia_factor = inertia_factor
        self.reward_discount_factor = reward_discount_factor
        self.max_horizon = max_horizon
        self.show_full_trajectory = show_full_trajectory

        # Generate new wind field
        self.wind_field = None
        self._generate_wind_field()

        # Create world map (island(s))
        self.world_map = self._create_world()
        self.island_layer = build_island_layer(self.world_map)
        self.is_stuck = False

        # Initialize boat state
        self.position = np.array([grid_size[0] // 2, 0])  # Start at bottom center
        self.velocity = np.array([0.0, 0.0])
        self.goal_position = np.array([grid_size[0] // 2, grid_size[1] - 1])  # Goal at top center

        # Initialize position history for trajectory visualization
        self.position_history = [self.position.copy()]

        # Initialize step count
        self.step_count = 0

        # Initialize last action
        self.last_action = None

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(9) # type: ignore # 0-7: Move in direction, 8: Stay in place

        # Calculate the shape for the full wind field (grid_size[0] x grid_size[1] x 2)
        wind_field_shape = (grid_size[0] * grid_size[1] * 2,)


        # Define observation space to include the full wind field and the world definition
        self.observation_space = gym.spaces.Box( # type: ignore
            low=-np.inf,
            high=np.inf,
            shape=(6 + wind_field_shape[0] + wind_field_shape[0] // 2,),  # [x, y, vx, vy, wx, wy, flattened wind field, flattened world]
            dtype=np.float32
        )

        # Initialize random number generator
        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        """Set the seed for the environment's random number generator."""
        self.np_random = np.random.default_rng(seed)

        return [seed]

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        # Initialize or reset the random number generator
        self.seed(seed)

        # Reset position to bottom center
        self.position = np.array([self.grid_size[0] // 2, 0])

        # Reset velocity
        self.velocity = np.array([0.0, 0.0])

        # Reset step count and last action
        self.step_count = 0
        self.last_action = None

        # Reset position history for trajectory visualization
        self.position_history = [self.position.copy()]

        # Generate new wind field
        self.wind_field = None
        self._generate_wind_field()

        # Create world map (island(s))
        self.world_map = self._create_world()
        self.island_layer = build_island_layer(self.world_map)
        self.is_stuck = False

        # Get initial observation
        observation = self._get_observation()

        # Get info
        info = {
            'position': self.position,
            'velocity': self.velocity,
            'wind': self._get_wind_at_position(self.position),
            'step': self.step_count
        }

        return observation, info

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: Integer in [0, 8] representing the action to take, or None to skip

        Returns:
            observation: Dictionary containing the new observation
            reward: Float reward signal
            terminated: Boolean indicating if the episode is over
            truncated: Boolean indicating if the episode was artificially terminated
            info: Dictionary containing additional information
        """

        # Store the action
        self.last_action = action

        if action is None:
            return (
                self._get_observation(),
                0.0,
                False,
                False,
                {}
            )

        self.step_count += 1

        # Convert action to direction
        direction = self._action_to_direction(action)

        # Get current wind at boat's position
        current_wind = self._get_wind_at_position(self.position)

        # Calculate new velocity based on sailing physics
        self.velocity = self._calculate_new_velocity(
            current_velocity=self.velocity,
            wind=current_wind,
            direction=direction
        )

        # final grid cell is discrete (ceil versus floor if neg/pos for symmetry)
        self.velocity = np.where(
            self.velocity < 0,
            np.ceil(self.velocity),
            np.floor(self.velocity)
        ).astype(np.int32)

        # Calculate new position
        new_position = self.position + self.velocity

        # Ensure position stays within bounds
        new_position = np.clip(
            new_position,
            [0, 0],
            [self.grid_size[0]-1, self.grid_size[1]-1]
        )

        def is_point_on_segment(A, B, epsilon=1e-9):
            xA, yA = A # boat's old position
            xB, yB = B # boat's new position
            xP, yP = self.goal_position

            # 1. Check collinearity using cross product
            cross = (xB - xA) * (yP - yA) - (yB - yA) * (xP - xA)
            if abs(cross) > epsilon:
                return False

            # 2. Check that P lies within the bounding box of segment [A, B]
            if (min(xA, xB) - epsilon <= xP <= max(xA, xB) + epsilon and
                    min(yA, yB) - epsilon <= yP <= max(yA, yB) + epsilon):
                return True

            return False

        # Ensure boat doesn't cross goal without winning (Super Naive + training 3 case)
        if is_point_on_segment(self.position, new_position):
            self.position = self.goal_position
        # Update the position normally if boat's new position is in water
        elif self.world_map[new_position[1]][new_position[0]] == 0 and not self.is_stuck:
            # Update position as usual
            self.position = new_position
        # if new position ends up in island, update one last time the position and mark the boat as stuck
        elif not self.is_stuck:
            self.position = new_position
            self.is_stuck = True

        # Update position history for trajectory visualization
        self.position_history.append(self.position.copy())

        # Check if reached goal (within 1 cell)
        distance_to_goal = np.linalg.norm(self.position - self.goal_position)
        reached_goal = distance_to_goal < 1.5

        # Calculate reward
        reward = self._calculate_reward(reached_goal, distance_to_goal)

        # Determine if episode is done
        terminated = reached_goal or self.is_stuck
        truncated = self.step_count >= self.max_horizon

        # Update windfield
        self._update_wind_field()

        observation = self._get_observation()

        info = {
            "position": self.position,
            "velocity": self.velocity,
            "wind": current_wind,
            "step_count": self.step_count,
            "distance_to_goal": distance_to_goal,
            "is_stuck": self.is_stuck,
        }

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            return None
        else:
            return self._render_frame()  # Default to rgb_array mode

    def _render_frame(self):
        """
        Render the current state as a frame.
        Uses the environment's wind_grid_density and wind_arrow_scale parameters.
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw the shared scene (ocean, island, wind, goal)
        draw_scene(ax, self.grid_size, self.island_layer, self.wind_field,
                   self.goal_position, wind_density=self.wind_grid_density,
                   wind_arrow_scale=self.wind_arrow_scale)

        # Trajectory
        if self.show_full_trajectory and len(self.position_history) > 1:
            draw_trajectory(ax, self.position_history)

        # Boat
        draw_boat(ax, self.position, self.velocity)

        # HUD
        wind_at_pos = self._get_wind_at_position(self.position)
        action_names = {
            0: "North", 1: "Northeast", 2: "East", 3: "Southeast",
            4: "South", 5: "Southwest", 6: "West", 7: "Northwest",
        }
        action_str = action_names.get(self.last_action, "None")
        info_text = (
            f"Step: {self.step_count}\n"
            f"Position: ({self.position[0]:.1f}, {self.position[1]:.1f})\n"
            f"Velocity: ({self.velocity[0]:.2f}, {self.velocity[1]:.2f})\n"
            f"Wind: ({wind_at_pos[0]:.2f}, {wind_at_pos[1]:.2f})\n"
            f"Distance to goal: {np.linalg.norm(self.position - self.goal_position):.2f}\n"
            f"Action: {action_str}"
        )
        ax.text(0.02, 0.02, info_text, fontsize=9, color='white',
                transform=ax.transAxes, verticalalignment='bottom',
                bbox=dict(facecolor='#1a2a3a', alpha=0.75, boxstyle='round,pad=0.5'))

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60',
                       label='Goal', markersize=10),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#C0392B',
                       label='Boat', markersize=10),
            plt.Line2D([0], [0], color='#888888', marker='>', linestyle='',
                       label='Wind', markersize=8),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#4A8B3F',
                       label='Island', markersize=10),
            plt.Line2D([0], [0], color='#F1C40F', marker='>', linestyle='',
                       label='Velocity', markersize=8),
            plt.Line2D([0], [0], color='#F1C40F', linewidth=2,
                       label='Trajectory'),
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                  bbox_to_anchor=(0, 0), fontsize=9,
                  facecolor='#1a2a3a', labelcolor='white',
                  framealpha=0.75, edgecolor='#555555')

        ax.set_title("Sailing Environment", fontsize=14, color='white',
                      bbox=dict(facecolor='#1a2a3a', alpha=0.8,
                                boxstyle='round,pad=0.5'))

        # Convert to image array
        fig.tight_layout()
        fig.canvas.draw()
        try:
            img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        except AttributeError:
            buf = fig.canvas.tostring_rgb()
            img = np.frombuffer(buf, dtype=np.uint8)
            total = len(img) // 3
            side = int(np.sqrt(total))
            img = img.reshape((side, side, 3))
        plt.close(fig)

        if self.render_mode == "human":
            plt.imshow(img)
            plt.axis('off')
            plt.draw()
            plt.pause(0.001)

        return img

    def _action_to_direction(self, action):
        """Convert action index to direction vector."""
        # Map actions to direction vectors using the convention:
        # NORTH = (0, 1) (increasing Y)
        # SOUTH = (0, -1) (decreasing Y)
        # EAST = (1, 0) (increasing X)
        # WEST = (-1, 0) (decreasing X)
        directions = [
            (0, 1),     # 0: North (increasing Y)
            (1, 1),     # 1: Northeast
            (1, 0),     # 2: East (increasing X)
            (1, -1),    # 3: Southeast
            (0, -1),    # 4: South (decreasing Y)
            (-1, -1),   # 5: Southwest
            (-1, 0),    # 6: West (decreasing X)
            (-1, 1),    # 7: Northwest
            (0, 0)      # 8: Stay
        ]
        return np.array(directions[action])

    def _create_world(self):
        """Separate island(s) from navigable water."""
        # 0: navigable water
        # 1: island
        world_map = np.zeros((self.grid_size[0], self.grid_size[1]))

        # rectangle
        x1, y1 = (38, 43)
        x2, y2 = (90, 85)
        world_map[y1:y2+1, x1:x2+1] = 1

        # triangle vertices
        xA, yA = 38, 43
        xB, yB = 90, 43
        xC, yC = 64, 17

        # bounding box of triangle
        xmin, xmax = min(xA, xB, xC), max(xA, xB, xC)
        ymin, ymax = min(yA, yB, yC), max(yA, yB, yC)

        # grid of candidate pixels
        X, Y = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))

        # barycentric coordinates
        den = ((yB - yC)*(xA - xC) + (xC - xB)*(yA - yC))
        w1 = ((yB - yC)*(X - xC) + (xC - xB)*(Y - yC)) / den
        w2 = ((yC - yA)*(X - xC) + (xA - xC)*(Y - yC)) / den
        w3 = 1 - w1 - w2

        mask = (w1 >= 0) & (w2 >= 0) & (w3 >= 0)

        world_map[Y[mask], X[mask]] = 1

        return world_map


    def _generate_wind_field(self):
        """
        Generate a smooth wind field from a coarse pattern (3x3, 5x5, etc.)
        and apply globally smooth rotational noise.
        """

        H, W = self.grid_size
        pattern = self.wind_init_params["pattern"]
        base_speed = self.wind_init_params["base_speed"]
        max_angle = np.deg2rad(self.wind_init_params["base_max_rotation_angle_degree"])

        n_rows = len(pattern)
        n_cols = len(pattern[0])

        # --- 1. CONVERT PATTERN TO ANGLES ---
        pattern_angles = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                dx, dy = pattern[i][j]
                pattern_angles[i, j] = np.arctan2(dy, dx)

        # --- 2. CREATE FINE GRID OF ANGLES ---
        # Meshgrid of output field
        y = np.linspace(n_rows - 1, 0, H)  # invert vertical axis
        x = np.linspace(0, n_cols - 1, W)
        Xi, Yi = np.meshgrid(x, y)  # shape HxW

        # Bilinear interpolation from pattern angles
        i0 = np.floor(Yi).astype(int)
        j0 = np.floor(Xi).astype(int)
        i1 = np.clip(i0 + 1, 0, n_rows - 1)
        j1 = np.clip(j0 + 1, 0, n_cols - 1)
        dy = Yi - i0
        dx = Xi - j0

        # Interpolate angles via sin/cos to avoid discontinuities
        angles00 = pattern_angles[i0, j0]
        angles10 = pattern_angles[i0, j1]
        angles01 = pattern_angles[i1, j0]
        angles11 = pattern_angles[i1, j1]

        cos00, sin00 = np.cos(angles00), np.sin(angles00)
        cos10, sin10 = np.cos(angles10), np.sin(angles10)
        cos01, sin01 = np.cos(angles01), np.sin(angles01)
        cos11, sin11 = np.cos(angles11), np.sin(angles11)

        # Bilinear interpolation in vector space
        cos_interp = (cos00 * (1 - dx) * (1 - dy) +
                      cos10 * dx * (1 - dy) +
                      cos01 * (1 - dx) * dy +
                      cos11 * dx * dy)
        sin_interp = (sin00 * (1 - dx) * (1 - dy) +
                      sin10 * dx * (1 - dy) +
                      sin01 * (1 - dx) * dy +
                      sin11 * dx * dy)

        base_angles = np.arctan2(sin_interp, cos_interp)

        # --- 3. ADD GLOBAL SMOOTH ROTATION NOISE ---
        # Create low-resolution noise grid (coarse)
        coarse_h, coarse_w = max(2, H // 10), max(2, W // 10)
        coarse_noise = self.np_random.uniform(-max_angle, max_angle, size=(coarse_h, coarse_w))

        # Upsample with bicubic interpolation for smoothness
        noise_field = zoom(coarse_noise, (H / coarse_h, W / coarse_w), order=3)

        # --- 4. APPLY ROTATION VECTORIELLE ---
        theta = base_angles + noise_field
        u = base_speed * np.cos(theta)
        v = base_speed * np.sin(theta)

        # --- 5. STORE IN WIND FIELD ---
        self.wind_field = np.stack([u, v], axis=-1)

    def _update_wind_field(self):
        """
        Rotate the windfield by a fixed angle (in degrees) at each time-step.
        Works for wind arrays of shape (W, H, 2).
        """

        delta_angle_deg = self.np_random.normal(
            self.wind_evol_params['mean_rotation_angle_degree'],
            self.wind_evol_params['std_rotation_angle_degree']
        )
        theta = np.deg2rad(delta_angle_deg)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        x = self.wind_field[:, :, 0]
        y = self.wind_field[:, :, 1]

        self.wind_field[:, :, 0] = x * cos_t - y * sin_t
        self.wind_field[:, :, 1] = x * sin_t + y * cos_t

    def _get_wind_at_position(self, position):
        """Get wind vector at given position."""
        x, y = position
        # Numpy arrays are indexed with [y, x] order
        return self.wind_field[int(y), int(x)]

    def _calculate_new_velocity(self, current_velocity, wind, direction):
        """
        Calculate new velocity based on sailing physics.

        Args:
            current_velocity: Current velocity vector
            wind: Wind vector at current position
            direction: Desired direction vector (normalized)

        Returns:
            new_velocity: New velocity vector
        """
        # Calculate angle between wind and direction
        wind_norm = np.linalg.norm(wind)
        if wind_norm > 0:
            wind_normalized = wind / wind_norm
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-10:  # Check for near-zero vector
                direction_normalized = np.array([1.0, 0.0])  # Default direction if input is zero
            else:
                direction_normalized = direction / direction_norm

            # Calculate sailing efficiency using the shared function
            sailing_efficiency = calculate_sailing_efficiency(direction_normalized, wind_normalized)

            # Calculate theoretical velocity (what the boat would achieve with no inertia)
            theoretical_velocity = direction * sailing_efficiency * wind_norm * self.boat_performance

            # Apply max speed limit to theoretical velocity
            speed = np.linalg.norm(theoretical_velocity)
            if speed > self.max_speed:
                theoretical_velocity = (theoretical_velocity / speed) * self.max_speed

            # Apply inertia: new_velocity = theoretical_velocity + alpha*(old_velocity - theoretical_velocity)
            # where alpha is the inertia_factor
            new_velocity = theoretical_velocity + self.inertia_factor * (current_velocity - theoretical_velocity)

            # Ensure the new velocity doesn't exceed max speed
            speed = np.linalg.norm(new_velocity)
            if speed > self.max_speed:
                new_velocity = (new_velocity / speed) * self.max_speed

        else:
            # If no wind, just maintain some inertia
            new_velocity = self.inertia_factor * current_velocity

        return new_velocity

    def _calculate_reward(self, reached_goal, distance_to_goal):
        """
        Calculate reward based on current state.

        Args:
            reached_goal: Boolean indicating if the goal was reached
            distance_to_goal: Current distance to the goal

        Returns:
            reward: 100 if goal reached, 0 otherwise
        """
        if reached_goal:
            return 100.0
        return 0.0

    def _get_observation(self):
        """
        Create the observation array [x, y, vx, vy, wx, wy, flattened wind field, flattened world].

        Returns:
            observation: A numpy array containing the agent's position, velocity,
                        the wind at the current position, the full wind field and the world definition.
        """
        # Get wind at current position
        current_wind = self._get_wind_at_position(self.position)

        # Flatten the wind field
        flattened_wind = self.wind_field.reshape(-1).astype(np.float32)

        # Flatten world
        flattened_world = self.world_map.reshape(-1).astype(np.float32)

        # Create observation array
        observation = np.concatenate([
            self.position,      # x, y
            self.velocity,      # vx, vy
            current_wind,       # wx, wy
            flattened_wind,     # Full wind field (flattened)
            flattened_world     # Full world map (flattened)
        ]).astype(np.float32)

        return observation

    @staticmethod
    def visualize_observation(observation, grid_size=None):
        """
        Create a visualization of the environment state from an observation.

        Parameters:
        - observation: The observation from the environment
        - grid_size: Optional grid size. If None, will use default from SailingEnv

        Returns:
        - img: A rendered image of the environment state
        """
        # Create a new environment
        vis_env = SailingEnv()

        # If grid_size is provided, use it
        if grid_size is not None:
            vis_env.grid_size = grid_size

        # Extract the agent's state from the observation
        agent_x, agent_y = observation[0], observation[1]
        agent_vx, agent_vy = observation[2], observation[3]

        # Update the agent's state in the environment
        vis_env.position = np.array([agent_x, agent_y])
        vis_env.velocity = np.array([agent_vx, agent_vy])

        # Extract world map size
        flattened_world_size = vis_env.world_map.size

        # Extract and reshape the wind field
        flattened_wind = observation[6:-flattened_world_size]
        wind_field = flattened_wind.reshape(vis_env.grid_size[1], vis_env.grid_size[0], 2)
        vis_env.wind_field = wind_field

        # Get the rendered image array
        img = vis_env.render()
        return img