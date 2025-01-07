import numpy as np
import matplotlib.pyplot as plt


def visualize_trajectories(actions):
    # Visualize all trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab20(np.linspace(0, 1, actions.shape[0]))
    for i in range(actions.shape[0]):
        trajectory = actions[i, :, 3:6]
        current_position = np.array([0.0, 0.0, 0.0])
        prev_position = current_position

        for j in range(trajectory.shape[0]):
            direction = trajectory[j]  # Direction vector at this waypoint
            magnitude = np.linalg.norm(direction)  # Magnitude of the direction vector
            direction_normalized = direction / magnitude if magnitude != 0 else direction  # Normalize the direction

            # Calculate the step (scaled direction)
            step = magnitude * direction_normalized
            ax.quiver(prev_position[0], prev_position[1], prev_position[2], trajectory[j][0], trajectory[j][1], trajectory[j][2], color=colors[i])
            prev_position = prev_position + step  # Move to the new position
        # positions = np.array(positions)
        # ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.set_xlim([-0.3, 0.3])
    ax.set_ylim([-0.3, 0.3])
    ax.set_zlim([0, 0.3])
    plt.show()