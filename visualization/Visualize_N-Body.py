# Python script to visualize N-Body simulation data from a CSV file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D plotting
from matplotlib.animation import FuncAnimation
from matplotlib import cm  # For colormaps

# Read the CSV file
csv_file_name = 'nbody_output.csv'
data = pd.read_csv(csv_file_name)

# Display the first few rows of the DataFrame
print("\nCSV File Head:")
print(data.head())

# Extract steps and bodies
steps = data['step'].unique()
num_steps = len(steps)
print("\nNum Steps: " + str(num_steps))

num_bodies = int(len(data['step']) / len(steps));
print("Num Bodies: " + str(num_bodies))


# Access specific columns (e.g., x, y, z positions)
pos_x = data['x']
pos_y = data['y']
pos_z = data['z']


# Set up the figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Initialize body positions
x = pos_x[0:num_bodies].to_numpy()
y = pos_y[0:num_bodies].to_numpy()
z = pos_z[0:num_bodies].to_numpy()

# Generate unique colors for each particle
colors = cm.jet(np.linspace(0, 1, num_bodies))  # Use the 'jet' colormap

# Create the initial scatter plot
sc = ax.scatter(x, y, z, c=colors, marker='o')

# Set axis limits and labels
ax.set_xlim(0, max(pos_x))
ax.set_ylim(0, max(pos_y))
ax.set_zlim(0, max(pos_z))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("N-Body Simulation with " + str(num_bodies) + " Bodies")

start_index = num_steps

def update(frame):
    global x, y, z
    global start_index, num_bodies

    # Update positions with next step
    x = pos_x[start_index: start_index + num_bodies].to_numpy()
    y = pos_y[start_index: start_index + num_bodies].to_numpy()
    z = pos_z[start_index: start_index + num_bodies].to_numpy()

    # Update step index starting point
    start_index += num_bodies

    # Update the scatter object with new data
    sc._offsets3d = (x, y, z)
    return sc,

# Create the animation
ani = FuncAnimation(fig, update, frames=num_steps, interval=100, blit=False)

plt.show()

# Save the animation as an mp4
print("Processing video file export...")
ani.save('nbody_simulation.mp4', writer='ffmpeg', fps=10)
print("Video export complete.")
