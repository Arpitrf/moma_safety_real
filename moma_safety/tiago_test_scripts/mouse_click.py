import matplotlib.pyplot as plt
import numpy as np

# Enable interactive mode
plt.ion()

# Create a sample 100x100 image
your_image = np.random.rand(100, 100)

fig, ax = plt.subplots()
ax.imshow(your_image, origin='upper')

# List to store the points
clicked_points = []

def onclick(event):
    # Ignore clicks outside the axes
    if event.inaxes is not None:
        # Convert the coordinates to integers and add to the list
        ix, iy = int(event.xdata), int(event.ydata)
        clicked_points.append((ix, iy))
        print(f"Point added: x = {ix}, y = {iy}")

# Connect the click event handler
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

# Your script can continue here
# For example, you can add a loop or a condition to wait for a specific number of points
# Let's wait for 3 points to be clicked before moving on
while len(clicked_points) < 1:
    plt.pause(0.1)  # Pause to yield control to the GUI event loop

print("1 point have been clicked, continuing with the rest of the script.")

# Now, you can access the clicked_points list
for point in clicked_points:
    print(point)

# Remember to turn off interactive mode if it's no longer needed
plt.ioff()

