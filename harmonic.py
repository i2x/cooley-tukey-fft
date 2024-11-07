import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.widgets import Button, CheckButtons

# Initial settings for the circles
radius = 1  # Circle radius
theta = np.linspace(0, 2 * np.pi, 100)  # Angle to generate circle
height_data1 = []  # Store the height of point on circle 1
height_data2 = []  # Store the height of point on circle 2
height_data3 = []  # Store the height of point on circle 3
height_sum = []  # Store the sum of heights of all circles

# Compute points on the circle
x = radius * np.cos(theta)
y = radius * np.sin(theta)

# Create the plot and set up subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 12))
fig.subplots_adjust(left=0.2)  # Adjust left padding to add space

ax1.set_aspect('equal')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax2.set_xlim(0, 360)
ax2.set_ylim(-1.5, 1.5)
ax3.set_xlim(0, 360)
ax3.set_ylim(-4.5, 4.5)

# Draw circles in the first subplot
circle_line1, = ax1.plot(x, y, 'b-')
point1, = ax1.plot([], [], 'ro')
circle_line2, = ax1.plot(x, y, 'g-')
point2, = ax1.plot([], [], 'bo')
circle_line3, = ax1.plot(x, y, 'g-')
point3, = ax1.plot([], [], 'go')

# Initialize plot lines (placeholders) for heights and sum of heights
height_line1, = ax2.plot([], [], 'r-', label="Circle 1")
height_line2, = ax2.plot([], [], 'b-', label="Circle 2")
height_line3, = ax2.plot([], [], 'g-', label="Circle 3")
height_sum_line, = ax3.plot([], [], 'm-', label="Sum of Heights")
ax2.set_xlabel('Angle (degrees)')
ax2.set_ylabel('Height (y position)')
ax2.legend()
ax3.set_xlabel('Angle (degrees)')
ax3.set_ylabel('Sum of Heights')
ax3.legend()

# Function to clear and reset the plots
def reset(event):
    global height_data1, height_data2, height_data3, height_sum

    # Clear data lists for each circle
    height_data1.clear()
    height_data2.clear()
    height_data3.clear()
    height_sum.clear()

    # Reset data for each line without clearing the entire axis
    height_line1.set_data([], [])
    height_line2.set_data([], [])
    height_line3.set_data([], [])
    height_sum_line.set_data([], [])
    
    # Reset the positions of the points on the circles
    point1.set_data([radius], [0])
    point2.set_data([radius], [0])
    point3.set_data([radius], [0])

    # Restart the animation frame sequence
    ani.frame_seq = ani.new_frame_seq()

# Update function for rotation
def update(frame):
    angle1 = np.radians(frame)
    angle2 = np.radians(frame * 2)
    angle3 = np.radians(frame * 3)  # Circle 3 rotates at 3x speed

    # Calculate positions for points on each circle
    point_x1, point_y1 = radius * np.cos(angle1), radius * np.sin(angle1)
    point_x2, point_y2 = radius * np.cos(angle2), radius * np.sin(angle2)
    point_x3, point_y3 = radius * np.cos(angle3), radius * np.sin(angle3)

    # Update positions of the points
    point1.set_data([point_x1], [point_y1])
    point2.set_data([point_x2], [point_y2])
    point3.set_data([point_x3], [point_y3])

    # Append height data
    height_data1.append(point_y1)
    height_data2.append(point_y2)
    height_data3.append(point_y3)
    height_sum.append(point_y1 + point_y2 + point_y3)

    # Update lines for each circle's height and sum of heights
    height_line1.set_data(np.arange(len(height_data1)), height_data1)
    height_line2.set_data(np.arange(len(height_data2)), height_data2)
    height_line3.set_data(np.arange(len(height_data3)), height_data3)
    height_sum_line.set_data(np.arange(len(height_sum)), height_sum)

    # Toggle visibility based on checkboxes
    circle_line1.set_visible(checkbox.get_status()[0])
    point1.set_visible(checkbox.get_status()[0])
    height_line1.set_visible(checkbox.get_status()[0])

    circle_line2.set_visible(checkbox.get_status()[1])
    point2.set_visible(checkbox.get_status()[1])
    height_line2.set_visible(checkbox.get_status()[1])

    circle_line3.set_visible(checkbox.get_status()[2])
    point3.set_visible(checkbox.get_status()[2])
    height_line3.set_visible(checkbox.get_status()[2])

    return point1, point2, point3, height_line1, height_line2, height_line3, height_sum_line

# Create animation without blit for compatibility
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)

# Create and position the reset button
reset_ax = plt.axes([0.8, 0.01, 0.1, 0.05])
reset_button = Button(reset_ax, 'Reset')
reset_button.on_clicked(reset)

# Create checkboxes to toggle visibility of each circle
check_ax = plt.axes([0.01, 0.4, 0.15, 0.15], frameon=False)
checkbox = CheckButtons(check_ax, ['Circle 1', 'Circle 2', 'Circle 3'], [True, True, True])

plt.tight_layout()
plt.show()
