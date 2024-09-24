import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from calibration import *

data_dir='captured_data/tetra/'
q1_exe=np.loadtxt(data_dir+'q1_exe.csv',delimiter=',')
scans=np.loadtxt(data_dir+'scans.csv',delimiter=',').reshape(len(q1_exe),-1,3)

filtered_scans=[]
###FILTER OUT INTENSITIES FIRST
for i in range(len(scans)):
    filtered_scans.append(scans[i][scans[i][:, 0] > 1])
    
# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', label='Line Profile')
edge_points_plot, = ax.plot([], [], 'ro', label='Edges')
ax.legend()
ax.set_xlim(-100, 100)
ax.set_ylim(0, 100)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Edge Detection in Line Profile')

# Update function for animation
def update(i):
    points = filtered_scans[i][:, 1:]

    edges=edge_detection(points)

    # Update the line and edge points in the plot
    line.set_data(points[:, 0], points[:, 1])
    edge_points_plot.set_data(points[edges][:, 0], points[edges][:, 1])

    return line, edge_points_plot

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(filtered_scans), interval=10, blit=True)

# Show the animation
plt.show()