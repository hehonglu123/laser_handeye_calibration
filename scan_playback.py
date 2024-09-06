import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

data_dir='captured_data/wall/'
q1_exe=np.loadtxt(data_dir+'q1_exe.csv',delimiter=',')
scans=np.loadtxt(data_dir+'scans.csv',delimiter=',').reshape(len(q1_exe),-1,3)

# Initialize the plot
fig, ax = plt.subplots()
scatter = ax.scatter([], [])

# Set axis limits (adjust as needed)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# Define the update function
def update(frame):
    print(frame)
    ax.clear()
    ax.set_xlim(-100, 100)
    ax.set_ylim(0, 100)
    scatter = ax.scatter(scans[frame][:, 1], scans[frame][:, 2])
    return scatter,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(scans), interval=20, blit=True)

# Display the animation
plt.show()