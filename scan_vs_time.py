import numpy as np 
import open3d as o3d

data_dir='captured_data/triangle/'
q1_exe=np.loadtxt(data_dir+'q1_exe.csv',delimiter=',')
scans=np.loadtxt(data_dir+'scans.csv',delimiter=',').reshape(len(q1_exe),-1,3)

streaming_rate=125

vis = o3d.visualization.Visualizer()
vis.create_window()

# Create an empty point cloud
pcd = o3d.geometry.PointCloud()

# Loop through the data and add points to the point cloud
points = []
for i in range(len(q1_exe)):

    for point in scans[i]:
        if point[0]>50:
            points.append([i, point[1], point[2]])

# Convert the points to a numpy array and set it to the point cloud
points = np.array(points)
pcd.points = o3d.utility.Vector3dVector(points)

# Add the point cloud to the visualizer
vis.add_geometry(pcd)

# Run the visualizer
vis.run()
vis.destroy_window()