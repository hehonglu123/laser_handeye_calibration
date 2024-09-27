from robotics_utils import *
import numpy as np
from motoman_def import *
import open3d as o3d
from calibration import *


def main():
	#####################################RR Robot#####################################
	config_dir='../../Welding_Motoman/config/'
	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
	robot_no_tool=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
	data_dir='captured_data/tetra/'
	q1_exe=np.loadtxt(data_dir+'q1_exe.csv',delimiter=',')
	scans=np.loadtxt(data_dir+'scans.csv',delimiter=',').reshape(len(q1_exe),-1,3)

	filtered_scans=[]
	###FILTER OUT INTENSITIES FIRST
	for i in range(len(scans)):
		filtered_scans.append(scans[i][scans[i][:, 0] > 50])

	edges_all=[]
	scan_idx_w_edges=[]
	edge_distances=[]

	for i in range(len(filtered_scans)):
		edge_indices=edge_detection(filtered_scans[i][:,1:])
		if len(edge_indices)>0:
			scan_idx_w_edges.append(i)
			edges_all.append(edge_indices)

			if len(edge_indices)>1:
				edge_distances.append(np.linalg.norm(filtered_scans[i][edge_indices[0]][1:]-filtered_scans[i][edge_indices[-1]][1:]))

			else:
				edge_distances.append(0)
	
	###Extract the corners
	p1_cam=filtered_scans[scan_idx_w_edges[0]][edges_all[0][0]][1:]
	p3_cam=filtered_scans[scan_idx_w_edges[-1]][edges_all[-1][-1]][1:]
	p1_cam_time_idx=scan_idx_w_edges[0]
	p3_cam_time_idx=scan_idx_w_edges[-1]
	p2_cam_time_idx=scan_idx_w_edges[np.argmax(edge_distances)]
	p2_cam=filtered_scans[p2_cam_time_idx][edges_all[np.argmax(edge_distances)][0]][1:]
	
	
	plt.plot(edge_distances)
	plt.xlabel('Scan Index')
	plt.ylabel('Edge Distance')
	plt.show()
	

	###append X=0 to the points in the front
	p1_cam=np.append(0,p1_cam)
	p2_cam=np.append(0,p2_cam)
	p3_cam=np.append(0,p3_cam)

	print('p1_cam:',p1_cam)
	print('p2_cam:',p2_cam)
	print('p3_cam:',p3_cam)

	###############################################visualize in 3d with time###############################################
	vis = o3d.visualization.Visualizer()
	vis.create_window()

	# Create an empty point cloud
	pcd = o3d.geometry.PointCloud()

	# Loop through the data and add points to the point cloud
	points = []
	for i in range(len(filtered_scans)):
		for point in filtered_scans[i]:
			if point[0]>50:	#filter out intensities
				points.append([i, point[1], point[2]])
	num_points_total=len(points)
	#make edges green
	for i in range(len(filtered_scans)):
		if i in scan_idx_w_edges:
			for edge_idx in edges_all[scan_idx_w_edges.index(i)]:
				points.append([i, filtered_scans[i][edge_idx][1], filtered_scans[i][edge_idx][2]+10])
	#make p1, p2, p3 red
	points.append([p1_cam_time_idx, p1_cam[0], p1_cam[1]+20])
	points.append([p2_cam_time_idx, p2_cam[0], p2_cam[1]+20])
	points.append([p3_cam_time_idx, p3_cam[0], p3_cam[1]+20])
	
	points_color=np.zeros((len(points),3))
	points_color[:num_points_total]=[0,0,1]
	points_color[num_points_total:-3]=[0,1,0]
	points_color[-3:]=[1,0,0]

	# Convert the points to a numpy array and set it to the point cloud
	points = np.array(points)
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(points_color)
	# Add the point cloud to the visualizer
	vis.add_geometry(pcd)

	# Run the visualizer
	vis.run()
	vis.destroy_window()

	#############################################################################CALIBRATION######################################################
	###USE TEACHPENDANT TO JOG TO 3 CORNERS, RECORD THE ABSOLUTE JOINT ANGLES READING
	signs=np.array([1,-1,1,-1,1,-1])
	p1_global=robot.fwd(np.radians([3.9377,-25.9357,-17.1939,-11.0083,-72.8311,79.0141])*signs).p
	p2_global=robot.fwd(np.radians([-4.8084,-23.8468,-19.9235,2.4930,-73.7432,100.4366])*signs).p
	p3_global=robot.fwd(np.radians([13.8536,17.2602,-69.9760,-90.8255,-29.3152,-161.8720])*signs).p
	p4_global=robot.fwd(np.radians([-3.6276,-22.2679,-19.6159,3.0346,-75.5856,101.4262])*signs).p

	print('p1_global:',p1_global)
	print('p2_global:',p2_global)
	print('p3_global:',p3_global)
	###interpolate to get the triangle edges
	p1_p3_global_all=np.linspace(p1_global,p3_global,p3_cam_time_idx-p1_cam_time_idx+1)
	p1_p2_global_all=np.linspace(p1_global,p2_global,p2_cam_time_idx-p1_cam_time_idx+1)
	p2_p3_global_all=np.linspace(p2_global,p3_global,p3_cam_time_idx-p2_cam_time_idx+1)


	################################################3 CORNER ONLY########################################

	pose1=robot_no_tool.fwd(q1_exe[p1_cam_time_idx])
	pose2=robot_no_tool.fwd(q1_exe[p2_cam_time_idx])
	pose3=robot_no_tool.fwd(q1_exe[p3_cam_time_idx])


	p1_eef=pose1.inv().R@p1_global+pose1.inv().p
	p2_eef=pose2.inv().R@p2_global+pose2.inv().p
	p3_eef=pose3.inv().R@p3_global+pose3.inv().p


	R,t=rigid_transform_3D(np.array([p1_cam,p2_cam,p3_cam]),np.array([p1_eef,p2_eef,p3_eef]))
	H=H_from_RT(R,t)
	H_gt = np.array([[ -2.67052498e-01, -9.32632411e-01,  2.42651496e-01, -7.90839037e+01],
					[-7.02253482e-01, 1.59028356e-02,  -7.11749357e-01, 5.05030933e+01],
					[6.59941672e-01, -3.60477302e-01, -6.59191250e-01,  5.56486397e+02],
					[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
	print('H:',H)

	###plot in 3d space
	H1=H_from_RT(pose1.R,pose1.p)
	H2=H_from_RT(pose2.R,pose2.p)
	H3=H_from_RT(pose3.R,pose3.p)
	p1_cam_global=H1@H@np.array([0,p1_cam[1],p1_cam[2],1])
	p2_cam_global=H2@H@np.array([0,p2_cam[1],p2_cam[2],1])
	p3_cam_global=H3@H@np.array([0,p3_cam[1],p3_cam[2],1])
	plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter(p1_global[0],p1_global[1],p1_global[2],c='r')
	ax.scatter(p2_global[0],p2_global[1],p2_global[2],c='r')
	ax.scatter(p3_global[0],p3_global[1],p3_global[2],c='r')
	ax.scatter(p1_cam_global[0],p1_cam_global[1],p1_cam_global[2],c='b')
	ax.scatter(p2_cam_global[0],p2_cam_global[1],p2_cam_global[2],c='b')
	ax.scatter(p3_cam_global[0],p3_cam_global[1],p3_cam_global[2],c='b')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	set_axes_equal(ax)
	plt.show()




if __name__ == '__main__':
	main()