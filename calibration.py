from robotics_utils import *
import numpy as np
from motoman_def import *
import open3d as o3d


def edge_detection(points):
	filter=np.array([-1,-1,-1,-1,-1,1,1,1,1,1])
	conv=np.vstack((np.convolve(points[:,0],filter,mode='same'),np.convolve(points[:,1],filter,mode='same'))).T
	conv=np.linalg.norm(conv,axis=1)
	edge_indices = np.where(conv > 30)[0]
	
	#for continuous edge_indices, filter the one with largest z
	last_valid_edge = 0

	# filter edge_indices to get the highest point
	for i in range(1, len(edge_indices)):
		if edge_indices[i] - edge_indices[last_valid_edge] <= 5:
			if points[edge_indices[i], 1] < points[edge_indices[last_valid_edge], 1]:
				edge_indices[i] = 0
			else:
				edge_indices[last_valid_edge] = 0
				last_valid_edge = i
		else:
			last_valid_edge = i
	#filter the edeges near boundaries
	edge_indices=edge_indices[(edge_indices>len(filter))&(edge_indices<len(points)-len(filter))]

	return edge_indices

def main():
	#####################################RR Robot#####################################
	config_dir='../../Welding_Motoman/config/'
	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
	robot_no_tool=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
	data_dir='captured_data/triangle/'
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
	# vis = o3d.visualization.Visualizer()
	# vis.create_window()

	# # Create an empty point cloud
	# pcd = o3d.geometry.PointCloud()

	# # Loop through the data and add points to the point cloud
	# points = []
	# for i in range(len(filtered_scans)):
	# 	for point in filtered_scans[i]:
	# 		if point[0]>50:	#filter out intensities
	# 			points.append([i, point[1], point[2]])
	# num_points_total=len(points)
	# #make edges green
	# for i in range(len(filtered_scans)):
	# 	if i in scan_idx_w_edges:
	# 		for edge_idx in edges_all[scan_idx_w_edges.index(i)]:
	# 			points.append([i, filtered_scans[i][edge_idx][1], filtered_scans[i][edge_idx][2]+10])
	# #make p1, p2, p3 red
	# points.append([p1_cam_time_idx, p1_cam[0], p1_cam[1]+20])
	# points.append([p2_cam_time_idx, p2_cam[0], p2_cam[1]+20])
	# points.append([p3_cam_time_idx, p3_cam[0], p3_cam[1]+20])
	
	# points_color=np.zeros((len(points),3))
	# points_color[:num_points_total]=[0,0,1]
	# points_color[num_points_total:-3]=[0,1,0]
	# points_color[-3:]=[1,0,0]

	# # Convert the points to a numpy array and set it to the point cloud
	# points = np.array(points)
	# pcd.points = o3d.utility.Vector3dVector(points)
	# pcd.colors = o3d.utility.Vector3dVector(points_color)
	# # Add the point cloud to the visualizer
	# vis.add_geometry(pcd)

	# # Run the visualizer
	# vis.run()
	# vis.destroy_window()

	#############################################################################CALIBRATION######################################################
	###USE TEACHPENDANT TO JOG TO 3 CORNERS, RECORD THE ABSOLUTE JOINT ANGLES READING
	signs=np.array([1,-1,1,-1,1,-1])
	p1_global=robot.fwd(np.radians([-19.9920,-17.9978,-19.3641,-33.3446,-20.8398,103.7901])*signs).p
	p2_global=robot.fwd(np.radians([-22.0563,-17.4322,-20.0974,-31.139,-19.8257,100.0584])*signs).p
	p3_global=robot.fwd(np.radians([-20.6861,-14.8484,-23.3194,-33.5871,-19.9471,103.59])*signs).p

	###interpolate to get the triangle edges
	p1_p3_global_all=np.linspace(p1_global,p3_global,p3_cam_time_idx-p1_cam_time_idx+1)
	p1_p2_global_all=np.linspace(p1_global,p2_global,p2_cam_time_idx-p1_cam_time_idx+1)
	p2_p3_global_all=np.linspace(p2_global,p3_global,p3_cam_time_idx-p2_cam_time_idx+1)


	################################################3 CORNER ONLY########################################

	# pose1=robot_no_tool.fwd(q1_exe[p1_cam_time_idx])
	# pose2=robot_no_tool.fwd(q1_exe[p2_cam_time_idx])
	# pose3=robot_no_tool.fwd(q1_exe[p3_cam_time_idx])

	# p1_eef=pose1.inv().R@p1_global+pose1.inv().p
	# p2_eef=pose2.inv().R@p2_global+pose2.inv().p
	# p3_eef=pose3.inv().R@p3_global+pose3.inv().p


	# R,t=rigid_transform_3D(np.array([p1_cam,p2_cam,p3_cam]),np.array([p1_eef,p2_eef,p3_eef]))
	# H=H_from_RT(R,t)
	# print('H:',H)


	################################################ALL TRIANGLE EDGES########################################
	p_eef_all=[]
	p_cam_all=[]
	for i in range(p1_cam_time_idx,p3_cam_time_idx+1):
		pose_cur=robot_no_tool.fwd(q1_exe[i])

		###p1-p3 right side edges always present
		p_eef_all.append(pose_cur.inv().R@p1_p3_global_all[i-p1_cam_time_idx]+pose_cur.inv().p)
		p_cam_all.append([0,filtered_scans[i][edges_all[i-p1_cam_time_idx][-1]][1],filtered_scans[i][edges_all[i-p1_cam_time_idx][-1]][2]])

		#if multiple edges detected
		if len(edges_all[i-p1_cam_time_idx])>0:
			if i<p2_cam_time_idx:
				p_eef_all.append(pose_cur.inv().R@p1_p2_global_all[i-p1_cam_time_idx]+pose_cur.inv().p)
			else:
				p_eef_all.append(pose_cur.inv().R@p2_p3_global_all[i-p2_cam_time_idx]+pose_cur.inv().p)

			p_cam_all.append([0,filtered_scans[i][edges_all[i-p1_cam_time_idx][0]][1],filtered_scans[i][edges_all[i-p1_cam_time_idx][0]][2]])
	p_eef_all=np.array(p_eef_all)
	p_cam_all=np.array(p_cam_all)


	R,t=rigid_transform_3D(p_cam_all,p_eef_all)
	H=H_from_RT(R,t)
	print('H:',H)
	#evaluate
	p_cam_all_transformed=(R@p_cam_all.T+t.reshape(-1,1)).T
	print('mean error:',np.mean(np.linalg.norm(p_cam_all_transformed-p_eef_all,axis=1)))

	####################################Compare with SERVOROBOT CALIBRAITON####################################
	H_gt = np.array([
    [-0.28078767, -0.92708288, 0.24834577, -78.12],
    [-0.70000236, 0.02078931, -0.71383787, 44.956],
    [0.65662393, -0.37427949, -0.65479758, 555.028],
    [0, 0, 0, 1]])
	#evaluate
	p_cam_all_transformed_gt=(H_gt[:3,:3]@p_cam_all.T+H_gt[:3,3].reshape(-1,1)).T
	print('mean error gt:',np.mean(np.linalg.norm(p_cam_all_transformed_gt-p_eef_all,axis=1)))

	#visualize in matplotlib
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(p_cam_all_transformed[:,0],p_cam_all_transformed[:,1],p_cam_all_transformed[:,2],c='r')
	ax.scatter(p_cam_all_transformed_gt[:,0],p_cam_all_transformed_gt[:,1],p_cam_all_transformed_gt[:,2],c='g')
	ax.scatter(p_eef_all[:,0],p_eef_all[:,1],p_eef_all[:,2],c='b')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.legend(['Transformed','Transformed_SR','EEF'])
	set_axes_equal(ax)
	plt.show()
	


if __name__ == '__main__':
	main()