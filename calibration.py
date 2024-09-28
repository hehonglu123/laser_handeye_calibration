from robotics_utils import *
import numpy as np
from motoman_def import *
import open3d as o3d

def rigid_transform_3D_planar(A, B):
	assert len(A) == len(B), "Point sets A and B must have the same number of points"
	assert A.shape[1] == B.shape[1], "Point sets A and B must have the same dimensionality"

	# Step 1: Compute centroids
	centroid_A = np.mean(A, axis=0)
	centroid_B = np.mean(B, axis=0)

	# Step 2: Center the points
	AA = (A - centroid_A).T[1:]
	BB = (B - centroid_B).T

	R_partial=BB@np.linalg.pinv(AA)

	R=np.eye(3)
	R[:,0]=np.cross(R_partial[:,0],R_partial[:,1])
	R[:,1]=R_partial[:,0]
	R[:,2]=R_partial[:,1]
	#normalize each axis
	R[:,0]=R[:,0]/np.linalg.norm(R[:,0])
	R[:,1]=R[:,1]/np.linalg.norm(R[:,1])
	R[:,2]=R[:,2]/np.linalg.norm(R[:,2])


	# Step 7: Compute translation vector t
	t = centroid_B - R @ centroid_A

	return R, t


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

	start_detection_idx=500
	for i in range(start_detection_idx,len(filtered_scans)):
		edge_indices=edge_detection(filtered_scans[i][:,1:])
		if len(edge_indices)>0:
			scan_idx_w_edges.append(i)
			edges_all.append(edge_indices)

			if len(edge_indices)>1:
				edge_distances.append(np.linalg.norm(filtered_scans[i][edge_indices[0]][1:]-filtered_scans[i][edge_indices[-1]][1:]))
			else:
				edge_distances.append(0)
				if i>2*len(filtered_scans)//3:
					break
	
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
	points.append([p1_cam_time_idx, p1_cam[1], p1_cam[2]+20])
	points.append([p2_cam_time_idx, p2_cam[1], p2_cam[2]+20])
	points.append([p3_cam_time_idx, p3_cam[1], p3_cam[2]+20])
	
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

	p1_global=robot.fwd(np.array([-0.47997916,  0.47394517, -0.06122775, -0.27545108, -0.71117267,  0.6653875])).p
	p2_global=robot.fwd(np.array([-0.51271596,  0.47143835, -0.06415327, -0.29002304, -0.71870424,  0.70679911])).p
	p3_global=robot.fwd(np.array([-0.4953717,   0.41889569, -0.13339059, -0.2882826,  -0.69849541,  0.69232999])).p

	print('p1_global:',p1_global)
	print('p2_global:',p2_global)
	print('p3_global:',p3_global)
	###interpolate to get the triangle edges
	p1_p3_global_all=np.linspace(p1_global,p3_global,p3_cam_time_idx-p1_cam_time_idx+1)
	p1_p2_global_all=np.linspace(p1_global,p2_global,p2_cam_time_idx-p1_cam_time_idx+1)
	p2_p3_global_all=np.linspace(p2_global,p3_global,p3_cam_time_idx-p2_cam_time_idx+1)


	################################################3 CORNER ONLY########################################

	# pose1=robot_no_tool.fwd(q1_exe[p1_cam_time_idx])
	# pose2=robot_no_tool.fwd(q1_exe[p2_cam_time_idx])
	# pose3=robot_no_tool.fwd(q1_exe[p3_cam_time_idx])

	# H1=H_from_RT(pose1.R,pose1.p)
	# H2=H_from_RT(pose2.R,pose2.p)
	# H3=H_from_RT(pose3.R,pose3.p)


	# p1_eef=(H_inv(H1)@np.array([p1_global[0],p1_global[1],p1_global[2],1]))[:3]
	# p2_eef=(H_inv(H2)@np.array([p2_global[0],p2_global[1],p2_global[2],1]))[:3]
	# p3_eef=(H_inv(H3)@np.array([p3_global[0],p3_global[1],p3_global[2],1]))[:3]


	# R,t=rigid_transform_3D(np.array([p1_cam,p2_cam,p3_cam]),np.array([p1_eef,p2_eef,p3_eef]))
	# H=H_from_RT(R,t)
	# H_gt = np.array([[-0.28078767, -0.92708288,  0.24834577, -78.12],
	# 				[-0.70000236,  0.02078931, -0.71383787, 44.956],
	# 				[0.65662393, -0.37427949, -0.65479758, 555.028],
	# 				[0,0,0,1]])
	# print('H:',H)

	# #visualize in matplotlib in EEF Frame
	# p1_cam_transformed=(H[:3,:3]@p1_cam.reshape(-1,1)+H[:3,3].reshape(-1,1))
	# p2_cam_transformed=(H[:3,:3]@p2_cam.reshape(-1,1)+H[:3,3].reshape(-1,1))
	# p3_cam_transformed=(H[:3,:3]@p3_cam.reshape(-1,1)+H[:3,3].reshape(-1,1))

	# p1_cam_transformed_gt=(H_gt[:3,:3]@p1_cam.reshape(-1,1)+H_gt[:3,3].reshape(-1,1))
	# p2_cam_transformed_gt=(H_gt[:3,:3]@p2_cam.reshape(-1,1)+H_gt[:3,3].reshape(-1,1))
	# p3_cam_transformed_gt=(H_gt[:3,:3]@p3_cam.reshape(-1,1)+H_gt[:3,3].reshape(-1,1))
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')

	# ax.scatter(p1_cam_transformed[0],p1_cam_transformed[1],p1_cam_transformed[2],c='r')
	# ax.scatter(p2_cam_transformed[0],p2_cam_transformed[1],p2_cam_transformed[2],c='g')
	# ax.scatter(p3_cam_transformed[0],p3_cam_transformed[1],p3_cam_transformed[2],c='b')
	# ax.scatter(p1_eef[0],p1_eef[1],p1_eef[2],c='r')
	# ax.scatter(p2_eef[0],p2_eef[1],p2_eef[2],c='g')
	# ax.scatter(p3_eef[0],p3_eef[1],p3_eef[2],c='b')
	# ax.scatter(p1_cam_transformed_gt[0],p1_cam_transformed_gt[1],p1_cam_transformed_gt[2],c='r')
	# ax.scatter(p2_cam_transformed_gt[0],p2_cam_transformed_gt[1],p2_cam_transformed_gt[2],c='g')
	# ax.scatter(p3_cam_transformed_gt[0],p3_cam_transformed_gt[1],p3_cam_transformed_gt[2],c='b')
	# ax.set_xlabel('X Label')
	# ax.set_ylabel('Y Label')
	# ax.set_zlabel('Z Label')
	# # ax.legend(['Transformed','Transformed_SR','EEF'])
	# plt.title('EEF Frame')
	# set_axes_equal(ax)
	# plt.show()

	# ###plot in 3d space
	# p1_cam_global=H1@H@np.array([0,p1_cam[1],p1_cam[2],1])
	# p2_cam_global=H2@H@np.array([0,p2_cam[1],p2_cam[2],1])
	# p3_cam_global=H3@H@np.array([0,p3_cam[1],p3_cam[2],1])
	# plt.figure()
	# ax = plt.axes(projection='3d')
	# ax.scatter(p1_global[0],p1_global[1],p1_global[2],c='r')
	# ax.scatter(p2_global[0],p2_global[1],p2_global[2],c='r')
	# ax.scatter(p3_global[0],p3_global[1],p3_global[2],c='r')
	# ax.scatter(p1_cam_global[0],p1_cam_global[1],p1_cam_global[2],c='b')
	# ax.scatter(p2_cam_global[0],p2_cam_global[1],p2_cam_global[2],c='b')
	# ax.scatter(p3_cam_global[0],p3_cam_global[1],p3_cam_global[2],c='b')
	# ax.set_xlabel('X Label')
	# ax.set_ylabel('Y Label')
	# ax.set_zlabel('Z Label')
	# plt.title('Global Frame')
	# set_axes_equal(ax)
	# plt.show()




	################################################ALL TRIANGLE EDGES########################################
	p1_p3_eef_all=[]
	p1_p2_eef_all=[]
	p2_p3_eef_all=[]
	p1_p3_cam_all=[]
	p1_p2_cam_all=[]
	p2_p3_cam_all=[]
	for i in range(p1_cam_time_idx,p3_cam_time_idx+1):
		pose_cur=robot_no_tool.fwd(q1_exe[i])
		H_cur_inv=H_inv(H_from_RT(pose_cur.R,pose_cur.p))

		###p1-p3 right side edges always present
		p1_p3_eef_all.append((H_cur_inv@np.array([p1_p3_global_all[i-p1_cam_time_idx][0],p1_p3_global_all[i-p1_cam_time_idx][1],p1_p3_global_all[i-p1_cam_time_idx][2],1]))[:3])
		p1_p3_cam_all.append([0,filtered_scans[i][edges_all[i-p1_cam_time_idx][-1]][1],filtered_scans[i][edges_all[i-p1_cam_time_idx][-1]][2]])

		#if multiple edges detected
		if len(edges_all[i-p1_cam_time_idx])>0:
			if i<p2_cam_time_idx:
				p1_p2_eef_all.append((H_cur_inv@np.array([p1_p2_global_all[i-p1_cam_time_idx][0],p1_p2_global_all[i-p1_cam_time_idx][1],p1_p2_global_all[i-p1_cam_time_idx][2],1]))[:3])
				p1_p2_cam_all.append([0,filtered_scans[i][edges_all[i-p1_cam_time_idx][0]][1],filtered_scans[i][edges_all[i-p1_cam_time_idx][0]][2]])
			else:
				p2_p3_eef_all.append((H_cur_inv@np.array([p2_p3_global_all[i-p2_cam_time_idx][0],p2_p3_global_all[i-p2_cam_time_idx][1],p2_p3_global_all[i-p2_cam_time_idx][2],1]))[:3])
				p2_p3_cam_all.append([0,filtered_scans[i][edges_all[i-p1_cam_time_idx][0]][1],filtered_scans[i][edges_all[i-p1_cam_time_idx][0]][2]])

	p1_p3_eef_all=np.array(p1_p3_eef_all)
	p1_p2_eef_all=np.array(p1_p2_eef_all)
	p2_p3_eef_all=np.array(p2_p3_eef_all)
	p1_p3_cam_all=np.array(p1_p3_cam_all)
	p1_p2_cam_all=np.array(p1_p2_cam_all)
	p2_p3_cam_all=np.array(p2_p3_cam_all)
	p_eef_all=np.vstack((p1_p3_eef_all,p1_p2_eef_all,p2_p3_eef_all))
	p_cam_all=np.vstack((p1_p3_cam_all,p1_p2_cam_all,p2_p3_cam_all))



	R,t=rigid_transform_3D_planar(p_cam_all,p_eef_all)
	H=H_from_RT(R,t)
	print('H:',H)
	#evaluate
	p_cam_all_transformed=(R@p_cam_all.T+t.reshape(-1,1)).T
	p1_p3_cam_all_transformed=(R@p1_p3_cam_all.T+t.reshape(-1,1)).T
	p1_p2_cam_all_transformed=(R@p1_p2_cam_all.T+t.reshape(-1,1)).T
	p2_p3_cam_all_transformed=(R@p2_p3_cam_all.T+t.reshape(-1,1)).T
	print('mean error:',np.mean(np.linalg.norm(p_cam_all_transformed-p_eef_all,axis=1)))

	####################################Compare with SERVOROBOT CALIBRAITON####################################

	H_gt = np.array([[-0.28078767, -0.92708288,  0.24834577, -78.12],
					[-0.70000236,  0.02078931, -0.71383787, 44.956],
					[0.65662393, -0.37427949, -0.65479758, 555.028],
					[0,0,0,1]])
	
	###evaluate
	p_cam_all_transformed_gt=(H_gt[:3,:3]@p_cam_all.T+H_gt[:3,3].reshape(-1,1)).T
	print('mean error gt:',np.mean(np.linalg.norm(p_cam_all_transformed_gt-p_eef_all,axis=1)))
	p1_p3_cam_all_transformed_gt=(H_gt[:3,:3]@p1_p3_cam_all.T+H_gt[:3,3].reshape(-1,1)).T
	p1_p2_cam_all_transformed_gt=(H_gt[:3,:3]@p1_p2_cam_all.T+H_gt[:3,3].reshape(-1,1)).T
	p2_p3_cam_all_transformed_gt=(H_gt[:3,:3]@p2_p3_cam_all.T+H_gt[:3,3].reshape(-1,1)).T

	#visualize in matplotlib
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# ax.scatter(p_cam_all_transformed[:,0],p_cam_all_transformed[:,1],p_cam_all_transformed[:,2],c='r')
	# ax.scatter(p_cam_all_transformed_gt[:,0],p_cam_all_transformed_gt[:,1],p_cam_all_transformed_gt[:,2],c='g')
	# ax.scatter(p_eef_all[:,0],p_eef_all[:,1],p_eef_all[:,2],c='b')
	ax.scatter(p1_p3_cam_all_transformed[:,0],p1_p3_cam_all_transformed[:,1],p1_p3_cam_all_transformed[:,2],c='r')
	ax.scatter(p1_p2_cam_all_transformed[:,0],p1_p2_cam_all_transformed[:,1],p1_p2_cam_all_transformed[:,2],c='g')
	ax.scatter(p2_p3_cam_all_transformed[:,0],p2_p3_cam_all_transformed[:,1],p2_p3_cam_all_transformed[:,2],c='b')
	ax.scatter(p1_p3_eef_all[:,0],p1_p3_eef_all[:,1],p1_p3_eef_all[:,2],c='r')
	ax.scatter(p1_p2_eef_all[:,0],p1_p2_eef_all[:,1],p1_p2_eef_all[:,2],c='g')
	ax.scatter(p2_p3_eef_all[:,0],p2_p3_eef_all[:,1],p2_p3_eef_all[:,2],c='b')
	# ax.scatter(p1_p3_cam_all_transformed_gt[:,0],p1_p3_cam_all_transformed_gt[:,1],p1_p3_cam_all_transformed_gt[:,2],c='r')
	# ax.scatter(p1_p2_cam_all_transformed_gt[:,0],p1_p2_cam_all_transformed_gt[:,1],p1_p2_cam_all_transformed_gt[:,2],c='g')
	# ax.scatter(p2_p3_cam_all_transformed_gt[:,0],p2_p3_cam_all_transformed_gt[:,1],p2_p3_cam_all_transformed_gt[:,2],c='b')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	# ax.legend(['Transformed','Transformed_SR','EEF'])
	set_axes_equal(ax)
	plt.show()


	####################################Compare in global frame####################################
	p1_p3_global_all=[]
	p1_p2_global_all=[]
	p2_p3_global_all=[]
	for i in range(p1_cam_time_idx,p3_cam_time_idx+1):
		pose_cur=robot_no_tool.fwd(q1_exe[i])
		H_cur=H_from_RT(pose_cur.R,pose_cur.p)
		
		###p1-p3 right side edges always present
		p1_p3_global_all.append((H_cur@H@np.array([0,filtered_scans[i][edges_all[i-p1_cam_time_idx][-1]][1],filtered_scans[i][edges_all[i-p1_cam_time_idx][-1]][2],1]))[:3])
		###p1-p2
		if len(edges_all[i-p1_cam_time_idx])>0:
			if i<p2_cam_time_idx:
				p1_p2_global_all.append((H_cur@H@np.array([0,filtered_scans[i][edges_all[i-p1_cam_time_idx][0]][1],filtered_scans[i][edges_all[i-p1_cam_time_idx][0]][2],1]))[:3])
			else:
				p2_p3_global_all.append((H_cur@H@np.array([0,filtered_scans[i][edges_all[i-p1_cam_time_idx][0]][1],filtered_scans[i][edges_all[i-p1_cam_time_idx][0]][2],1]))[:3])

	p1_p3_global_all=np.array(p1_p3_global_all)
	p1_p2_global_all=np.array(p1_p2_global_all)
	p2_p3_global_all=np.array(p2_p3_global_all)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(p1_p3_global_all[:,0],p1_p3_global_all[:,1],p1_p3_global_all[:,2],c='r')
	ax.scatter(p1_p2_global_all[:,0],p1_p2_global_all[:,1],p1_p2_global_all[:,2],c='g')
	ax.scatter(p2_p3_global_all[:,0],p2_p3_global_all[:,1],p2_p3_global_all[:,2],c='b')
	ax.scatter(p1_global[0],p1_global[1],p1_global[2],c='r')
	ax.scatter(p2_global[0],p2_global[1],p2_global[2],c='g')
	ax.scatter(p3_global[0],p3_global[1],p3_global[2],c='b')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	set_axes_equal(ax)
	plt.show()

if __name__ == '__main__':
	main()