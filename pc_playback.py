import copy, time
import numpy as np
from motoman_def import *
import open3d as o3d


def main():
	#####################################RR Robot#####################################
	config_dir='../../Welding_Motoman/config/'
	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
	robot_no_tool=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
	robot_guess=copy.deepcopy(robot_no_tool)
	T_guess=np.array([-40,-150,400])
	R_guess=np.array([[0,-0.92718385,-0.37460659],[1,0,0],[0,-0.37460659,0.92718385]])	#from torch
	R_guess=R_guess@Ry(np.pi/12)
	robot_guess.robot.R_tool=R_guess
	robot_guess.robot.p_tool=T_guess

	#read data
	data_dir='captured_data/wall/'
	q1_exe=np.loadtxt(data_dir+'q1_exe.csv',delimiter=',')
	scans=np.loadtxt(data_dir+'scans.csv',delimiter=',').reshape(len(q1_exe),-1,3)
	q1_exe=q1_exe[:625]
	scans=scans[:625]
	
	# Create a visualizer window
	vis = o3d.visualization.Visualizer()
	vis.create_window()

	# Create an initial point cloud
	pcd = o3d.geometry.PointCloud()
	scan_cur=scans[0]
	pc_scan_cur=[]
	#filter out intensity below 50%
	indices=np.where(scan_cur[:,0]>1)[0]
	points_in_scan_frame=np.hstack((np.zeros((len(indices),1)),scan_cur[indices,1:]))
	pc_scan_cur=(robot_guess.fwd(q1_exe[0]).R @ points_in_scan_frame.T + robot_guess.fwd(q1_exe[0]).p.reshape(3,1)).T
	
	print(pc_scan_cur.shape)

	pcd.points = o3d.utility.Vector3dVector(pc_scan_cur)
	vis.add_geometry(pcd)

	for i in range(len(scans)):
		scan_cur=scans[i]
		pc_scan_cur=[]
		#filter out intensity below 50%
		indices=np.where(scan_cur[:,0]>1)[0]

		points_in_scan_frame=np.hstack((np.zeros((len(indices),1)),scan_cur[indices,1:]))
		pc_scan_cur=(robot_guess.fwd(q1_exe[i]).R @ points_in_scan_frame.T + robot_guess.fwd(q1_exe[i]).p.reshape(3,1)).T
		

		all_points = np.vstack((np.asarray(pcd.points), pc_scan_cur))
		pcd.points = o3d.utility.Vector3dVector(all_points)

		# Update the visualizer
		vis.update_geometry(pcd)
		vis.poll_events()
		vis.update_renderer()
		time.sleep(0.1)  # Sleep for a short duration to simulate real-time update
	
	
	
	

if __name__ == '__main__':
	main()