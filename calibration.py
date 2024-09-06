import cv2, os, copy
import numpy as np
from motoman_def import *
import open3d as o3d

def evaluate_hand_eye_calibration(R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base, rvecs, tvecs):
	H_cam2gripper = H_from_RT(R_cam2gripper, t_cam2gripper)

	R_target2base = []
	rvec_target2base = []
	t_target2base = []
	for i in range(len(t_gripper2base)):
		H_gripper2base=H_from_RT(R_gripper2base[i], t_gripper2base[i])
		H_target2cam=H_from_RT(cv2.Rodrigues(rvecs[i])[0], tvecs[i])
		H_target2base=H_gripper2base @ H_cam2gripper @ H_target2cam

		R_target2base.append(H_target2base[:3,:3])
		rvec_target2base.append(cv2.Rodrigues(H_target2base[:3,:3])[0])
		t_target2base.append(H_target2base[:3,3])
	
	#3d scatter t_target2base
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(*zip(*t_target2base))
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	set_axes_equal(ax)
	plt.show()


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
	pc=[]
	for i in range(len(scans)):
		scan_cur=scans[i]
		pc_scan_cur=[]
		#filter out intensity below 50%
		indices=np.where(scan_cur[:,0]>1)[0]
		for j in indices:
			pc_scan_cur.append(robot_guess.fwd(q1_exe[i]).R @ np.array([0,scan_cur[j][1],scan_cur[j][2]])+robot_guess.fwd(q1_exe[i]).p)
			
		pc.extend(pc_scan_cur)
	
	#3d scatter pc in o3d
	vis = o3d.visualization.Visualizer()
	vis.create_window()
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pc)
	vis.add_geometry(pcd)
	vis.run()
	vis.destroy_window()

	

if __name__ == '__main__':
	main()