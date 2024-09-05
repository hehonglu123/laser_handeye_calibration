import cv2, time, os
import numpy as np
from RobotRaconteur.Client import *
from motoman_def import *
from StreamingSend import *

from calibration import *

def connect_failed(s, client_id, url, err):
    print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))

# Create a directory to save the images
os.makedirs('captured_data', exist_ok=True)


def main():

	#####################################RR Fujicam#####################################
	sub=RRN.SubscribeService('rr+tcp://localhost:12181/?service=fujicam')
	obj = sub.GetDefaultClientWait(30)		#connect, timeout=30s
	scan_wire=sub.SubscribeWire("lineProfile")
	sub.ClientConnectFailed += connect_failed

	
	#####################################RR Robot#####################################
	config_dir='../../Welding_Motoman/config/'
	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
	robot_no_tool=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
	positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_extended_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose_mocap.csv')
	RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
	streaming_rate=125.
	SS=StreamingSend(RR_robot_sub,streaming_rate=streaming_rate)

	#####################################Path Gen1 Translation#####################################
	q2_default=np.zeros(6)
	q2_default[0]=np.pi/2
	R_default=np.array([[-1, 0, -0.    ],
						[0., 1,  0.    ],
						[0., 0.,-1.    ]])
	q_positioner_home=np.array([-15.*np.pi/180.,np.pi/2])
	p_positioner_home=positioner.fwd(q_positioner_home,world=True).p
	measure_distance=150 #distance from tip with wire, as a guess only
	total_time=5
	num_points=int(total_time*streaming_rate)

	p_start=p_positioner_home+np.array([0,100,measure_distance])
	p_end=p_positioner_home+np.array([0,-100,measure_distance])
	p_all=np.linspace(p_start,p_end,num_points)
	q1_cmd_all1=robot.find_curve_js(p_all,[R_default]*num_points,np.zeros(6))

	

	

	#####################################Path Gen2 Circular Rotation#####################################
	rotation_z_offset=-150
	center_of_rotation=p_positioner_home+np.array([0,0,rotation_z_offset])
	q1_cmd_all2=[]
	for angle in np.linspace(np.pi/5,-np.pi/6,num_points):	#define rotation range
		v_z=-Rx(angle)[:,-1]
		v_x=np.array([-1,0,0])
		v_y=np.cross(v_z,v_x)
		p_in_base_frame=center_of_rotation-(measure_distance-rotation_z_offset)*v_z			###back project measure_distance-mm away from torch
		R_in_base_frame=np.vstack((v_x,v_y,v_z)).T
		q1_cmd_all2.append(robot.inv(p_in_base_frame,R_in_base_frame,last_joints=np.zeros(6))[0])


	
	##############################################MOTION#############################################################
	scans = []
	q1_exe=[]

	SS.jog2q(np.hstack((q1_cmd_all1[0],q2_default,q_positioner_home)))
	now=time.perf_counter()
	###CONTINUOUS CAPTURE for CALIBRATION
	for i in range(len(q1_cmd_all1)):
		SS.position_cmd(np.hstack((q1_cmd_all1[i],q2_default,q_positioner_home)),now)
		now=time.perf_counter()
		q1_exe.append(SS.q_cur[0:6])
		scan_data=np.vstack((scan_wire.InValue.I_data,scan_wire.InValue.Y_data,scan_wire.InValue.Z_data)).T
		scans.append(scan_data.flatten())

	print("FIRST MOTION INDEX: ",len(q1_exe))

	SS.jog2q(np.hstack((q1_cmd_all2[0],q2_default,q_positioner_home)))
	now=time.perf_counter()
	###CONTINUOUS CAPTURE for CALIBRATION
	for i in range(len(q1_cmd_all2)):
		SS.position_cmd(np.hstack((q1_cmd_all2[i],q2_default,q_positioner_home)),now)
		now=time.perf_counter()
		q1_exe.append(SS.q_cur[0:6])
		scan_data=np.vstack((scan_wire.InValue.I_data,scan_wire.InValue.Y_data,scan_wire.InValue.Z_data)).T
		scans.append(scan_data.flatten())
			

	### Save the captured data
	recorded_dir = 'captured_data/SR_block/'
	os.makedirs(recorded_dir, exist_ok=True)
	np.savetxt(recorded_dir+'scans.csv', np.array(scans), delimiter=',')
	np.savetxt(recorded_dir+'q1_exe.csv', np.array(q1_exe), delimiter=',')


if __name__ == "__main__":
	main()