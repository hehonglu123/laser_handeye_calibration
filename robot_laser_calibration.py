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
	sub=RRN.SubscribeService('rr+tcp://localhost:12181/?service=fujicam')
	obj = sub.GetDefaultClientWait(30)		#connect, timeout=30s
	scan_change=sub.SubscribeWire("lineProfile")

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
	R_default=np.array([[0, 1, -0.    ],
            [ 1, 0,  0.    ],
            [0.,      0.,     -1.    ]])
	q_positioner_home=np.array([-15.*np.pi/180.,np.pi/2])
	center_of_rotation=positioner.fwd(q_positioner_home,world=True).p
	measure_distance=150
	total_time=10
	num_points=int(total_time*streaming_rate)
	q2_cmd_all1=[]
	for angle in np.linspace(np.pi/5,3*np.pi/5,num_points):	#define rotation range
		v_z_global=-Rx(angle)[:,1]
		v_z=H2010_1440[:3,:3]@v_z_global ###pointing toward positioner's X with 15deg tiltd angle looking down
		v_y=VectorPlaneProjection(np.array([-1,0,0]),v_z)	###FLIR's Y pointing toward 1440's -X in 1440's base frame, projected on v_z's plane
		v_x=np.cross(v_y,v_z)
		p2_in_base_frame=center_of_rotation_in_base_frame-measure_distance*v_z			###back project measure_distance-mm away from torch
		R2=np.vstack((v_x,v_y,v_z)).T
		q2_cmd_all1.append(robot2.inv(p2_in_base_frame,R2,last_joints=np.zeros(6))[0])

	#####################################Path Gen2 Circular Rotation#####################################
	q2_cmd_all2=[]
	for angle in np.linspace(np.pi/2,np.pi/4,num_points):	#define rotation range
		v_z_global=-Ry(-angle)[:,0]
		v_z=H2010_1440[:3,:3]@v_z_global ###pointing toward positioner's X with 15deg tiltd angle looking down
		v_y=VectorPlaneProjection(np.array([-1,0,0]),v_z)	###FLIR's Y pointing toward 1440's -X in 1440's base frame, projected on v_z's plane
		v_x=np.cross(v_y,v_z)
		p2_in_base_frame=center_of_rotation_in_base_frame-measure_distance*v_z			###back project measure_distance-mm away from torch
		R2=np.vstack((v_x,v_y,v_z)).T
		q2_cmd_all2.append(robot2.inv(p2_in_base_frame,R2,last_joints=np.zeros(6))[0])


	
	##############################################MOTION#############################################################

	

	# Arrays to store object points and image points from all images
	images = []
	associated_q2=[]

	SS.jog2q(np.hstack((q1_default,q2_cmd_all1[0],q_positioner_home)))
	now=time.perf_counter()
	###CONTINUOUS CAPTURE for CALIBRATION
	for i in range(len(q2_cmd_all1)):
		SS.position_cmd(np.hstack((q1_default,q2_cmd_all1[i],q_positioner_home)),now)
		now=time.perf_counter()
		if ir_img is not None and image_updated:
			associated_q2.append(SS.q_cur[6:12])
			image_updated = False	
			images.append(ir_img)

	SS.jog2q(np.hstack((q1_default,q2_cmd_all2[0],q_positioner_home)))
	now=time.perf_counter()
	###CONTINUOUS CAPTURE for CALIBRATION
	for i in range(len(q2_cmd_all2)):
		SS.position_cmd(np.hstack((q1_default,q2_cmd_all2[i],q_positioner_home)),now)
		now=time.perf_counter()
		if ir_img is not None and image_updated:
			associated_q2.append(SS.q_cur[6:12])
			image_updated = False
			images.append(ir_img)
			



if __name__ == "__main__":
	main()