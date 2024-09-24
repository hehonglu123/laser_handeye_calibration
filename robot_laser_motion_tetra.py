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
	obj = sub.GetDefaultClientWait(2)		#connect, timeout=30s
	scan_wire=sub.SubscribeWire("lineProfile")
	sub.ClientConnectFailed += connect_failed

	
	#####################################RR Robot#####################################
	config_dir='../../Welding_Motoman/config/'
	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
	robot_no_tool=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
	streaming_rate=125.
	RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
	SS=StreamingSend(RR_robot_sub,streaming_rate=streaming_rate)

	q2_default=np.zeros(6)
	q2_default[0]=np.pi/2
	q_positioner_home=np.array([-15.*np.pi/180.,np.pi/2])

	########################################TAUGHT POINT FROM TEACHPENDANT##################################
	###USE TEACHPENDANT TO JOG TO 3 CORNERS, RECORD THE ABSOLUTE JOINT ANGLES READING
	signs=np.array([1,-1,1,-1,1,-1])
	q1=np.radians([3.9377,-25.9357,-17.1939,-11.0083,-72.8311,79.0141])*signs
	q2=np.radians([-4.8084,-23.8468,-19.9235,2.4930,-73.7432,100.4366])*signs
	q3=np.radians([13.8536,17.2602,-69.9760,-90.8255,-29.3152,-161.8720])*signs
	q4=np.radians([-3.6276,-22.2679,-19.6159,3.0346,-75.5856,101.4262])*signs
	
	p1=robot.fwd(q1).p
	p2=robot.fwd(q2).p
	p3=robot.fwd(q3).p
	p4=robot.fwd(q4).p
	#get the plane from 3 points
	normal,centroid=fit_plane(np.vstack((p1,p2,p3)))
	centroid=np.average(np.vstack((p1,p3)),axis=0)
	centroid=np.average(np.vstack((centroid,p2)),axis=0)
	length=np.average(np.array([np.linalg.norm(p1-p2),np.linalg.norm(p2-p3),np.linalg.norm(p3-p1)]))
	#make sure normal is pointing down
	if normal[2]>0:
		normal=-normal
	

	
	#####################################Path Gen1 Translation#####################################
	moving_direction=p3-p1
	moving_direction=moving_direction/np.linalg.norm(moving_direction)
	R_torch=np.vstack((np.cross(-moving_direction,normal),-moving_direction,normal)).T
	tilting_angle=np.radians(-45)
	z_offset=45
	R_torch=R_torch@Rx(tilting_angle)
	p_start=centroid-1.5*length*moving_direction+z_offset*(-normal)
	p_end=centroid+1.5*length*moving_direction+z_offset*(-normal)

	print("p_start: ",p_start)
	print("p_end: ",p_end)
	print("R_torch: ",R_torch)

	total_time=10
	num_points=int(total_time*streaming_rate)
	p_all=np.linspace(p_start,p_end,num_points)
	q1_cmd_all1=[]
	for p in p_all:
		q1_cmd_all1.append(robot.inv(p,R_torch,last_joints=np.zeros(6))[0])


	
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


	### Save the captured data
	recorded_dir = 'captured_data/tetra/'
	os.makedirs(recorded_dir, exist_ok=True)
	np.savetxt(recorded_dir+'scans.csv', np.array(scans), delimiter=',')
	np.savetxt(recorded_dir+'q1_exe.csv', np.array(q1_exe), delimiter=',')


if __name__ == "__main__":
	main()