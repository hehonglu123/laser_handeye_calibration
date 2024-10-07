import numpy as np
from RobotRaconteur.Client import *
from motoman_def import *
from StreamingSend import *



#####################################RR Robot#####################################
config_dir='../../Welding_Motoman/config/'
robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
robot_fujicam=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path='fujicam.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
robot_fujicam_sr=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')

streaming_rate=125.
RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
SS=StreamingSend(RR_robot_sub,streaming_rate=streaming_rate)


q_taught=np.array([-0.43531088,  0.45939827, -0.06757185,  0.20312021, -0.69181849, -0.48304591])
p_global=robot.fwd(q_taught).p
R_fujicam=Rz(np.pi/2)

q_verification=robot_fujicam_sr.inv(p_global,R_fujicam,last_joints=SS.q_cur[:6])
q_verification=np.append(q_verification,SS.q_cur[6:])
SS.jog2q(q_verification)