import numpy as np
from RobotRaconteur.Client import *
from motoman_def import *
from StreamingSend import *

def main():

	#####################################RR Robot#####################################
	streaming_rate=125.
	RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
	SS=StreamingSend(RR_robot_sub,streaming_rate=streaming_rate)
	q_all=[]
	while True:
		try:
			input('Press Enter to store q')
			q_all.append(SS.q_cur[:6])
		except KeyboardInterrupt:
			break
	
	print(q_all)

if __name__ == '__main__':
	main()