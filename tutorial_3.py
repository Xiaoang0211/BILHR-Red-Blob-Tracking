#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

import sys
from naoqi import ALProxy
from CMAC import CMAC
import pickle

class Central:
    # set fixed positions for elbow and HeadYaw
    def set_position(self):
<<<<<<< HEAD
        self.set_joint_angles(-0.3,"RElbowYaw")
        rospy.sleep(1)
        self.set_joint_angles(0.1,"RElbowRoll")
        rospy.sleep(1)
        self.set_joint_angles(0,"HeadYaw")
        rospy.sleep(1)


    def red_blob_position(self, x):
        print(x)
        self.set_joint_angles(x[0],"RShoulderPitch")
        self.set_joint_angles(x[1],"RShoulderRoll")
=======

        self.set_joint_angles(-0.3,"RElbowYaw")
        
        self.set_joint_angles(0.1,"RElbowRoll")
        
        self.set_joint_angles(0,"HeadYaw")

    def red_blob_position(self, x):

        self.set_joint_angles(x[0],"RShoulderPitch")
        
        self.set_joint_angles(x[1],"RShoulderRoll")
    
>>>>>>> ac758ec3605f9553eb6a062b45f4ab1679deb6df

    def __init__(self):
        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.jointPub = 0
        self.stiffness = False

        ## tutorial 3
        # new properties for T3
        self.object_coor = np.ones((1,2))
        self.shoulder_state = np.ones((1,2))
        self.map_matrix = np.zeros((150,4))
        self.counter = 0

        pass


    def key_cb(self,data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def joints_cb(self,data):
        #rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name 
        self.joint_angles = data.position
        self.joint_velocities = data.velocity

        pass

    def bumper_cb(self,data):
        rospy.loginfo("bumper: "+str(data.bumper)+" state: "+str(data.state))
        if data.bumper == 0:
            self.stiffness = True
        elif data.bumper == 1:
            self.stiffness = False

    def touch_cb(self,data):
        #rospy.loginfo("touch button: "+str(data.button)+" state: "+str(data.state))
        if data.button == 1 and self.counter < 150:
            # to inform the user that button 1 is clicked
            rospy.loginfo("button 1 clicked!")

            # record shoulder states when button 1 is clicked
            self.shoulder_state[0,0] = self.joint_angles[20]
            self.shoulder_state[0,1] = self.joint_angles[21]
            
            # add the newly recorded shoulder states and object's coordinates to the dataset
            self.map_matrix[self.counter,:] = np.concatenate((self.shoulder_state,self.object_coor), axis = 1)

            # uncomment this if you want to check the newly recorded shoulder state(the angle)
            #rospy.loginfo(str(self.joint_angles[20]))

            # show how far are we by recording
            rospy.loginfo(str(self.map_matrix[self.counter,:])+ str(self.map_matrix.shape) + str(self.counter))

            self.counter += 1
        elif(self.counter == 150):
            rospy.loginfo("record finished!")
            # save the dataset to the given directory
<<<<<<< HEAD
            np.save('/home/bio/ros/bioinspired_ws/src/tutorial_3/scripts/samples_straighthand.npy',self.map_matrix)
=======
            np.save('/home/bio/ros/bioinspired_ws/src/tutorial_3/scripts/samples_backup.npy',self.map_matrix)
>>>>>>> ac758ec3605f9553eb6a062b45f4ab1679deb6df
            


        


    def image_cb(self,data):
        bridge_instance = CvBridge()
        try:
            cv_image = bridge_instance.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
        ### task 9
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        #sets color range to detect shades of red
        low_red1 = np.array([0, 100, 20])
        high_red1 = np.array([10, 255, 255])
        low_red2 = np.array([160, 100, 20])
        high_red2 = np.array([179, 255, 255])

        #creates masks with red regions
        lower_red_mask = cv2.inRange(hsv, low_red1, high_red1)
        upper_red_mask = cv2.inRange(hsv, low_red2, high_red2)
        full_mask = lower_red_mask + upper_red_mask
        
        #cuts out the red areas
        red = cv2.bitwise_and(cv_image, cv_image, mask = full_mask)

        #converts red mask to gray
        gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
        
        #turns image into black and white
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 12)
        cv2.imshow('Test0', thresh)

        #morphological transformation to get nice shaped blob, which we don't
        #actually not used here because the detected objects are also not perfectly round-shaped blob
        # get the desired structuring element, well ellipse (blob)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        #blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel) # erosion followed by dilation,uselful in removing noise

	# inverse the black and white
        #blob = (255 - blob)

        # inverse the black and white
        thresh = (255 - thresh)
        #cv2.imshow('Test1', thresh)

        # Get contours
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1] #TODO: Describe what this does
        # get the largest contour
        result = cv_image.copy()
        if (len(cnts) != 0):
            big_contour = max(cnts, key=cv2.contourArea)

        

            # draw contour
            result = cv_image.copy()
            cv2.drawContours(result, [big_contour], -1, (0,255,0), 1)
            #extract center coordinates
            cX = 0
            cY = 0
            M = cv2.moments(big_contour)
            if (float(M["m00"]) != 0):

                cX = float(M["m10"]) / float(M["m00"])
                cY = float(M["m01"]) / float(M["m00"])

            
            self.object_coor[0,0] = cX
            self.object_coor[0,1] = cY
            np.reshape(self.object_coor,(1,2))

        #rospy.loginfo(str(self.object_coor.shape))


        

        #draws circle at center coordinates
            cv2.circle(result, (int(cX), int(cY)), 7, (0, 255, 0), -1)
        #cv2.imshow("Red", red)
        cv2.imshow('Test2', result)
        #logs center coordinates
        #rospy.loginfo("X: %f \t Y: %f", int(cX), int(cY))

        cv2.waitKey(3) # a small wait time is needed for the image to be displayed correctly



    # sets the stiffness for all joints. can be refined to only toggle single joints, set values between [0,1] etc
    def set_stiffness(self,value):
        if value == True:
            service_name = '/body_stiffness/enable'
        elif value == False:
            service_name = '/body_stiffness/disable'
        try:
            stiffness_service = rospy.ServiceProxy(service_name,Empty)
            stiffness_service()
        except rospy.ServiceException, e:
            rospy.logerr(e)

    def set_joint_angles(self,head_angle,joint_name):

        joint_angles_to_set = JointAnglesWithSpeed()
        
        joint_angles_to_set.joint_names.append(joint_name) # each joint has a specific name, look into the joint_state topic or google
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.1 # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)
        
        


    def central_execute(self):
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10)

        # by default all joints are relaxed
        self.set_stiffness(False) # always check that your robot is in a stable position before disabling the stiffness!!

        rate = rospy.Rate(10) # sets the sleep time to 10ms

        ### Task 1 begins
        IP = "10.152.246.182"
        PORT = 9559
        proxy = ALProxy("ALMotion", IP, PORT)
        
<<<<<<< HEAD
        training_data = np.load('/home/bio/ros/bioinspired_ws/src/tutorial_3/scripts/bilhr2022/samples.npy')

        y = training_data[:, 2:]
        max_y = np.amax(y, axis=0)
        min_y = np.amin(y, axis=0)
        n_input = 2
        n_output = 2
        n_a = 5
        res = np.array([50, 50])
        disp5 = np.array([[3, 2], [4, 2], [1, 1], [3, 0], [0, 4]])
        cmac = CMAC(n_input, n_output, n_a, res, disp5, max_y, min_y)
        cmac.W = np.load('/home/bio/ros/bioinspired_ws/src/tutorial_3/scripts/W_straighthand.npy')
=======

>>>>>>> ac758ec3605f9553eb6a062b45f4ab1679deb6df
        while not rospy.is_shutdown():
            
            # this part is to set the robot arm to a fixed position
            # because of the consideration that our training result is probably not that robust
            if(self.stiffness == 0):
                # let all joints be relaxed
                proxy.setStiffnesses("Body", 0)
            elif(self.stiffness == 1):
                # set elbow and head yaw to fixed position
                self.set_position()
                # set the elbow and HeadYaw stiff
<<<<<<< HEAD
                name = ['RElbowYaw', 'RElbowRoll', 'HeadYaw',]
                stiffness = 0.9
                proxy.setStiffnesses(name, stiffness)

            name = ["RShoulderPitch", "RShoulderRoll"]
            stiffness = 1
            proxy.setStiffnesses(name, stiffness)
            ### Task 1 ends
        ### task 2 done
        ### task 3 apply the CMAC controller to NAO
            # the pixel coordinate of object as input for mapping
     
            y_sample = self.object_coor
            
            #load the object, the trained CMAC
            #with open('trained_cmac.pkl', 'rb') as f:
            #    cmac_test = pickle.load(f)

            y_sample = y_sample.reshape((n_input,))

            #shoulder setting calculated by the trained CMAC
            x_sample = cmac.cmacMap(y_sample)
            #apply the shoulder setting to NAO
            self.red_blob_position(x_sample)
            
=======
                name = ['RElbowYaw', 'RElbowRoll', 'HeadYaw']
                stiffness = 0.9
                proxy.setStiffnesses(name, stiffness)


        ### Task 1 ends
        ### task 2 done
        ### task 3 apply the CMAC controller to NAO
            # the pixel coordinate of object as input for mapping
            y_sample = self.object_coor
            #load the object, the trained CMAC
            with open('trained_cmac.pkl', 'rb') as f:
                cmac_test = pickle.load(f)

            
            #training_data = np.load('samples.npy')
            #y = training_data[:, 2:]
            #max_y = np.amax(y, axis=0)
            #min_y = np.amin(y, axis=0)
            #n_input = 2
            #n_output = 2
            #n_a = 5
            #res = np.array([50, 50])
            #disp5 = np.array([[3, 2], [4, 2], [1, 1], [3, 0], [0, 4]])
            #cmac = CMAC(n_input, n_output, n_a, res, disp5, max_y, min_y)
            #cmac.W = np.load('/home/bio/ros/bioinspired_ws/src/tutorial_3/scripts/W.npy')

            #shoulder setting calculated by the trained CMAC
            x_sample = cmac_test.cmacMap(y_sample)
            #apply the shoulder setting to NAO
            self.red_blob_position(x_sample)
>>>>>>> ac758ec3605f9553eb6a062b45f4ab1679deb6df

            rate.sleep()

    # rospy.spin() just blocks the code from exiting, if you need to do any periodic tasks use the above loop
    # each Subscriber is handled in its own thread
    #rospy.spin()

if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
