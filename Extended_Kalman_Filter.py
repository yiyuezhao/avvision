import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

data=pd.read_csv("File_Path")

t = data['t']  # timestamps [s]

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]

v_var = 0.01  # translation velocity variance  
om_var = 0.1  # rotational velocity variance 
r_var = 0.1  # range measurements variance
b_var = 0.1  # bearing measurement variance

Q_km = np.diag([v_var, om_var]) # input noise covariance 
cov_y = np.diag([r_var, b_var])  # measurement noise covariance 

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance

# Wraps angle to (-pi,pi] range
def wraptopi(x):
    if   x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x

def measurement_update(lk, rk, bk, P_check, x_check):
    theta = x_check[:,2]
    bk = wraptopi(bk)
    theta= wraptopi(theta)    
    A = lk[0] - x_check[:,0] - d * math.cos(theta)
    B = lk[1] - x_check[:,1] - d * math.sin(theta)
    C = (A ** 2 + B ** 2) ** (0.5)
    # 1. Compute measurement Jacobian
    
    M = np.identity(2)
    H = np.array([[-A/C,-B/C,(d/C)*(A*math.sin(theta)- B*math.cos(theta))],
                  [B/(C**2),-A/(C**2),((-d/(C**2))*math.cos(theta)*(A+B))-1]])
    H = H.reshape(2,3) ## the dimension of H is (2,3,1),so you need to reshape it.
    # 2. Compute Kalman Gain
    K = P_check @ H.T @ np.linalg.inv(H @ P_check @ H.T + M @cov_y @ M.T )
    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    h = math.atan2(B,A) - theta
    h = wraptopi(h)
    ss = K @ np.array([rk-C,bk-h])
    x_check = x_check + ss.reshape(1,3)
    
    theta = x_check[:,2]
    theta= wraptopi(theta)

    # 4. Correct covariance
    P_check = (np.identity(3)-K @ H) @ P_check 

    return x_check, P_check

# Main EKF prediction 
for k in range(1, len(t)):  # start at 1 because we've set the initial prediciton
    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)
    x_check = x_est[k - 1,:]
    P_check = P_est[k - 1,:,:]
    theta = x_check[2]
    theta = wraptopi(theta)
    
    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
    cc = delta_t*np.array([[math.cos(theta),0],[math.sin(theta),0],[0,1]])@ np.array([[v[k]],[om[k]]])
    x_check = x_check.reshape(1,3)
    x_check = x_check + cc.T
    
    theta = x_check[0,2]
    theta = wraptopi(theta)

    # 2. Motion model jacobian with respect to last state
    F_km = np.array([[1,0,-delta_t * v[k] *math.sin(theta)],[0,1,delta_t * v[k]* math.cos(theta)],[0,0,1]])

    # 3. Motion model jacobian with respect to noise
    L_km = delta_t * np.array([[math.cos(theta),0],[math.sin(theta),0],[0,1]])

    # 4. Propagate uncertainty
    P_check =  F_km @ P_check @  F_km.T + L_km @ Q_km @L_km.T

    # 5. Update state estimate using available landmark measurements
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)
    # Set final state predictions for timestep
    x_est[k, 0] = x_check[0,0]
    x_est[k, 1] = x_check[0,1]
    x_est[k, 2] = x_check[0,2]
    P_est[k, :, :] = P_check