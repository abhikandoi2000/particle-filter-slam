#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy import io
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')

from dataset import JointDataset, LidarDataset
from imgutils import upscale
from transform import T


CELL_SIZE = 0.05 # meters per cell
PIXELS_PER_METER=int(1/CELL_SIZE)
MAP_SIZE = 70 # meters
GRID_HEIGHT = MAP_SIZE*PIXELS_PER_METER
GRID_WIDTH = MAP_SIZE*PIXELS_PER_METER

LIDAR_HEIGHT_HEAD_FRAME = 0.15 # meters
HEAD_HEIGHT_COM_FRAME = 0.33 # meters; head from center of mass (COM)
COM_HEIGHT_WORLD_FRAME = 0.93 # meters; center of mass from ground, vertical height


# In[2]:


def get_coord_of_obstacle(distance, angle):
    x = distance * np.sin(angle)
    y = distance * np.cos(angle)
    return x, y

# lidar_pt = (x, y); tuple of point's x, y coordinates
# robot_pose = (x, y, alpha) of robot in world frame
# joint_angles = (neck_angle, head_angle) in head frame
def lidar_to_world_frame(lidar_pt, robot_pose, joint_angles):
    theta_neck, theta_head = joint_angles
    robot_x, robot_y, alpha_robot = robot_pose

    head_T_lidar = T((0, 0, LIDAR_HEIGHT_HEAD_FRAME), (theta_neck, 0, theta_head)) # no rotation about y-axis of head
    com_T_head = T((0, 0, HEAD_HEIGHT_COM_FRAME), (0, 0, 0)) # only vertical translation
    world_T_com = T((robot_x, robot_y, COM_HEIGHT_WORLD_FRAME), (-alpha_robot, 0, 0)) # only yaw of robot

    point_x_lidar, point_y_lidar = lidar_pt
    point_lidar = np.array([[point_x_lidar],[point_y_lidar],[0],[1]])
    point_head = np.matmul(head_T_lidar, point_lidar)
    point_com = np.matmul(com_T_head, point_head)
    point_world = np.matmul(world_T_com, point_com)

    point_world = point_world.T
    point_world = point_world / point_world[0,3] # non-homogenized point

    x, y, z = point_world[0,0], point_world[0,1], point_world[0,2]

    return x, y, z


def world_to_grid_location(world_pt):
    """ grid has x axis horizontal and increases towards right
        has y axis vertical and increases towards bottom
    """
    proj_world_pt = np.array([[world_pt[0]], [world_pt[1]], [1]]) # projected to xy plane, homogeneous
    R = np.array([[                 0,  PIXELS_PER_METER,    GRID_WIDTH/2],
                   [-PIXELS_PER_METER,                  0,   GRID_HEIGHT/2],
                   [                0,                  0,               1]])
    point_grid = np.matmul(R, proj_world_pt).reshape(3)
    return int(np.floor(point_grid[0])), int(np.floor(point_grid[1]))


def cv_world_to_grid_location(world_pt):
    """ grid has x axis horizontal and increases towards right
        has y axis vertical and increases towards bottom
    """
    proj_world_pt = np.array([[world_pt[0]], [world_pt[1]], [1]]) # projected to xy plane, homogeneous
    R = np.array([[  PIXELS_PER_METER,                  0,    GRID_WIDTH/2],
                   [                0,  -PIXELS_PER_METER,   GRID_HEIGHT/2],
                   [                0,                  0,               1]])
    point_grid = np.matmul(R, proj_world_pt).reshape(3)
    return int(np.floor(point_grid[0])), int(np.floor(point_grid[1]))


# In[3]:


# independent of current pose
def motion_noise():
    # sample from noise distribution
    mu_x, sigma_x = 0, 0.04 # mean and standard deviation
    epsilon_x = np.random.normal(mu_x, sigma_x, 1)[0]

    mu_y, sigma_y = 0, 0.04 # mean and standard deviation
    epsilon_y = np.random.normal(mu_y, sigma_y, 1)[0]

    mu_alpha, sigma_alpha = 0, 0.02 # mean and standard deviation
    epsilon_alpha = np.random.normal(mu_alpha, sigma_alpha, 1)[0]

    return np.array([epsilon_x, epsilon_y, epsilon_alpha])

def next_pose(current_pose, delta_pose, noise):
    pose_t_plus_1 = np.add(current_pose, delta_pose)
    if not noise:
        return pose_t_plus_1
    epsilon = motion_noise()
    return np.add(pose_t_plus_1, epsilon)


# In[4]:


def draw_lidar_ray_and_surface(occupancy, distance, ray_angle, robot_pose, joint_angles):
    start_drw = time.process_time()
    x, y = get_coord_of_obstacle(distance, ray_angle)
    lidar_pt = (x, y)
    point_world = lidar_to_world_frame(lidar_pt, robot_pose, joint_angles)
    p_world_z = point_world[2]

    # ignore points close to ground
    if p_world_z < 0.05:
        return occupancy

    # Draw a white line with thickness of 1 px
    color = (255, 255, 255) # BGR
    thickness = 1 # pixels
    robot_grid_loc = world_to_grid_location(robot_pose)
    surface_grid_loc = world_to_grid_location(point_world)
    
    occupancy = cv2.line(occupancy, robot_grid_loc, surface_grid_loc,
                            color, thickness)
    """
    if surface_grid_loc[1] >= 0 and surface_grid_loc[1] < GRID_HEIGHT and \
        surface_grid_loc[0] >= 0 and surface_grid_loc[0] < GRID_WIDTH:
        occupancy[surface_grid_loc[1], surface_grid_loc[0], :] = np.array([255,0,0])
    """
    return occupancy


def draw_path_followed(pose_list):
    # draw robot trajectory
    alpha_robot = 0 # radians
    robot_curr_pos = (0, 0) # meters
    robot_start_pos = robot_curr_pos
    robot_start_pos_grid = world_to_grid_location(robot_start_pos)
    radius = 1 # pixels
    color = (255, 0, 0) # BGR
    thickness = 1
    color_robot = (0, 0, 255) # BGR
    
    # draw start point (green point)
    occupancy = cv2.circle(occupancy, world_to_grid_location((0, 0)), radius*3, (0,255,0), thickness*3)

    for idx, lidar_data in enumerate(scanset):
        x_robot, y_robot = robot_curr_pos
        # use delta_pose to update absolute position of robot
        y_robot = y_robot - lidar_data['delta_pose'][1] # meters
        x_robot = x_robot + lidar_data['delta_pose'][0] # meters
        alpha_robot = alpha_robot + lidar_data['delta_pose'][2] # radians
        robot_curr_pos = (x_robot, y_robot)

        robot_curr_pos_grid = world_to_grid_location(robot_curr_pos)


        arrow_end_pt = world_to_grid_location((x_robot + 0.50*np.cos(alpha_robot), 
                                            y_robot - 0.50*np.sin(alpha_robot)))

        # draw the robot's position
        if idx % 500 == 0:
            occupancy = cv2.line(occupancy, robot_start_pos_grid,
                                  robot_curr_pos_grid, color, thickness)
            occupancy = cv2.arrowedLine(occupancy, robot_curr_pos_grid, arrow_end_pt, (60, 76, 231), 1, tipLength = 0.4)
            occupancy = cv2.circle(occupancy, robot_curr_pos_grid, radius*2, color_robot, thickness*2)
            robot_start_pos = (x_robot, y_robot) # to draw trajectory
            robot_start_pos_grid = world_to_grid_location(robot_start_pos)
        

    # draw end point (yellow point)
    occupancy = cv2.circle(occupancy, robot_curr_pos_grid, radius*3, (0, 79, 255), thickness*3)
    occupancy = cv2.arrowedLine(occupancy, robot_curr_pos_grid,
                                arrow_end_pt, (60, 76, 231), 1, tipLength = 0.4)


# In[5]:


angular_range = np.arange(-135, 135.25, 0.25) * np.pi / float(180)
DISTANCE_FAR_AWAY = 30 # meters
DISTANCE_TOO_CLOSE = 0.1 # meters

def occupied(lidar_scan, joint_angles, robot_pose):
    angles_0 = angular_range.T
    ranges_0 = np.double(lidar_scan).T

    # take valid indices
    indValid_0 = np.logical_and((ranges_0 < DISTANCE_FAR_AWAY),                             (ranges_0 > DISTANCE_TOO_CLOSE))
    ranges_0 = ranges_0[indValid_0]
    angles_0 = angles_0[indValid_0]

    # xy position in the sensor frame
    x_coords_0 = np.array([ranges_0*np.cos(angles_0)])
    y_coords_0 = np.array([ranges_0*np.sin(angles_0)])

    # convert position in the map frame here 
    tmp = np.concatenate([x_coords_0, y_coords_0],axis=0)
    Y = np.concatenate([tmp, np.zeros(x_coords_0.shape)], axis=0)
    homogeneous_pts = np.concatenate([Y, np.ones(x_coords_0.shape)], axis=0)

    # find lidar points in world frame
    theta_neck, theta_head = joint_angles
    robot_x, robot_y, alpha_robot = robot_pose

    head_T_lidar = T((0, 0, LIDAR_HEIGHT_HEAD_FRAME), (theta_neck, 0, theta_head)) # no rotation about y-axis of head
    com_T_head = T((0, 0, HEAD_HEIGHT_COM_FRAME), (0, 0, 0)) # only vertical translation
    points_head = np.matmul(head_T_lidar, homogeneous_pts)
    points_com = np.matmul(com_T_head, points_head)
    
    world_T_com = T((robot_x, robot_y, COM_HEIGHT_WORLD_FRAME),                     (alpha_robot, 0, 0)) # only yaw of robot

    points_world = np.matmul(world_T_com, points_com) # shape is (4, xs0.shape)
    points_world = points_world / points_world[3,:] # non homogenise
    
    
    laserGrid = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    vertcl_grid_0 = np.int16((-points_world[1,:]+ (MAP_SIZE/2) ) / CELL_SIZE)
    horzntl_grid_0 = np.int16((points_world[0,:]+ (MAP_SIZE/2) ) / CELL_SIZE)
    laserGrid[vertcl_grid_0, horzntl_grid_0] = 1
    
    return laserGrid


# In[16]:


ANGULAR_RANGE = np.arange(-135, 135.25, 0.25) * np.pi / float(180)
ANGULAR_RANGE_T = ANGULAR_RANGE.T


def getContourPts(lidar_scan, robot_pose, joint_angles):
    angles_0 = ANGULAR_RANGE_T
    ranges_0 = np.double(lidar_scan).T

    # take valid indices
    indValid_0 = np.logical_and((ranges_0 < DISTANCE_FAR_AWAY),                             (ranges_0 > DISTANCE_TOO_CLOSE))
    ranges_0 = ranges_0[indValid_0]
    angles_0 = angles_0[indValid_0]

    # xy position in the sensor frame
    x_coords_0 = np.array([ranges_0*np.cos(angles_0)])
    y_coords_0 = np.array([ranges_0*np.sin(angles_0)])

    # convert position in the map frame here 
    tmp = np.concatenate([x_coords_0, y_coords_0],axis=0)
    Y = np.concatenate([tmp, np.zeros(x_coords_0.shape)], axis=0)
    homogeneous_pts = np.concatenate([Y, np.ones(x_coords_0.shape)], axis=0)

    # find lidar points in world frame
    theta_neck, theta_head = joint_angles
    robot_x, robot_y, alpha_robot = robot_pose

    head_T_lidar = T((0, 0, LIDAR_HEIGHT_HEAD_FRAME), (theta_neck, 0, theta_head)) # no rotation about y-axis of head
    com_T_head = T((0, 0, HEAD_HEIGHT_COM_FRAME), (0, 0, 0)) # only vertical translation
    points_head = np.matmul(head_T_lidar, homogeneous_pts)
    points_com = np.matmul(com_T_head, points_head)
    
    world_T_com = T((robot_x, robot_y, COM_HEIGHT_WORLD_FRAME),                     (alpha_robot, 0, 0)) # only yaw of robot

    points_world = np.matmul(world_T_com, points_com) # shape is (4, xs0.shape)
    
    points_world = np.hstack((points_world, np.array([[robot_x],[robot_y],[0],[1]])))
    
    R = np.array([[                 0,  -PIXELS_PER_METER,    0, GRID_WIDTH/2],
                   [PIXELS_PER_METER,                  0,   0, GRID_HEIGHT/2],
                   [                0,                  0,   0,            0],
                   [                0,                  0,   0,            1]])
    points_grid = np.matmul(R, points_world)
    
    points_grid = points_grid / points_grid[3,:] # non homogenise
    
    x = list(np.asarray(points_grid[0,:]).reshape(ranges_0.shape[0]+1))
    y = list(np.asarray(points_grid[1,:]).reshape(ranges_0.shape[0]+1))

    contour = np.array(list(zip(y, x))).reshape((-1,1,2)).astype(np.int32)

#     print(contour.shape)
    return contour

def updateLogOdds(log_odd_map, lidar_data, pose, occupancy_img=None):
    timestamp = lidar_data['timestamp']
    joint_angles = joint_dataset.angles_at_time(timestamp) # theta_neck, theta_head
    lidar_scan = lidar_data['scan']
    
    robot_pose = tuple(pose)
    
    if occupancy_img is None:
        # init occupancy map
        occupancy_img = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), np.uint8)
        
        contour = getContourPts(lidar_scan, robot_pose, joint_angles)
        occupancy_img = cv2.drawContours(occupancy_img, contours=[contour],                                         contourIdx=-1, color=(255,255,255), thickness=-1)
        
    laserGrid = occupied(lidar_scan, joint_angles, robot_pose)
    
    # update log odds map using the scan
    # update log odds for occupied and unoccupied cells
    binary_occupancy = np.multiply(occupancy_img[:,:,0],                             occupancy_img[:,:,0] == 255)
    occupied_cells = laserGrid
    free_cells = np.multiply(binary_occupancy,                             occupancy_img[:,:,1] == 255)
    free_cells[occupied_cells == 1] = 0
    binary_occupancy = occupied_cells.astype(int) + np.multiply(free_cells.astype(int), -1)
    true_vs_false_positive = 80/20
    del_lambda_t = np.multiply(binary_occupancy, np.log(true_vs_false_positive))
    log_odd_map = np.add(log_odd_map, del_lambda_t)
    return log_odd_map

def noise_matrix(N):
    mu_x, sigma_x = 0, 0.01 # mean and standard deviation
    epsilon_x = np.random.normal(mu_x, sigma_x, (N,1))
    mu_y, sigma_y = 0, 0.01 # mean and standard deviation
    epsilon_y = np.random.normal(mu_y, sigma_y, (N,1))
    mu_alpha, sigma_alpha = 0, 0.005 # mean and standard deviation
    epsilon_alpha = np.random.normal(mu_alpha, sigma_alpha, (N,1))
    xy_noise = np.hstack((epsilon_x, epsilon_y))
    return np.hstack((xy_noise, epsilon_alpha))

def vectorizedMapCorr(log_odd_map, vp):
    ver_grid = np.int16(np.multiply((-vp[1,:]+MAP_SIZE/2), PIXELS_PER_METER))
    hor_grid = np.int16(np.multiply((vp[0,:]+MAP_SIZE/2), PIXELS_PER_METER))
    
    corr = np.sum(log_odd_map[ver_grid, hor_grid] > 0)
    return corr


def findLidarCorrelation(log_odd_map, lidar_scan, robot_pose,                          joint_angles):
    angles = ANGULAR_RANGE_T
    ranges = np.double(lidar_scan).T

    # take valid indices for points in range
    validIndexes = np.logical_and((ranges < DISTANCE_FAR_AWAY),                             (ranges > DISTANCE_TOO_CLOSE))
    
    ranges = ranges[validIndexes]
    angles = angles[validIndexes]

    # xy position in the sensor frame
    xs0 = np.array([ranges * np.cos(angles)])
    ys0 = np.array([ranges * np.sin(angles)])

    # convert position in the map frame here 
    tmp = np.concatenate([xs0, ys0], axis=0)
    Y = np.concatenate([tmp, np.zeros(xs0.shape)], axis=0)
    homogeneous_pts = np.concatenate([Y, np.ones(xs0.shape)], axis=0)

    # find lidar points in world frame
    theta_neck, theta_head = joint_angles
    robot_x, robot_y, alpha_robot = robot_pose
    head_T_lidar = T((0, 0, LIDAR_HEIGHT_HEAD_FRAME), (theta_neck, 0, theta_head)) # no rotation about y-axis of head
    com_T_head = T((0, 0, HEAD_HEIGHT_COM_FRAME), (0, 0, 0)) # only vertical translation
    points_head = np.matmul(head_T_lidar, homogeneous_pts)
    points_com = np.matmul(com_T_head, points_head)

    max_corr_yaw_idx = -1
    max_corr_val = -1

    # yaw variation
    alpha_range = np.array([-0.03, 0, 0.03])
#     alpha_range = [0] # no yaw variation

    for yaw_idx, del_alpha in enumerate(alpha_range):
        world_T_com = T((robot_x, robot_y, COM_HEIGHT_WORLD_FRAME),                         (alpha_robot+del_alpha, 0, 0)) # only yaw of robot

        points_world = np.matmul(world_T_com, points_com) # shape is (4, xs0.shape)
        points_world = points_world / points_world[3,:] # non homogenise
        corr = vectorizedMapCorr(log_odd_map, points_world)
        max_c = corr

        if max_c > max_corr_val:
            max_corr_val = max_c
            max_corr_yaw_idx = yaw_idx
    
    return max_corr_val, alpha_range[max_corr_yaw_idx], xs0, ys0

def drawCorrelationGraphs(xs0, ys0, robot_x, robot_y, occupancy_map, laserGrid, overlap, c):
    # plot laser scan, occupancy map and correlation coefficient map
    fig = plt.figure(figsize=(16,16))

    #plot original lidar points
    ax1 = fig.add_subplot(231)
    plt.plot(xs0, ys0, '.k')
    plt.scatter(robot_x, robot_y, s=30, c='r')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Laser reading (red being robot location)")
    plt.axis('equal')

    #plot map
    ax2 = fig.add_subplot(234)
    plt.imshow(occupancy_map, cmap="hot")
    plt.title("Occupancy map")

    #plot map
    ax3 = fig.add_subplot(235)
    plt.imshow(laserGrid, cmap="hot")
    plt.title("Laser scan")

    #plot map
    ax4 = fig.add_subplot(236)
    plt.imshow(overlap, cmap="hot")
    plt.title("Overlap b/w occupancy and laser scan")

    #plot correlation
    ax5 = fig.add_subplot(232, projection='3d')
    X, Y = np.meshgrid(np.arange(0, 3), np.arange(0, 3))
    ax5.plot_surface(X, Y, c, linewidth=0, cmap=plt.cm.jet,
                        antialiased=False, rstride=1, cstride=1)
    plt.title("Correlation coefficient map")
    plt.show()

def updateStep(corr_vals, weights):
    corr_vals = corr_vals - np.max(corr_vals)
    # print('[ CORRELATION VALUES ]:', corr_vals)
    
    p_h_z_given_x_and_m = softmax(corr_vals)

    denominator = np.sum(np.multiply(weights, p_h_z_given_x_and_m))
    weights = np.multiply(weights, p_h_z_given_x_and_m) / denominator

    # print('[ WEIGHTS ]:', weights)
    
    return weights

def drawParticlesOnMap(log_odd_map, poses, weights, N, trajList):
    # draw particles on map
    binary_map = log_odd_map.copy()
    binary_map[binary_map > 0] = 1e-9
    binary_map[binary_map == 0] = 0.5
    binary_map[binary_map < 0] = 1

    plt.figure(figsize=(10, 10))
    plt.imshow(binary_map, cmap='gray')
    for particle in np.arange(0, N):
        pose = poses[particle,:]
        robot_y_grid = np.int16((-pose[1] + (MAP_SIZE/2) ) / CELL_SIZE)
        robot_x_grid = np.int16((pose[0] + (MAP_SIZE/2) ) / CELL_SIZE)
        color = 'r' if weights[particle] > 0.6 else 'g'
        plt.scatter(robot_x_grid, robot_y_grid, s=20+50*weights[particle], c=color)
        if weights[particle] > 0.6:
            dx = 45*np.cos(pose[2])
            dy = -45*np.sin(pose[2])
            plt.arrow(robot_x_grid, robot_y_grid, dx, dy,                       head_width=1.2, head_length=1.2, fc='r', ec='r')
    
    robottraj_y_grid = np.int16((-trajList[:,1] + (MAP_SIZE/2) ) / CELL_SIZE)
    robottraj_x_grid = np.int16((trajList[:,0] + (MAP_SIZE/2) ) / CELL_SIZE)
    plt.scatter(robottraj_x_grid, robottraj_y_grid, s=30, c='b')
    plt.title('particle positions')
    plt.show()

def resampleParticles(weights, N, poses):
#     print('RE-sampling particles')
    new_weights = np.full(N, 1/N)
    candidates = np.arange(N)
    drawn_indexes = np.random.choice(candidates, N,                                      p=weights)
    return poses[drawn_indexes], new_weights

def drawLogOddsAndOccupancyMaps(log_odd_map):
    plt.figure(figsize=(10, 10))
    plt.imshow(log_odd_map)
    plt.title('log odds map')
    plt.show()

    binary_map = log_odd_map.copy()
    binary_map[binary_map > 0] = 1e-9
    binary_map[binary_map == 0] = 0.5
    binary_map[binary_map < 0] = 1
    plt.figure(figsize=(10, 10))
    plt.imshow(binary_map, cmap='gray')
    plt.title('binary map')
    plt.show()


# In[17]:


def SLAM(particles, scanset, joint_dataset, update_interval, draw_interval, traj_draw_interval):
    N = particles # number of particles
    weights = np.full(N, 1/N)
    poses = np.zeros((N, 3))
    corr_vals = np.zeros(N)

    log_odd_map = np.zeros((GRID_HEIGHT, GRID_WIDTH), np.float32)
    start = time.process_time()
    
    delta_x, delta_y, delta_alpha = scanset[0]['delta_pose']
    delta_pose = np.array([delta_x, -delta_y, delta_alpha])

    poses = np.add(poses, delta_pose)
    
    trajList = poses[0,:]
    
    log_odd_map = updateLogOdds(log_odd_map, scanset[0], poses[0,:])
    print('Used First Lidar Scan to update log odds map')
    
    done = 0
    for lidar_data in scanset[1:]:
        done = done + 1
        occupancy_imgs = [None] * N
        timestamp, delta_pose, lidar_scan = lidar_data['timestamp'],                                         lidar_data['delta_pose'],                                         lidar_data['scan']
        delta_x, delta_y, delta_alpha = delta_pose
        delta_pose = np.array([delta_x, -delta_y, delta_alpha])
        
        poses = np.add(poses, delta_pose)
        noise_val = noise_matrix(N)
        assert noise_val.shape == (N, 3),             'noise matrix isnt well shaped, should be Nx3'
        poses = np.add(poses, noise_val)
        
        joint_angles = joint_dataset.angles_at_time(timestamp) # theta_neck, theta_head

        for particle_idx in np.arange(0, N):
            occupancy_img = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), np.uint8)

            if done % update_interval == 0 or done % draw_interval == 0:
                start_draw = time.process_time()
                robot_pose = tuple(poses[particle_idx,:])
                
                contour = getContourPts(lidar_scan, robot_pose, joint_angles)
                occupancy_img = cv2.drawContours(occupancy_img, contours=[contour],                                                  contourIdx=-1, color=(255,255,255), thickness=-1)

                occupancy_imgs[particle_idx] = occupancy_img
                #print('time taken to run draw lidar contour is {} sec'.format(time.process_time()-start_draw))

                
                start_lidar_corr = time.process_time()
                # update particle weights
                max_corr_val, del_yaw, xs0, ys0 = findLidarCorrelation(log_odd_map,                                                         lidar_scan,                                                         robot_pose,                                                         joint_angles)
                poses[particle_idx,2] += del_yaw
                #print('time taken to run findLidarCorrelation is {} sec'.format(time.process_time()-start_lidar_corr))
                    
                #if done % draw_interval == 0:
                #   drawCorrelationGraphs(xs0, ys0, robot_pose[0], robot_pose[1], \
                #                          occupancy_map, laserGrid, \
                #                          overlap, c)

                corr_vals[particle_idx] = max_corr_val
                
        if done % update_interval == 0 or done % draw_interval == 0 or done % traj_draw_interval == 0:
            start_update = time.process_time()

            if done % draw_interval == 0:
                print('\nProcessed a group of scan(s), done {}, time since start is {}'.format(
                        done, time.process_time() - start))

            weights = updateStep(corr_vals, weights)
            
            # UPDATE log odds map using best particle's scan
            best_particle = np.argmax(weights)
            
            best_part_pose = poses[best_particle,:]
            if done % traj_draw_interval == 0:
                trajList = np.vstack((trajList, best_part_pose))
            
            log_odd_map = updateLogOdds(log_odd_map, lidar_data,                                         best_part_pose,                                         occupancy_imgs[best_particle])

            if done % draw_interval == 0 or done == 1:
                drawParticlesOnMap(log_odd_map, poses, weights, N, trajList)

            Neff = 1/np.sum(np.square(weights))
            if Neff <= max(1+1e-5, 0.2 * N):
                poses, weights = resampleParticles(weights, N, poses)

    drawLogOddsAndOccupancyMaps(log_odd_map)


# In[12]:


def mapCorrelation(im, x_range, y_range, xmin, xmax, ymin, ymax, vp):
    nx = im.shape[0]
    ny = im.shape[1]
    xresolution = (xmax-xmin)/nx # 40/80 ie. 0.5
    yresolution = (ymax-ymin)/ny # 0.5
    
    nxs = x_range.size
    nys = y_range.size
    cpr = np.zeros((nxs, nys))
    
    laserGrid = np.zeros((nx, ny))
    x_grid = np.int16((vp[1,:]-ymin)/yresolution)
    y_grid = np.int16((vp[0,:]-xmin)/xresolution)
    laserGrid[x_grid, y_grid] = 1

    for jy in range(0, nys):
        y1 = vp[1,:] + y_range[jy] # 1 x 1076
        iy = np.int16(np.floor((y1-ymin)/yresolution))
        for jx in range(0, nxs):
            x1 = vp[0,:] + x_range[jx] # 1 x 1076
            ix = np.int16(np.floor((-x1-xmin)/xresolution))
            valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)),                                         np.logical_and((ix >=0), (ix < nx)))
            overlaps = im[ix[valid], iy[valid]]
            cpr[jx,jy] = np.sum(overlaps)
    return cpr, laserGrid, np.multiply(laserGrid, im)

def robot_trajectory(scanset, start, end):
    pose = np.zeros((1, 3))
    scanset_crop = scanset[start:end]
    idx = 0
    for lidar_data in scanset_crop:
        idx = idx + 1
        timestamp, delta_pose, lidar_scan = lidar_data['timestamp'],                                             lidar_data['delta_pose'],                                             lidar_data['scan']
        delta_x, delta_y, delta_alpha = delta_pose
        delta_pose = np.array([delta_x, delta_y, delta_alpha])

        # use motion model (no noise) to update pose
        pose = next_pose(pose, delta_pose, noise=False)
        
        posevals = pose[0].tolist()
        if idx % 50 == 0:
            print('at index {}, robot is at pose ({}, {}, {} degree)'                   .format(idx, round(posevals[0], 4), round(posevals[1], 4), round(posevals[2]*180/np.pi, 4)))

def abs_pose_at(scanset, end):
    pose = np.zeros((1, 3))
    scanset_crop = scanset[:end+1]
    idx = 0
    for lidar_data in scanset_crop:
        idx = idx + 1
        timestamp, delta_pose, lidar_scan = lidar_data['timestamp'],                                             lidar_data['delta_pose'],                                             lidar_data['scan']
        delta_x, delta_y, delta_alpha = delta_pose
        delta_pose = np.array([delta_x, delta_y, delta_alpha])
        
        pose = np.add(pose, delta_pose)
        
        posevals = pose[0].tolist()
    return np.array(posevals)


# In[ ]:


lidar_dataset = LidarDataset("lidar/train_lidar0.mat")
joint_dataset = JointDataset("joint/train_joint0.mat")


# In[ ]:


simple_scanset = []
printable_scanset = []
positions = [0, 3200, 3800, 4400, 4800]
prev_pose = np.array([0,0,0])

for pos in positions:
    lidar_data = lidar_dataset.scanset[pos]
    timestamp, delta_pose, lidar_scan = lidar_data['timestamp'],                                             lidar_data['delta_pose'],                                             lidar_data['scan']
    pose = abs_pose_at(lidar_dataset.scanset, pos)
    delta_pose = np.subtract(pose, prev_pose)
    simple_scanset.append({'timestamp': timestamp, 
                           'delta_pose': (delta_pose[0], delta_pose[1], delta_pose[2]),
                            'scan': lidar_scan})
    printable_scanset.append({'timestamp': timestamp, 
                           'delta_pose': (delta_pose[0], delta_pose[1], delta_pose[2])})
    print('lidar idx {}'.format(pos))
    print('timestamp: {}'.format(timestamp))
    print('absolute pose', pose)
    print('delta pose', delta_pose)
    print('\n')
    prev_pose = pose


# In[ ]:


# robot_trajectory(lidar_dataset.scanset, 0, len(lidar_dataset.scanset))
abs_pose_at(lidar_dataset.scanset, 4206)


# In[ ]:


simple_scanset = []
printable_scanset = []
positions = [0, 4206, 5549, 7562, 9685]
prev_pose = np.array([0,0,0])

for pos in positions:
    lidar_data = lidar_dataset.scanset[pos]
    timestamp, delta_pose, lidar_scan = lidar_data['timestamp'],                                             lidar_data['delta_pose'],                                             lidar_data['scan']
    pose = abs_pose_at(lidar_dataset.scanset, pos)
    delta_pose = np.subtract(pose, prev_pose)
    simple_scanset.append({'timestamp': timestamp, 
                           'delta_pose': (delta_pose[0], delta_pose[1], delta_pose[2]),
                            'scan': lidar_scan})
    printable_scanset.append({'timestamp': timestamp, 
                           'delta_pose': (delta_pose[0], delta_pose[1], delta_pose[2])})
    print('lidar idx {}'.format(pos))
    print('timestamp: {}'.format(timestamp))
    print('absolute pose', pose)
    print('delta pose', delta_pose)
    print('\n')
    prev_pose = pose

simple_joints = JointDataset("joint/train_joint3.mat")
simple_joints.set_timestamps([1426800936.265628, 1426801041.361129, 1426801074.92363,                             1426801125.17188, 1426801178.141763])
"""
org_strs = ["1426800936.265628,0,0",
"1426801041.361129,-0.00020685383727340204,0.27896308494690997",
"1426801074.92363,0.0001447976860913814,0.3659864942878302",
"1426801125.17188,0.0,0.28152807252910017",
"1426801178.141763,0.43625474280960486,0.19167076561753432"]
"""


strs = ["1426800936.265628,0,0",
"1426801041.361129, -0.00020685383727340204,0.27896308494690997",
"1426801074.92363,0.0001447976860913814,0.3659864942878302",
"1426801125.17188,0.0,0.28152807252910017",
"1426801178.141763,0.43625474280960486,0.19167076561753432"]
jtset = []
for jointstr in strs:
    vals = jointstr.split(',')
    timestamp = float(vals[0])
    neck_angle = -float(vals[1])
    head_angle = -float(vals[2])
    jtset.append({'timestamp': timestamp,'neck_angle': neck_angle, 'head_angle': head_angle})
    
simple_joints.set_jointset(jtset)

print(jtset)

"""

"""


# In[ ]:


simple_scanset[1]['delta_pose'] = (simple_scanset[1]['delta_pose'][0], 
                                   simple_scanset[1]['delta_pose'][1], 
                                   -np.pi/2)

simple_scanset[0]['delta_pose'] = (simple_scanset[0]['delta_pose'][0], 
                                   simple_scanset[0]['delta_pose'][1], 
                                   -np.pi/6)


# In[ ]:


particles = 1
# file 4, hallway straight 6017 to 7100
SLAM(particles, simple_scanset[:1], simple_joints,          update_interval=1, draw_interval=1)


# In[ ]:


particles = 1
# file 4, hallway straight 6017 to 7100
SLAM(particles, simple_scanset[:1], simple_joints,          update_interval=1, draw_interval=1)


# In[ ]:





# In[19]:


lidar_dataset = LidarDataset("lidar/train_lidar3.mat")
joint_dataset = JointDataset("joint/train_joint3.mat")

particles = 50
# file 4, hallway straight 6017 to 7100
SLAM(particles, lidar_dataset.scanset, joint_dataset,          update_interval=10, draw_interval=400, traj_draw_interval=400)


# In[ ]:


# 3200
# -0.0,-0.18776122809306703

# 3800
# -2.0685383727340202e-05,-0.18776122809306703
# neck, head


# In[ ]:


def draw_lidar_ray(occupancy, distance, ray_angle, robot_pose, joint_angles):
    x, y = get_coord_of_obstacle(distance, ray_angle)
    lidar_pt = (x, y)
    point_world = lidar_to_world_frame(lidar_pt, robot_pose, joint_angles)
    p_world_z = point_world[2]
    
    # ignore points close to ground
    if p_world_z < 0.05:
        return occupancy

    # Draw a white line with thickness of 1 px
    color = (255, 255, 255) # BGR
    thickness = 1 # pixels
    robot_grid_loc = world_to_grid_location(robot_pose)
    surface_grid_loc = world_to_grid_location(point_world)
    occupancy = cv2.line(occupancy, robot_grid_loc, surface_grid_loc,
                            color, thickness)
    return occupancy


def move_and_draw_occupancy(scanset, joint_dataset, draw_interval):
    occupancy_img = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), np.uint8)
    log_odd_map = np.zeros((GRID_HEIGHT, GRID_WIDTH), np.float32)

    # remove far away points
    DISTANCE_FAR_AWAY = 30 # meters
    DISTANCE_TOO_CLOSE = 0.5 # meters

    pose_t = np.array([0, 0, 0]) # m, m, radians
    pose_t_plus_1 = pose_t # robot doesn't move, if no data is provided

    done = 0

    start = time.process_time()

    for lidar_data in scanset:
        done = done + 1
        if done % 2000 == 0:
            print('done {}, total time is {}'.format(done,
                time.process_time() - start))
        
        timestamp, delta_pose, lidar_scan = lidar_data['timestamp'],                                         lidar_data['delta_pose'],                                         lidar_data['scan']
        delta_x, delta_y, delta_alpha = delta_pose
        delta_pose = np.array([delta_x, -delta_y, delta_alpha])

        # use motion model to update pose
        pose_t_plus_1 = next_pose(pose_t, delta_pose, noise=True)

        joint_angles = joint_dataset.angles_at_time(timestamp) # theta_neck, theta_head
        
        for (key, distance) in enumerate(lidar_scan):
            if distance > DISTANCE_FAR_AWAY or distance < DISTANCE_TOO_CLOSE:
                lidar_scan[key] = 0.0
            else:
                ray_angle = ANGULAR_RANGE[key] # in radians
                robot_pose = tuple(pose_t_plus_1)
                if done % draw_interval == 0:
                    occupancy_img = draw_lidar_ray(occupancy_img, distance, ray_angle,
                                                robot_pose, joint_angles)

        if done % draw_interval == 0:
            # update log odds for occupied and unoccupied cells
            binary_occupancy = (0 < occupancy_img[:,:,0])
            binary_occupancy = binary_occupancy.astype(int)
            binary_occupancy[binary_occupancy == 1] = -1
            binary_occupancy[binary_occupancy == 0] = 1
            true_vs_false_positive = 80/20
            del_lambda_t = np.multiply(binary_occupancy, np.log(true_vs_false_positive))
            log_odd_map = np.add(log_odd_map, del_lambda_t)

    #plt.figure(figsize=(20, 20))
    #plt.imshow(log_odd_map)
    #plt.title('log odds map')
    #plt.show()

    return pose_t_plus_1, occupancy_img


def trajectory_map(scanset, joint_dataset):
    final_pose, occupancy = move_and_draw_occupancy(scanset, joint_dataset, draw_interval=2000)

    print('robot moved to ({},{},{})'.format(*tuple(final_pose)))

    # draw robot trajectory
    alpha_robot = 0 # radians
    robot_curr_pos = (0, 0) # meters
    robot_start_pos = robot_curr_pos
    robot_start_pos_grid = world_to_grid_location(robot_start_pos)
    radius = 1 # pixels
    color = (255, 0, 0) # BGR
    thickness = 1
    color_robot = (0, 0, 255) # BGR
    
    # draw start point (green point)
    occupancy = cv2.circle(occupancy, world_to_grid_location((0, 0)), radius*3, (0,255,0), thickness*3)

    for idx, lidar_data in enumerate(scanset):
        x_robot, y_robot = robot_curr_pos
        # use delta_pose to update absolute position of robot
        y_robot = y_robot + lidar_data['delta_pose'][1] # meters
        x_robot = x_robot + lidar_data['delta_pose'][0] # meters
        alpha_robot = alpha_robot + lidar_data['delta_pose'][2] # radians
        robot_curr_pos = (x_robot, y_robot)

        robot_curr_pos_grid = world_to_grid_location(robot_curr_pos)


        arrow_end_pt_grid = world_to_grid_location((x_robot + 0.50*np.cos(alpha_robot), 
                                            y_robot + 0.50*np.sin(alpha_robot)))

        # draw the robot's position
        if idx % 300 == 0:
            occupancy = cv2.line(occupancy, robot_start_pos_grid,
                                  robot_curr_pos_grid, color, thickness)
            occupancy = cv2.arrowedLine(occupancy, robot_curr_pos_grid, arrow_end_pt, (60, 76, 231), 1, tipLength = 0.4)
            occupancy = cv2.circle(occupancy, robot_curr_pos_grid, radius*2, color_robot, thickness*2)
            robot_start_pos = (x_robot, y_robot) # to draw trajectory
            robot_start_pos_grid = world_to_grid_location(robot_start_pos)
        

    # draw end point (yellow point)
    occupancy = cv2.circle(occupancy, robot_curr_pos_grid, radius*3, (0, 79, 255), thickness*3)
    occupancy = cv2.arrowedLine(occupancy, robot_curr_pos_grid,
                                arrow_end_pt_grid, (60, 76, 231), 1, tipLength = 0.4)
    return occupancy


# In[ ]:


# draw trajectory map of robot
img = trajectory_map(lidar_dataset.scanset[0:4000], joint_dataset)
plt.figure(figsize=(20, 20))
plt.title('trajectory map')
plt.imshow(upscale(img, percent=100))
plt.show()


# In[ ]:




