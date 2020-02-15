import cv2

from dataset import JointDataset, LidarDataset
from particlefilter import occupancy_map, trajectory_map
from particlefilter import upscale

lidar_dataset = LidarDataset("lidar/train_lidar4.mat")
joint_dataset = JointDataset("joint/train_joint4.mat")

# idx 8078 has a max distance close to 30 meters
# values = lidar_dataset.get_scan_values_at(0)
# print(values['delta_pose'])
# show_lidar_scan(values)

# draw a 2d occupancy grid map using 
# img = occupancy_map(values)
# cv2.imshow('occupancy map', img)
# cv2.waitKey(0)

"""
l5235 = lidar_dataset.scanset[5235]
l5235['delta_pose'][0] += 10
l5235['delta_pose'][1] += 10

l5236 = lidar_dataset.scanset[5236]
l5236['delta_pose'][0] += 8
l5236['delta_pose'][1] += -5
sample_lidardataset1 = lidar_dataset.scanset[5230:5235]
sample_lidardataset1.append(l5235)
sample_lidardataset1.append(l5236)

# draw trajectory map of robot
img = trajectory_map(sample_lidardataset1, joint_dataset.jointset)
cv2.imshow('trajectory map', img)
cv2.waitKey(0)
"""


"""
l5235 = lidar_dataset.scanset[5235]
l5235['delta_pose'][0] += 10
l5235['delta_pose'][1] += 10
sample_lidardataset = [lidar_dataset.scanset[3000], l5235]
img2 = trajectory_map(sample_lidardataset, joint_dataset.jointset)

cv2.imshow('trajectory map', img2)
cv2.waitKey(0)
"""
