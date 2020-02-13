from dataset import LidarDataset

filenumber = str(1)
lidar_dataset = LidarDataset("lidar/train_lidar{}.mat".format(filenumber))
data = lidar_dataset.scanset # array of dictionary {'timestamp', 'delta_pose', 'scan'}
print('total {} scans'.format(len(lidar_dataset.scanset)))
max_x = -1
min_x = 1000000000
max_x_idx = 0
min_x_idx = 0

max_y = -1
min_y = 1000000000
max_y_idx = 0
min_y_idx = 0


max_a = -1
min_a = 1000000000
max_a_idx = 0
min_a_idx = 0


query=[0,500,1500,2000,3000,4000,5000]

pose = (0,0,0)
filename = 'train_lidar{}.csv'.format(filenumber)
with open(filename, 'w') as f:
	for idx, scaninfo in enumerate(data):
		timestamp = scaninfo['timestamp']
		delta_pose = scaninfo['delta_pose']
		x,y,yaw = delta_pose
		x0,y0,yaw0 = pose
		pose = (x+x0,y+y0,yaw+yaw0)
		if idx in query:
			print('pose at {} is '.format(idx), pose)
		if pose[0] > max_x:
			max_x = pose[0]
			max_x_idx = idx
		if pose[0] < min_x:
			min_x = pose[0]
			min_x_idx = idx

		if pose[1] > max_y:
			max_y = pose[1]
			max_y_idx = idx
		if pose[1] < min_y:
			min_y = pose[1]
			min_y_idx = idx

		if pose[2] > max_a:
			max_a = pose[2]
			max_a_idx = idx
		if pose[2] < min_a:
			min_a = pose[2]
			min_a_idx = idx
		
		f.write('{},{},{},{}\n'.format(timestamp, *delta_pose))

print('written to file', filename)
print('max x is {} at index {}'.format(max_x, max_x_idx))
print('min x is {} at index {}'.format(min_x, min_x_idx))


print('max y is {} at index {}'.format(max_y, max_y_idx))
print('min y is {} at index {}'.format(min_y, min_y_idx))


print('max yaw is {} at index {}'.format(max_a, max_a_idx))
print('min yaw is {} at index {}'.format(min_a, min_a_idx))
