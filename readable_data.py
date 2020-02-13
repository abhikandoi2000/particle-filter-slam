from dataset import JointDataset

joint_dataset = JointDataset("joint/train_joint0.mat")
data = joint_dataset.jointset # array of dictionary {'timestamp', 'neck_angle', 'head_angle'}

max_neck_angle = -1
max_head_angle = -1
min_neck_angle = 100
min_head_angle = 100

filename = 'train_joint0.csv'
with open(filename, 'w') as f:
	for jointinfo in data:
		t = jointinfo['timestamp']
		neck_angle = jointinfo['neck_angle']
		head_angle = jointinfo['head_angle']
		if neck_angle > max_neck_angle:
			max_neck_angle = neck_angle
		if head_angle > max_head_angle:
			max_head_angle = head_angle
		if neck_angle < min_neck_angle:
			min_neck_angle = neck_angle
		if head_angle < min_head_angle:
			min_head_angle = head_angle
		f.write('{},{},{}\n'.format(t, neck_angle, head_angle))

print('written to file', filename)
print('min, max neck angles are {} and {}'.format(min_neck_angle, max_neck_angle))
print('min, max head angles are {} and {}'.format(min_head_angle, max_head_angle))