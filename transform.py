import numpy as np

# yaw
def Rz(alpha):
	return np.matrix([[np.cos(alpha), -np.sin(alpha), 0],
					  [np.sin(alpha),  np.cos(alpha), 0],
					  [0,              0,             1]])

# roll
def Ry(beta):
	return np.matrix([[np.cos(beta),  0,  np.sin(beta)],
					  [0,             1,             0],
					  [-np.sin(beta), 0,  np.cos(beta)]])

# pitch
def Rx(gamma):
	return np.matrix([[1,  0,                          0],
					  [0,  np.cos(gamma), -np.sin(gamma)],
					  [0,  np.sin(gamma),  np.cos(gamma)]])


# translation is tuple (x, y, z) in meters
# angles is tuple (yaw, roll, pitch) in radians
def T(translation, angles):
	x, y, z = translation
	alpha, beta, gamma = angles

	rot = np.matmul(Ry(beta), Rx(gamma))
	rot = np.matmul(Rz(alpha), rot)

	translation = np.array([[x],[y],[z]])
	tranformation = np.hstack((rot, translation))
	lastrow = np.matrix([0, 0, 0, 1])
	tranformation = np.vstack((tranformation, lastrow))
	return tranformation