from scipy import io
from bisect import bisect_left
import numpy as np

class JointDataset:
    """Wrapper class to load joint angles data
    """
    def __init__(self, filename):
        self.rawdata = io.loadmat(filename)
        key_names_joint = ['ts', 'head_angles']
        joint_data = {kn: self.rawdata[kn] for kn in key_names_joint}
        timestamps = joint_data['ts'][0]
        neck_angles = -joint_data['head_angles'][0]
        head_angles = -joint_data['head_angles'][1]

        self._jointset = []
        self._timestamps = []

        for idx, timestamp in enumerate(timestamps):
            values = {
                'timestamp': timestamp,
                'neck_angle': neck_angles[idx],
                'head_angle': head_angles[idx]
            }
            self._timestamps.append(timestamp)
            self._jointset.append(values)

    def angles_at_time(self, timestamp):
        """
        Returns neck and head angles closest in time to given timestamp.

        returns angles for the smallest timestamp if two are equally close.
        """
        pos = bisect_left(self._timestamps, timestamp)
        if pos == 0:
            values = self._jointset[0]
            return (values['neck_angle'], values['head_angle'])
        if pos == len(self._timestamps):
            values = self._jointset[-1]
            return (values['neck_angle'], values['head_angle'])
        before = self._timestamps[pos - 1]
        after = self._timestamps[pos]

        if after - timestamp < timestamp - before:
            idx = self._timestamps.index(after)
            values = self._jointset[idx]
            return (values['neck_angle'], values['head_angle'])
        else:
            idx = self._timestamps.index(before)
            values = self._jointset[idx]
            return (values['neck_angle'], values['head_angle'])

    def set_jointset(self, jointset):
        self._jointset = jointset

    def set_timestamps(self, timestamps):
        self._timestamps = timestamps

    @property
    def jointset(self):
        return self._jointset

    @property
    def timestamps(self):
        return self._timestamps
    
    


class LidarDataset:
    """Wrapper class to load lidar data
    """
    def __init__(self, filename):
        self.rawdata = io.loadmat(filename)
        self._scanset = []

        # transform data into simpler format
        for idx, val in enumerate(self.rawdata['lidar'][0]):
            timestamp, delta_pose, scan = self._get_current_values(val)
            values = {
                'timestamp': timestamp,
                'delta_pose': delta_pose,
                'scan': scan
            }
            self._scanset.append(values)

    def _get_current_values(self, val):
        scan = val[0][0][1][0]
        delta_pose = val[0][0][2][0]
        timestamp = val[0][0][0][0][0]
        assert len(delta_pose) == 3, \
            'delta_pose needs exactly 3 values, got {} instead'.format(len(delta_pose))
        return timestamp, delta_pose, scan

    def get_scan_values_at(self, idx):
        return self._scanset[idx]

    @property
    def size(self):
        return len(self._scanset)

    @property
    def scanset(self):
        return self._scanset

