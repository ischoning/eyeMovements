class Sample(object):
    def __init__(self, df):
        self.df = df
        self.current_i = 0

    def process(self):
        """
        move preprocessing of data here
        """
        return None

    def update_i(self):
        self.current_i += 1

    def current_i(self):
        return self.current_i

    def next_i(self):
        if self.current_i < len(self.df) - 2:
            return self.current_i + 1
        else:
            raise Exception('Sample index out of range. Reached end of dataset.')

    def prev_i(self):
        if self.current_i != 0:
            return self.current_i - 1
        else:
            raise Exception('Sample index out of range. Index at pos 0.')

    def get_features(self, j):
        d_l = self.df.d_l[j]
        d_r = self.df.d_r[j]
        vel_l = self.df.vel_l[j]
        vel_r = self.df.vel_r[j]
        accel_l = self.df.accel_l[j]
        accel_r = self.df.accel_r[j]
        pupil_l = self.df.left_pupil_measure1[j]
        pupil_r = self.df.right_pupil_measure1[j]
        return {'Left':{'d':d_l, 'vel':vel_l, 'accel':accel_l, 'pupil':pupil_l},
                'Right':{'d':d_r, 'vel':vel_r, 'accel':accel_r, 'pupil':pupil_r}}

    def get_window(self, i):
        """
        Keeps track of data in 3 sample window: current, previous, and next.
        :returns three lists of features: previous, current, and next
        """
        self.current_i = i

        if self.current_i > 0:
            prev = self.get_features(self.prev_i())
        else:
            prev = {}
        if self.current_i == len(self.df) - 1:
            next = {}
        else:
            next = self.get_features(self.next_i())
        current = self.get_features(self.current_i)

        return prev, current, next