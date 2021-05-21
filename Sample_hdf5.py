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
        d = self.df.d[j]
        vel = self.df.vel[j]
        accel = self.df.accel[j]
        pupil = self.df.pupil_measure1[j]
        return {'d':d, 'vel':vel, 'accel':accel, 'pupil':pupil}

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