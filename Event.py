class Event(object):
    def __init__(self, df, event):
        self.event = event
        self.df = df

    def get_window(self):
        start = self.df.time(i)
        end = self.df.time(j)