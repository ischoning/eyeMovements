###############################################################################
# Event Detection Algorithm Suite
#  Copyright (C) 2012 Gian Perrone (http://github.com/gian)
#
#  Permission to use, copy, modify, and distribute this software and its
#  documentation for any purpose and without fee is hereby granted,
#  provided that the above copyright notice appear in all copies and that
#  both the copyright notice and this permission notice and warranty
#  disclaimer appear in supporting documentation, and that the name of
#  the above copyright holders, or their entities, not be used in
#  advertising or publicity pertaining to distribution of the software
#  without specific, written prior permission.
#
#  The above copyright holders disclaim all warranties with regard to
#  this software, including all implied warranties of merchantability and
#  fitness. In no event shall the above copyright holders be liable for
#  any special, indirect or consequential damages or any damages
#  whatsoever resulting from loss of use, data or profits, whether in an
#  action of contract, negligence or other tortious action, arising out
#  of or in connection with the use or performance of this software.
###############################################################################

class EventStream(object):
    """The base type for all event detection providers."""

    def __init__(self, sampleStream):
        """Initialise an event detector with an input iterator"""

        self.input = sampleStream

    def __iter__(self):
        return self

    def next(self):
        """Event detectors should override the next method."""
        raise StopIteration

    def centroid(self, window):
        """Compute a centroid for a window of points."""
        xs = 0
        ys = 0

        if len(window) == 0:
            raise StopIteration

        for p in window:
            xs = xs + p.x
            ys = ys + p.y

        xc = round(xs / float(len(window)))
        yc = round(ys / float(len(window)))

        pc = window[0]
        pc.x = xc
        pc.y = yc

        return pc


class DetectorEvent(object):
    def __init__(self):
        self.type = "none"


class EFixation(DetectorEvent):
    def __init__(self, center, length, start, end):
        self.type = "fixation"
        self.center = center
        self.length = length
        self.start = start
        self.end = end

    def __str__(self):
        return "Fixation at (%d,%d) of %d samples, starting at sample %d" % (
        self.center.x, self.center.y, self.length, self.start.index)


class ESaccade(DetectorEvent):
    def __init__(self, length, start, end):
        self.type = "saccade"
        self.length = length
        self.start = start
        self.end = end

    def __str__(self):
        return "Saccade of %d samples, (%d,%d) -> (%d,%d)" % (
        self.length, self.start.x, self.start.y, self.end.x, self.end.y)