from heapq import heappush, heappop


class Heap:
    """ Heap, returning minimum element based on python stdlib
    """

    def __init__(self):
        self.data = []

    def push(self, a):
        heappush(self.data, a)

    def min(self):
        return self.data[0]

    def pop(self):
        return heappop(self.data)

    def empty(self):
        return len(self.data)
