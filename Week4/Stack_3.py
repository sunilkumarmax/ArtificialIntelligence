"""Stack class"""
class Stack(object):
    """A Stack class"""

    def __init__(self):
        self.items = []

    def is_empty(self):
        """Is empty function"""
        return self.items == []

    def insert(self, item):
        """Push function"""
        self.items.append(item)
        return

    def remove(self):
        """Dequeue function"""
        return self.items.pop()

    def size(self):
        """Size of the queue class"""
        return len(self.items)
