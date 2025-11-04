import random
from collections import defaultdict


class Cache:
    def __init__(
        self,
    ):
        self.data = defaultdict(list)
    
    def add(self, k, v):
        self.data[k].append(v)
    
    def get(self, k):
        return random.choice(self.data[k])
    
    def len(self, k):
        return len(self.data[k])