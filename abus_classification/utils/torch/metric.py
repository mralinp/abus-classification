from abc import ABC, abstractmethod


class Metric(ABC):
    
    def __call__(self, *args, **kwargs):
        return self.calculate(*args, *kwargs)
    
    @abstractmethod
    def calculate(self, *args, **kwargs):
        pass