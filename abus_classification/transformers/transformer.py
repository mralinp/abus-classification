from abc import ABC, abstractmethod


class Transformer(ABC):
    
    def __call__(self, *args):
        return self.transform(*args)
    
    @abstractmethod
    def transform(self, *args):
        pass
