from abc import ABC, abstractmethod


class Downloader(ABC):
        
    @abstractmethod
    def download(self, url, output_path):
        pass
    
    @abstractmethod
    def save(self):
        pass
