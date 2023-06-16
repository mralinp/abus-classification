import os                                                                                                                                                                                                          
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import wget
import os



class Downloader:
    
    def __init__(self, dataset_name='iipl-abus') -> None:
        self.dataset_name = dataset_name
        load_dotenv(Path("../.env"))
        self.token = os.getenv("API_TOKEN")
                     
    def download(self):
        path = f"../data/{self.dataset_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        wget.download(f"https://iipl.ir/datasets/{self.dataset_name}?token={self.token}", path)
    
    def extract(self):
        pass
    
    def purge(self):
        pass
    
    
if __name__ == "__main__":
    downloader = Downloader('tdsc')
