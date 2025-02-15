import os
from zipfile import ZipFile
import gdown

from abus_classification.datasets.downloader import Downloader


class GoogleDriveDownloader(Downloader):
    
    def __init__(self, credentials=None):
        self.credentials = credentials
        self.file_name = None
        
    def download(self, file_id, output_path):
        gdown.download(id=file_id, output=output_path)
        self.file_name = output_path

    def save(self):
        if self.file_name:
            with ZipFile(self.file_name) as zip_ref:
                zip_ref.extractall(os.path.dirname(self.file_name))