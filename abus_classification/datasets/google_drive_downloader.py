import os
import zipfile

import gdown

from .downloader import Downloader


class GoogleDriveDownloader(Downloader):
    
    def __init__(self, credentials=None):
        self.credentials = credentials
        self.file_name = None
        
    def download(self, file_url, output_path):
        gdown.download(file_url, output=output_path)
        self.file_name = output_path

    def save(self):
        if self.file_name:
            with zipfile.ZipFile(self.file_name) as zip_ref:
                zip_ref.extractall(os.path.dirname(self.file_name))