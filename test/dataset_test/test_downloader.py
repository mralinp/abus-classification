from abus_classification.datasets.google_drive_downloader import GoogleDriveDownloader


file_id = "1NsYIqatNp2D4yCj8PwZ8g9IbAuAdPX6F"

def test_google_cloud_downloader():
    downloader = GoogleDriveDownloader()
    downloader.download(file_id, "./data/tdsc.zip")