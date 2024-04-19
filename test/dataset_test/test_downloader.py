from abus_classification.datasets.google_drive_downloader import GoogleDriveDownloader


url = "1NsYIqatNp2D4yCj8PwZ8g9IbAuAdPX6F"

def test_google_cloud_downloader():
    downloader = GoogleDriveDownloader()
    downloader.download(url, "./dataset/tdsc.zip")