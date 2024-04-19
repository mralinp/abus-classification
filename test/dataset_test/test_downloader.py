from abus_classification.datasets.google_drive_downloader import GoogleDriveDownloader


google_drive_10mb_test_file_id = "1Fn6psOjknovxmShESRYpaxDAbb9txvH7"

def test_google_cloud_downloader():
    downloader = GoogleDriveDownloader()
    downloader.download(google_drive_10mb_test_file_id, "./data/tdsc/labels.csv")