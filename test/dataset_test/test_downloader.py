from abus_classification.datasets.google_drive_downloader import GoogleDriveDownloader


file_id = "0B1MVW1mFO2zmdGhyaUJESWROQkE"

def test_google_cloud_downloader():
    downloader = GoogleDriveDownloader()
    downloader.download(file_id, "./100mb-test-file.zip")