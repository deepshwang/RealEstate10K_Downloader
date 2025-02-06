import os
import sys
import glob

from skimage import io
from skimage.transform import resize
import yt_dlp

from multiprocessing import Pool
import shutil
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

failure_log_lock = Lock()


def download_and_process(data, vid, num_videos, mode, output_root):
    videoname = data.url.split("=")[-1]
    print(f"[INFO] Downloading {vid + 1}/{num_videos}: {videoname} ...")
    try:
        # pytube is unstable, use yt_dlp instead
        ydl_opts = {
            "format": "bestvideo[height<=480]",
            "outtmpl": f"./{videoname}",
            "cookiefile": "./cookies.txt",
        }

        # Initialize yt_dlp and download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([data.url])
    except Exception:
        with failure_log_lock:
            failure_log = open("failed_videos_" + mode + ".txt", "a")
            failure_log.writelines(data.url + "\n")
            failure_log.close()
        return

    with Pool(processes=16) as pool:
        pool.map(
            wrap_process,
            [(data, seq_id, videoname, output_root) for seq_id in range(len(data))],
        )
    os.system("rm " + videoname)  # remove videos


class Data:
    def __init__(self, url, seqname, list_timestamps):
        self.url = url
        self.list_seqnames = []
        self.list_list_timestamps = []

        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def add(self, seqname, list_timestamps):
        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def __len__(self):
        return len(self.list_seqnames)


def process(data, seq_id, videoname, output_root):
    seqname = data.list_seqnames[seq_id]
    if not os.path.exists(output_root + seqname):
        os.makedirs(output_root + seqname)
    else:
        print(f"[WARNING] The output dir {output_root + seqname} has already existed.")
        return False

    list_str_timestamps = []
    for timestamp in data.list_list_timestamps[seq_id]:
        timestamp = int(timestamp / 1000)
        str_hour = str(int(timestamp / 3600000)).zfill(2)
        str_min = str(int(int(timestamp % 3600000) / 60000)).zfill(2)
        str_sec = str(int(int(int(timestamp % 3600000) % 60000) / 1000)).zfill(2)
        str_mill = str(int(int(int(timestamp % 3600000) % 60000) % 1000)).zfill(3)
        _str_timestamp = str_hour + ":" + str_min + ":" + str_sec + "." + str_mill
        list_str_timestamps.append(_str_timestamp)

    # extract frames from a video
    for idx, str_timestamp in enumerate(list_str_timestamps):
        command = (
            "ffmpeg -loglevel error -ss "
            + str_timestamp
            + " -i "
            + videoname
            + " -vframes 1 -f image2 "
            + output_root
            + seqname
            + "/"
            + str(data.list_list_timestamps[seq_id][idx])
            + ".png"
        )
        try:
            os.system(command)
        except Exception as err:
            print(f"[ERROR] Failed to process {data.url}: {err}")
            shutil.rmtree(output_root + seqname)  # delete the output dir
            return True

    png_list = glob.glob(output_root + "/" + seqname + "/*.png")

    for pngname in png_list:
        image = io.imread(pngname)
        if int(image.shape[1] / 2) < 500:
            break
        image = resize(
            image,
            (int(image.shape[0] / 2), int(image.shape[1] / 2)),
            anti_aliasing=True,
        )
        image = (image * 255).astype("uint8")
    return False


def wrap_process(list_args):
    return process(*list_args)


class DataDownloader:
    def __init__(self, dataroot, mode="test", split_idx=0, num_splits=1, num_threads=4):
        print("[INFO] Loading data list ... ", end="")
        self.dataroot = dataroot
        self.list_seqnames = sorted(glob.glob(dataroot + "/*.txt"))
        # Split the sequence list into num_splits parts and take split_idx-th part
        if split_idx is not None and num_splits is not None:
            total_seqs = len(self.list_seqnames)
            split_size = total_seqs // num_splits
            start_idx = split_idx * split_size
            end_idx = start_idx + split_size if split_idx < num_splits - 1 else total_seqs
            self.list_seqnames = self.list_seqnames[start_idx:end_idx]

        self.output_root = "./dataset/" + mode + "/"
        self.mode = mode
        self.num_threads = num_threads
        os.makedirs(self.output_root, exist_ok=True)

        self.list_data = []
        for txt_file in self.list_seqnames:
            dir_name = txt_file.split("/")[-1]
            seq_name = dir_name.split(".")[0]

            # extract info from txt
            seq_file = open(txt_file, "r")
            lines = seq_file.readlines()
            youtube_url = ""
            list_timestamps = []
            for idx, line in enumerate(lines):
                if idx == 0:
                    youtube_url = line.strip()
                else:
                    timestamp = int(line.split(" ")[0])
                    list_timestamps.append(timestamp)
            seq_file.close()

            isRegistered = False
            for i in range(len(self.list_data)):
                if youtube_url == self.list_data[i].url:
                    isRegistered = True
                    self.list_data[i].add(seq_name, list_timestamps)
                else:
                    pass

            if not isRegistered:
                self.list_data.append(Data(youtube_url, seq_name, list_timestamps))

        # self.list_data.reverse()
        print(f"[INFO] {len(self.list_data)} movies are used in {self.mode} mode")

    def run(self):
        num_videos = len(self.list_data)
        print(f"[INFO] Start downloading {num_videos} movies")
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(
                    download_and_process,

                    data,
                    vid,
                    num_videos,
                    self.mode,
                    self.output_root,
                )
                for vid, data in enumerate(self.list_data)
            ]
            for future in futures:
                future.result()
        print("[INFO] Done!")

    def show(self):
        print("########################################")
        global_count = 0
        for data in self.list_data:
            print(" URL : {}".format(data.url))

            for idx in range(len(data)):
                print(" SEQ_{} : {}".format(idx, data.list_seqnames[idx]))
                print(" LEN_{} : {}".format(idx, len(data.list_list_timestamps[idx])))
                global_count = global_count + 1
            print("----------------------------------------")

        print("TOTAL : {} sequnces".format(global_count))
        print("########################################")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download and process RealEstate10K dataset')
    parser.add_argument('mode', choices=['test', 'train'], help='Dataset split to process (test or train)')
    parser.add_argument('--split_idx', type=int, default=None, help='Index of current split')
    parser.add_argument('--num_splits', type=int, default=None, help='Total number of splits')
    parser.add_argument('--num_threads', type=int, default=8, help='Number of download threads to use')
    args = parser.parse_args()

    mode = args.mode
    split_idx = args.split_idx
    num_splits = args.num_splits
    num_threads = args.num_threads

    dataroot = "./RealEstate10K/" + mode
    downloader = DataDownloader(dataroot, mode, split_idx, num_splits, num_threads)


    downloader.show()
    downloader.run()
