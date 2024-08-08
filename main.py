import threading

import yaml
import argparse
import os
import subprocess

from math import ceil

import asyncio

import multiprocessing as mp

from threading import Lock

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def get_segments_and_framerate(file, frames_per_segment):
    framerate_command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        '-show_entries', 'stream=avg_frame_rate', file
    ]

    duration_seconds_command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        '-show_entries', 'format=duration', file
    ]
    framerate = subprocess.run(framerate_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    framerate_num, framerate_denom = map(float, framerate.stdout.strip().split('/'))
    duration_seconds = subprocess.run(duration_seconds_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration_seconds = float(duration_seconds.stdout.strip())
    return int(((framerate_num / framerate_denom) * duration_seconds)  / frames_per_segment), \
        float(framerate_num / framerate_denom)

def frame_to_timestamp(frame, framerate):
    return float(frame / framerate)


class encode_watcher:
    def __init__(self, file_fullpath, in_path, out_path, num_segments):
        self.file_fullpath = file_fullpath
        self.in_path = in_path
        self.out_path = out_path
        self.num_segments = num_segments
        self.completed_segments = 0
        self.lock = Lock()

    def add_segment_completion(self):
        with self.lock:
            self.completed_segments = 0
            if self.completed_segments == self.num_segments:
                self.mux()

    def mux(self):
        print(f"Muxing {self.file_fullpath}")

class encode_segment:
    def __init__(self, framerate, file_fullpath, out_path, segment_start, segment_end, ffmpeg_video_string, preset,
                 filename):
        self.framerate = framerate
        self.file_fullpath = file_fullpath
        self.out_path = out_path
        self.segment_start = frame_to_timestamp(segment_start, framerate)
        self.segment_end = frame_to_timestamp(segment_end, framerate)
        self.ffmpeg_video_string = ffmpeg_video_string
        self.preset = preset
        self.filename = filename

    async def encode(self, hostname, current_user):
        #subprocess.run(f"ssh {current_user}@{hostname} '{ffmpeg_video_string}'" stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #print(f"ssh {current_user}@{hostname} 'ffmpeg -ss {self.segment_start} -to {self.segment_end} \
        #-i {self.file_fullpath} {self.ffmpeg_video_string} {self.out_path}/{self.preset['name']}/{self.filename}'")
        cmd = (
            f"ssh {current_user}@{hostname} 'ffmpeg -ss {self.segment_start} -to {self.segment_end} -i "
            f"{self.file_fullpath} {self.ffmpeg_video_string} {self.out_path}/{self.preset['name']}/{self.segment_start}"
            f"-{self.segment_end}_{self.filename}'"
        )
        print(cmd)
        process = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await process.communicate()
        return stdout, stderr, process.returncode

class encode_job:
    def __init__(self, proper_name, input_file, preset, out_path, filename):
        self.input_file = input_file
        self.proper_name = proper_name
        self.frames_per_segment = 2000
        self.num_segments, self.framerate = get_segments_and_framerate(self.input_file, self.frames_per_segment)
        self.preset = preset
        self.out_path = out_path
        self.filename = filename

        if not os.path.isdir(f"{self.out_path}/{preset['name']}/"):
            os.makedirs(f"{self.out_path}/{preset['name']}/")

    def create_segment_encode_list(self):
        segment_list = []
        for x in range(self.num_segments):
            segment_list += [encode_segment(framerate=self.framerate, file_fullpath=self.input_file,
                out_path=self.out_path, ffmpeg_video_string=self.preset['ffmpeg_video_string'],
                segment_start = (x * self.frames_per_segment) + 1,
                segment_end = (x + 1) * self.frames_per_segment, preset=self.preset, filename=self.filename)]
        return segment_list

class encode_worker:
    def __init__(self, hostname, current_user):
        self.hostname = hostname
        self.current_user = current_user
        self.running = False

    async def is_running(self):
        return self.running

    async def execute_encode(self, segment_to_encode):
        self.running = True
        stdout, stderr, returncode = await segment_to_encode.encode(self.hostname, self.current_user)
        self.running = False
        return stdout, stderr, returncode

async def job_handler(segment_list, worker_list):
    while len(segment_list) > 0:
        segment_index = 0
        for worker in worker_list:
            if await worker.is_running() == False and segment_index < len(segment_list):
                stdout, stderr, returncode = await asyncio.create_task(worker.execute_encode(segment_list[segment_index]))
                print(f"Running {segment_list[segment_index]} on {worker.hostname}")
                print(stdout, stderr, returncode)
                if returncode == 0:
                    segment_list.pop(segment_index)
                    segment_index += 1

        await asyncio.sleep(1)




async def main():
    current_user = os.getlogin()

    parser = argparse.ArgumentParser(description="Load a YAML configuration file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument('--in_path', type=str, required=True, help="Path containing input video files.")
    parser.add_argument('--out_path', type=str, required=True, help="Path to output video files.")

    args = parser.parse_args()

    config = load_config(args.config)
    print("Loaded configuration:")
    print(config)
    print(args.in_path)
    print(args.out_path)

    job_list = []
    worker_list = []

    for file in os.listdir(args.in_path):
        file_fullpath = os.path.join(args.in_path, file)
        print(file_fullpath)
        for preset in config['presets']:
            print(preset['name'])
            print(preset['ffmpeg_video_string'])

            job_list += [encode_job(proper_name=f"{preset['name']}_{file}", input_file=file_fullpath, preset=preset,
                out_path=args.out_path, filename=file)]

    for worker in config['nodes']:
        print(worker['hostname'])
        worker_list += [encode_worker(hostname=worker['hostname'], current_user=current_user)]

    job_list = sorted(job_list, key=lambda x: x.proper_name)
    for jobs in job_list:
        print(jobs.proper_name)
        job_segment_list = jobs.create_segment_encode_list()

        await job_handler(job_segment_list, worker_list)

if __name__ == "__main__":
    asyncio.run(main())