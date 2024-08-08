import yaml
import argparse
import os
import subprocess
import uuid
import multiprocessing as mp
import time

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

class encode_segment:
    def __init__(self, framerate, file_fullpath, out_path, segment_start, segment_end, ffmpeg_video_string, preset,
                 filename, encode_job):
        self.framerate = framerate
        self.file_fullpath = file_fullpath
        self.out_path = out_path
        self.segment_start = frame_to_timestamp(segment_start, framerate)
        self.segment_end = frame_to_timestamp(segment_end, framerate)
        self.ffmpeg_video_string = ffmpeg_video_string
        self.preset = preset
        self.filename = filename
        self.uuid = uuid.uuid4()
        self.encode_job = encode_job

    def encode(self, hostname, current_user):
        cmd = (
            f"ssh -t {current_user}@{hostname} 'ffmpeg -ss {self.segment_start} -to {self.segment_end} -i "
            f"{self.file_fullpath} {self.ffmpeg_video_string} {self.out_path}/{self.preset['name']}/{self.segment_start}"
            f"-{self.segment_end}_{self.filename}'"
        )
        print(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return stdout, stderr, process.returncode

class encode_job:
    def __init__(self, proper_name, input_file, preset, out_path, filename):
        self.input_file = input_file
        self.proper_name = proper_name
        self.frames_per_segment = 100
        self.num_segments, self.framerate = get_segments_and_framerate(self.input_file, self.frames_per_segment)
        self.preset = preset
        self.out_path = out_path
        self.filename = filename
        self.segments_completed = 0

        if not os.path.isdir(f"{self.out_path}/{preset['name']}/"):
            os.makedirs(f"{self.out_path}/{preset['name']}/")

    def tally_completed_segments(self):
        self.segments_completed += 1
        print(f"HEY IDIOT LOOK HERE {self.segments_completed}")
        if self.segments_completed == self.num_segments:
            print("do muxing here...")

    def create_segment_encode_list(self):
        segment_list = []
        for x in range(self.num_segments):
            segment_list += [encode_segment(framerate=self.framerate, file_fullpath=self.input_file,
                out_path=self.out_path, ffmpeg_video_string=self.preset['ffmpeg_video_string'],
                segment_start = (x * self.frames_per_segment) + 1,
                segment_end = (x + 1) * self.frames_per_segment, preset=self.preset, filename=self.filename,
                encode_job=self)]
        return segment_list

class encode_worker(mp.Process):
    def __init__(self, hostname, current_user, segment_queue, results_queue):
        super().__init__()
        self.hostname = hostname
        self.current_user = current_user
        self.segment_queue = segment_queue
        self.results_queue = results_queue
        self.running = mp.Value('b', False)

    def run(self):
        while True:
            while not self.running.value:
                if not self.segment_queue.empty():
                    segment_to_encode = self.segment_queue.get()
                    stdout, stderr, returncode = self.execute_encode(segment_to_encode=segment_to_encode)
                    self.results_queue.put((segment_to_encode, returncode, stdout, stderr))
                    self.running.value = False
            time.sleep(1)

    def is_running(self):
        return self.running.value

    def execute_encode(self, segment_to_encode):
        self.running.value = True
        stdout, stderr, returncode = segment_to_encode.encode(self.hostname, self.current_user)
        return stdout, stderr, returncode

def job_handler(segment_list, worker_list, segment_queue, results_queue):
    while len(segment_list) > 0:
        segment_index = 0
        for worker in worker_list:
            if not results_queue.empty():
                results = results_queue.get()
                print(results)
                for segment in segment_list:
                    if segment.uuid == results[0].uuid:
                        segment.encode_job.tally_completed_segments()
                        segment_list.remove(segment)
                        break

            elif not worker.is_running():
                print(worker.is_running())
                segment_queue.put(segment_list[segment_index])
                print(f"Running {segment_list[segment_index]} on {worker.hostname}")
                segment_index += 1

            time.sleep(1)




def main():
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

    segment_queue = mp.Queue()
    results_queue = mp.Queue()

    for worker in config['nodes']:
        print(worker['hostname'])
        worker_list += [encode_worker(hostname=worker['hostname'], current_user=current_user,
                        segment_queue=segment_queue, results_queue=results_queue)]
        worker_list[len(worker_list)-1].start()

    job_list = sorted(job_list, key=lambda x: x.proper_name)
    job_segment_list = []
    for jobs in job_list:
        print(jobs.proper_name)
        job_segment_list += jobs.create_segment_encode_list()

    job_handler(job_segment_list, worker_list, segment_queue, results_queue)

if __name__ == "__main__":
    main()