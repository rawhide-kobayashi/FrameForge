import yaml
import argparse
import os
import subprocess
import uuid
import multiprocessing as mp
import time
import re
import math
import shlex

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
    print(framerate)
    framerate_num, framerate_denom = map(float, framerate.stdout.strip().split('/'))
    duration_seconds = subprocess.run(duration_seconds_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration_seconds = float(duration_seconds.stdout.strip())
    return math.ceil(int((framerate_num / framerate_denom) * duration_seconds)  / frames_per_segment), \
        float(framerate_num / framerate_denom), round(duration_seconds * (framerate_num / framerate_denom))

def frame_to_timestamp(frame, framerate):
    return round(float(frame / framerate), 3)

def seconds_to_frames(seconds, framerate):
    return round(framerate * seconds)

class progress_bar(mp.Process):
    def __init__(self):
        super().__init__()

class mux_worker(mp.Process):
    def __init__(self, preset, out_path, filename, completed_segment_filename_list, additional_content):
        super().__init__()
        self.preset = preset
        self.out_path = out_path
        self.filename = filename
        self.completed_segment_filename_list = completed_segment_filename_list
        self.additional_content = additional_content

    def run(self):
        with open(f"{self.out_path}/{self.preset['name']}/temp/{self.filename}/mux.txt", 'w') as mux:
            for x in self.completed_segment_filename_list:
                mux.write(f"file '{x}'\n")

        ffmpeg_concat_string = (f"ffmpeg -f concat -safe -0 -i {self.out_path}/{self.preset['name']}/temp/"
                                f"{self.filename}/mux.txt -c copy {self.out_path}/{self.preset['name']}/temp/"
                                f"{self.filename}/{shlex.quote(self.filename)} -y")
        print(ffmpeg_concat_string)
        process = subprocess.Popen(ffmpeg_concat_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stdout, stderr, process.returncode)
        os.remove(f"{self.out_path}/{self.preset['name']}/temp/{self.filename}/mux.txt")
        for x in self.additional_content:
            print(x)
        self.close()

class encode_segment:
    def __init__(self, framerate, file_fullpath, out_path, segment_start, segment_end, ffmpeg_video_string, preset,
                 filename, encode_job, num_frames):
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
        self.assigned = False
        self.num_frames = num_frames
        self.file_output_fstring = (f"{self.out_path}/{self.preset['name']}/temp/{filename}/{self.segment_start}-"
                                    f"{self.segment_end}_{self.filename}")

    def check_if_exists(self):
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            '-show_entries', 'format=duration', shlex.quote(self.file_output_fstring)
        ]
        if os.path.isfile(self.file_output_fstring):
            duration = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if duration.returncode == 0:
                duration = round(float(duration.stdout.strip()), 3)
                if seconds_to_frames(duration, self.framerate) == self.num_frames:
                    self.encode_job.tally_completed_segments(self.file_output_fstring)
                    return True
                else:
                    print(f"{self.file_output_fstring} wrong duration {duration}")
                    os.remove(self.file_output_fstring)
                    return False
            else:
                print(f"{self.file_output_fstring} bad returncode {duration.returncode}")
                os.remove(self.file_output_fstring)
                return False
        else:
            return False

    def encode(self, hostname, current_user):
        cmd = (
            f"ssh -t {current_user}@{hostname} 'ffmpeg -ss {self.segment_start} -to {self.segment_end} -i "
            f"{self.file_fullpath} {self.ffmpeg_video_string} {shlex.quote(self.file_output_fstring)}'"
        )
        print(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return stdout, stderr, process.returncode

class encode_job:
    def __init__(self, proper_name, input_file, preset, out_path, filename, additional_content):
        self.input_file = input_file
        self.proper_name = proper_name
        self.frames_per_segment = 2000
        self.num_segments, self.framerate, self.frames_total = get_segments_and_framerate(self.input_file,
            self.frames_per_segment)
        self.preset = preset
        self.out_path = out_path
        self.filename = filename
        self.segments_completed = 0
        self.completed_segment_filename_list = []
        self.additional_content = additional_content

        if not os.path.isdir(f"{self.out_path}/{self.preset['name']}/temp/{self.filename}/"):
            os.makedirs(f"{self.out_path}/{self.preset['name']}/temp/{self.filename}/")

    def tally_completed_segments(self, filename):
        self.segments_completed += 1
        self.completed_segment_filename_list.append(filename)
        print(self.segments_completed)
        print(self.proper_name)
        if self.segments_completed == self.num_segments:
            print("do muxing here...")
            self.completed_segment_filename_list = sorted(self.completed_segment_filename_list,
                key=lambda x: float(re.search(r'/(\d+\.\d+)-', x).group(1)))
            print(self.completed_segment_filename_list)
            mux = mux_worker(self.preset, self.out_path, self.filename, self.completed_segment_filename_list,
                             self.additional_content)
            mux.start()

    def create_segment_encode_list(self):
        segment_list = []
        frames_assigned = 0
        last_segment_compensation = self.frames_per_segment
        for x in range(self.num_segments):
            if self.frames_total - frames_assigned < self.frames_per_segment:
                last_segment_compensation = self.frames_total - frames_assigned
            segment_list += [encode_segment(framerate=self.framerate, file_fullpath=self.input_file,
                out_path=self.out_path, ffmpeg_video_string=self.preset['ffmpeg_video_string'],
                segment_start = x * self.frames_per_segment,
                segment_end = (x + 1) * self.frames_per_segment, preset=self.preset, filename=self.filename,
                encode_job=self, num_frames=last_segment_compensation)]
            frames_assigned += self.frames_per_segment
        return segment_list

class encode_worker(mp.Process):
    def __init__(self, hostname, current_user, results_queue):
        super().__init__()
        self.hostname = hostname
        self.current_user = current_user
        self.current_segment = mp.Manager().Value(typecode=None, value=None)
        self.results_queue = results_queue
        self.is_running = mp.Value('b', False)

    def run(self):
        while True:
            if not self.current_segment.value is None and self.is_running.value == True:
                stdout, stderr, returncode = self.execute_encode(segment_to_encode=self.current_segment.value)
                self.results_queue.put((self.current_segment.value, returncode, stdout, stderr))
                self.is_running.value = False
            time.sleep(1)

    def execute_encode(self, segment_to_encode):
        stdout, stderr, returncode = segment_to_encode.encode(self.hostname, self.current_user)
        return stdout, stderr, returncode

def job_handler(segment_list, worker_list, results_queue):

    segment_list[:] = [segment for segment in segment_list if not segment.check_if_exists()]

    segment_index = 0
    while len(segment_list) > 0:
        while segment_list[segment_index].assigned:
            segment_index += 1

        for worker in worker_list:
            if worker.current_segment.value is None or worker.is_running.value == False:
                if not results_queue.empty():
                    results = results_queue.get()
                    print(results)
                    for segment in segment_list:
                        if segment.uuid == results[0].uuid:
                            segment.encode_job.tally_completed_segments(segment.file_output_fstring)
                            segment_list.remove(segment)
                            break
                    worker.current_segment.value = None

                else:
                    worker.current_segment.value = segment_list[segment_index]
                    worker.is_running.value = True
                    print(f"Running {segment_list[segment_index]} on {worker.hostname}")
                    segment_list[segment_index].assigned = True
                    segment_index += 1

        segment_index = 0
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
                out_path=args.out_path, filename=file, additional_content=config['additional_content'])]

    results_queue = mp.Queue()

    job_list = sorted(job_list, key=lambda x: x.proper_name)
    job_segment_list = []
    for jobs in job_list:
        print(jobs.proper_name)
        job_segment_list += jobs.create_segment_encode_list()

    for worker in config['nodes']:
        print(worker['hostname'])
        worker_list += [encode_worker(hostname=worker['hostname'], current_user=current_user,
                        results_queue=results_queue)]
        worker_list[len(worker_list)-1].start()

    job_handler(job_segment_list, worker_list, results_queue)

if __name__ == "__main__":
    main()