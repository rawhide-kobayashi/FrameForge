import yaml
import argparse
import os
import subprocess
import uuid
import multiprocessing as mp
import time
import re
import math
from asciimatics.screen import ManagedScreen
from asciimatics.widgets import Frame, Layout, Label, Text, TextBox, VerticalDivider, Divider
import tqdm
import select

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
    return math.ceil(int((framerate_num / framerate_denom) * duration_seconds)  / frames_per_segment), \
        float(framerate_num / framerate_denom), round(duration_seconds * (framerate_num / framerate_denom))

def frame_to_timestamp(frame, framerate):
    return round(float(frame / framerate), 3)

def seconds_to_frames(seconds, framerate):
    return round(framerate * seconds)

class marquee:
    def __init__(self, string):
        self.string = list(string)
        self.marquee_scroller = []
        self.marquee_string = ""

    def reinit(self, marquee_width):
        self.marquee_scroller = [i - 1 for i in range(marquee_width)]
    def advance(self):
        for x in range(len(self.marquee_scroller)):
            if self.marquee_scroller[x] + 1 < len(self.string):
                self.marquee_scroller[x] += 1
            else:
                self.marquee_scroller[x] = 0

        marquee_string = ""
        for x in range(len(self.marquee_scroller)):
            marquee_string += self.string[self.marquee_scroller[x]]

        return marquee_string

class progress_bar(mp.Process):
    def __init__(self, worker_list, segment_list, mux_info_queue):
        super().__init__()
        self.worker_list = worker_list
        self.segment_list = segment_list
        os.environ["TERM"] = "xterm-256color"
        self.total_frames = 0
        self.cumulative_frames = 0
        self.init_time = time.time()
        self.mux_info_queue = mux_info_queue

    @ManagedScreen
    def run(self, screen=None):
        for segment in self.segment_list:
            self.total_frames += segment.num_frames

        worker_stdout_strings = [""] * len(self.worker_list)
        frame = Frame(screen, screen.height, screen.width, has_border=False, can_scroll=False)

        layout_header = Layout([1])
        frame.add_layout(layout_header)
        total_progress_meter_label = Label(label=tqdm.tqdm.format_meter(n=0, total=self.total_frames, elapsed=time.time() - self.init_time, unit='frames'), align='^')
        layout_header.add_widget(total_progress_meter_label, column=0)
        current_mux_info_label = Label(label="Mux status: Waiting...")
        layout_header.add_widget(current_mux_info_label, column=0)
        layout_header.add_widget(Divider(line_char='#'), column=0)
        layout_node_columns = Layout(columns=list([12, 1] * len(self.worker_list))[:-1])
        frame.add_layout(layout_node_columns)

        hostname_labels = []
        hostname_marquee = []
        worker_cum_values = {}
        worker_last_values = {}
        worker_cur_values = {}
        worker_cur_vals_label = {}
        worker_status_header = {}
        worker_status_stderr = {}
        for x, worker in enumerate(self.worker_list):
            hostname_marquee.append(marquee(f"<--{worker.hostname}-->"))
            hostname_marquee[x].reinit(int(screen.width / len(self.worker_list)) - 2)
            label = Label(hostname_marquee[x].marquee_string)
            hostname_labels.append(label)
            layout_node_columns.add_widget(label, x * 2)
            if x < len(self.worker_list) - 1:
                layout_node_columns.add_widget(VerticalDivider(), (x * 2 + 1))
            worker_cum_values[worker] = {'Frame': 0, 'FPS': 0.0, '%RT': 0.0, 'accumulation': 0}
            worker_last_values[worker] = {'Frame': 0, 'FPS': 0.0, '%RT': 0.0}
            worker_cur_values[worker] = {'Frame': 0, 'FPS': 0.0, '%RT': 0.0}
            worker_cur_vals_label[worker] = {}
            for stat in worker_cur_values[worker].keys():
                worker_cur_vals_label[worker][stat] = Label(worker_cur_values[worker][stat])
                layout_node_columns.add_widget(worker_cur_vals_label[worker][stat], x * 2)
            worker_status_header[worker] = Label("Status: Waiting...")
            layout_node_columns.add_widget(worker_status_header[worker], x * 2)
            worker_status_stderr[worker] = TextBox(height=10, line_wrap=True, as_string=True, readonly=True)
            worker_status_stderr[worker].disabled = True
            worker_status_stderr[worker].value = ""
            layout_node_columns.add_widget(worker_status_stderr[worker], x * 2)

        frame.fix()

        while True:
            for x, worker in enumerate(self.worker_list):
                if not worker.stdout_queue.empty():
                    worker_stdout_strings[x] = worker.stdout_queue.get()
                    match = re.search(pattern=r'frame=\s*(\d+)\s*fps=\s*([\d\.]+)\s*q=.*?size=\s*([\d\.]+\s*[KMG]i?B)\s*time=.*?bitrate=\s*([\d\.]+kbits/s)\s*speed=\s*([\d\.x]+)', string=worker_stdout_strings[x])
                    if match:
                        worker_status_header[worker].text = "Status: OK"
                        worker_status_stderr[worker].value = ""
                        worker_cur_values[worker]['Frame'] = int(match.group(1))
                        worker_cur_values[worker]['FPS'] = float(match.group(2))
                        worker_cur_values[worker]['%RT'] = float(match.group(5)[:-1])
                        for stat in worker_cur_values[worker]:
                            try:
                                if stat == 'Frame':
                                    if worker_cur_values[worker][stat] != worker_last_values[worker][stat]:
                                        worker_cur_vals_label[worker][stat].text = f"Frame: {worker_cur_values[worker][stat]}"
                                        if worker_cur_values[worker][stat] > worker_last_values[worker][stat]:
                                            self.cumulative_frames += worker_cur_values[worker][stat] - worker_last_values[worker][stat]
                                        total_progress_meter_label.text = tqdm.tqdm.format_meter(n=self.cumulative_frames, total=self.total_frames,
                                                                                      elapsed=time.time() - self.init_time,
                                                                                      unit='frames')
                                else:
                                    worker_cur_vals_label[worker][stat].text = f"{stat}: {round((worker_cur_values[worker][stat] + worker_cum_values[worker][stat]) / worker_cum_values[worker]['accumulation'], 3)}"

                            except ZeroDivisionError:
                                continue
                            worker_cum_values[worker][stat] += worker_cur_values[worker][stat]
                            worker_last_values[worker][stat] = worker_cur_values[worker][stat]
                        worker_cum_values[worker]['accumulation'] += 1
                if not worker.stderr_queue.empty():
                    worker_stderr_strings = worker.stderr_queue.get()
                    worker_status_header[worker].text = "Status: BAD!"
                    worker_status_stderr[worker].value = f"Code {worker_stderr_strings[1]}, {worker_stderr_strings[2]}"
                hostname_labels[x].text = hostname_marquee[x].advance()

            if not self.mux_info_queue.empty():
                while not self.mux_info_queue.empty():
                    current_mux_info_label.text = str(self.mux_info_queue.get())

            frame.update(0)
            screen.refresh()
            time.sleep(0.1)

class mux_worker(mp.Process):
    def __init__(self, preset, out_path, filename, completed_segment_filename_list, additional_content, file_index, mux_info_queue):
        super().__init__()
        self.preset = preset
        self.out_path = out_path
        self.filename = filename
        self.completed_segment_filename_list = completed_segment_filename_list
        self.additional_content = additional_content
        self.file_index = file_index
        self.mux_info_queue = mux_info_queue

    def run(self):
        with open(f"{self.out_path}/{self.preset['name']}/temp/{self.filename}/mux.txt", 'w') as mux:
            for x in self.completed_segment_filename_list:
                mux.write(f"file '{x.replace("'", "'\\''")}'\n")

        #we ssh into ourselves here to get unbuffered stdout from ffmpeg...
        #I can't figure out how to get it without ssh!?
        ffmpeg_concat_string = (
            f'ssh -t {os.getlogin()}@localhost "ffmpeg -f concat -safe -0 -i '
            f'\\"{self.out_path}/{self.preset['name']}/temp/{self.filename}/mux.txt\\" -c copy '
            f'\\"{self.out_path}/{self.preset['name']}/temp/{self.filename}/{self.filename}\\" -y'
            f'"'
        )

        stderr = ""
        with subprocess.Popen(ffmpeg_concat_string, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True) as process:
            while process.poll() is None:
                stdout = process.stdout.readline()
                self.mux_info_queue.put(f"Muxing segments: {stdout.strip()}")
                err = select.select([process.stderr], [], [], 0.01)[0]
                if err:
                    stderr = process.stderr.read()
                time.sleep(0.01)

        if not process.returncode == 0:
            self.mux_info_queue.put(f"Bad mux! {stderr}")
            self.close()

        mux_video_only = "-A -S -B -T --no-chapters --no-attachments --no-global-tags"
        mux_audio_only = "-D -S -B -T --no-chapters --no-attachments --no-global-tags"
        mux_attachments_only = "-A -D -S -B -T --no-chapters --no-global-tags"
        mux_subtitles_only = "-A -D -B -T --no-chapters --no-attachments --no-global-tags"
        mux_chapters_only = "-A -D -S -B -T --no-attachments --no-global-tags"

        mkvmerge_mux_string = (f'mkvmerge -v --title "{os.path.splitext(self.filename)[0]}" -o "{self.out_path}/'
                               f'{self.preset['name']}/{self.filename}" {mux_video_only} --video-tracks 0 "{self.out_path}/{self.preset['name']}/temp/{self.filename}/{self.filename}" ')

        for path in self.additional_content:
            for content_type in self.additional_content[path]:
                if 'chapters' in self.additional_content[path][content_type]:
                    mkvmerge_mux_string += f'{mux_chapters_only} --chapters "{path}{self.additional_content[path]['file_list'][self.file_index]}" '

                if 'attachments' in self.additional_content[path][content_type]:
                    mkvmerge_mux_string += f'{mux_attachments_only} --attachments -1:all "{path}{self.additional_content[path]['file_list'][self.file_index]}" '

                if 'audio' in self.additional_content[path][content_type]:
                    for track_id in self.additional_content[path][content_type]['audio']:
                        if self.preset['name'] in self.additional_content[path][content_type]['audio'][track_id]['presets']:
                            ffmpeg_cmd = (f'ssh -t {os.getlogin()}@localhost "ffmpeg -i \\"{path}{self.additional_content[path]['file_list'][self.file_index]}\\" '
                                f'-map 0:{track_id} -map_metadata -{track_id} -map_chapters -{track_id} {self.preset['ffmpeg_audio_string']} '
                                f'\\"{self.out_path}/{self.preset['name']}/temp/{self.filename}/audio_{track_id}_'
                                f'{self.additional_content[path][content_type]['audio'][track_id]['lang']}_'
                                f'{self.additional_content[path][content_type]['audio'][track_id]['track_name']}_{self.filename}\\" -y'
                                f'"'
                            )

                            with subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                  text=True, shell=True) as process:
                                while process.poll() is None:
                                    stdout = process.stdout.readline()
                                    self.mux_info_queue.put(f"Re-encoding/copying audio: {stdout.strip()}")
                                    err = select.select([process.stderr], [], [], 0.01)[0]
                                    if err:
                                        stderr = process.stderr.read()
                                    time.sleep(0.01)

                            if not process.returncode == 0:
                                self.mux_info_queue.put(f"Bad mux! {stderr}")
                                self.close()

                            if self.additional_content[path][content_type]['audio'][track_id].get('default_flag', False):
                                mkvmerge_mux_string += f'--default-track-flag 0 '
                            if self.additional_content[path][content_type]['audio'][track_id].get('original_language', False):
                                mkvmerge_mux_string += f'--original-flag 0 '

                            mkvmerge_mux_string += (f'--audio-tracks '
                                f'0 '
                                f'--language 0:{self.additional_content[path][content_type]['audio'][track_id]['lang']} '
                                f'--track-name 0:"{self.additional_content[path][content_type]['audio'][track_id]['track_name']}" '
                                f'{mux_audio_only} "{self.out_path}/{self.preset['name']}/temp/{self.filename}/audio_{track_id}_'
                                f'{self.additional_content[path][content_type]['audio'][track_id]['lang']}_'
                                f'{self.additional_content[path][content_type]['audio'][track_id]['track_name']}_{self.filename}" ')

                if 'subtitles' in self.additional_content[path][content_type]:
                    for track_id in self.additional_content[path][content_type]['subtitles']:
                        if self.preset['name'] in self.additional_content[path][content_type]['subtitles'][track_id]['presets']:
                            if self.additional_content[path][content_type]['subtitles'][track_id].get('default_flag', False):
                                mkvmerge_mux_string += f'--default-track-flag {track_id} '
                            if self.additional_content[path][content_type]['subtitles'][track_id].get('forced', False):
                                mkvmerge_mux_string += f'--forced-display-flag {track_id} '
                            if self.additional_content[path][content_type]['subtitles'][track_id].get('original_language', False):
                                mkvmerge_mux_string += f'--original-flag {track_id} '
                            mkvmerge_mux_string += (f'--subtitle-tracks '
                                f'{track_id} '
                                f'--language {track_id}:{self.additional_content[path][content_type]['subtitles'][track_id]['lang']} '
                                f'--track-name {track_id}:"{self.additional_content[path][content_type]['subtitles'][track_id]['track_name']}" '
                                f'{mux_subtitles_only} "{path}{self.additional_content[path]['file_list'][self.file_index]}" ')

        with subprocess.Popen(mkvmerge_mux_string, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True) as process:
            while process.poll() is None:
                stdout = process.stdout.readline()
                self.mux_info_queue.put(f"Final mux: {stdout.strip()}")
                err = select.select([process.stderr], [], [], 0.01)[0]
                if err:
                    stderr = process.stderr.read()
                time.sleep(0.01)

        if not process.returncode == 0:
            self.mux_info_queue.put(f"Bad mux! {stderr}")
            self.close()

        self.mux_info_queue.put(f"Good mux: {self.preset['name']}/{self.filename}")

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

    def check_if_exists(self, mux_info_queue):
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            '-show_entries', 'format=duration', self.file_output_fstring
        ]
        if os.path.isfile(self.file_output_fstring):
            duration = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if duration.returncode == 0:
                duration = round(float(duration.stdout.strip()), 3)
                if seconds_to_frames(duration, self.framerate) == self.num_frames:
                    self.encode_job.tally_completed_segments(self.file_output_fstring, mux_info_queue)
                    return True
                else:
                    os.remove(self.file_output_fstring)
                    return False
            else:
                os.remove(self.file_output_fstring)
                return False
        else:
            return False

    def encode(self, hostname, current_user, stdout_queue):
        cmd = (
            f'ssh -o ServerAliveInterval=10 -t {current_user}@{hostname} "ffmpeg -ss {self.segment_start} -to {self.segment_end} -i '
            f'\\"{self.file_fullpath}\\" '
            f"{self.ffmpeg_video_string} "
            f'\\"{self.file_output_fstring}\\"'
            f'"'
        )
        stderr = ""
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True) as process:
            while process.poll() is None:
                stdout = process.stdout.readline()
                stdout_queue.put(stdout)
                err = select.select([process.stderr], [], [], 0.01)[0]
                if err:
                    stderr = process.stderr.read()
                time.sleep(0.01)

        return process.returncode, stderr

class encode_job:
    def __init__(self, proper_name, input_file, preset, out_path, filename, additional_content, file_index):
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
        self.file_index = file_index

        if not os.path.isdir(f"{self.out_path}/{self.preset['name']}/temp/{self.filename}/"):
            os.makedirs(f"{self.out_path}/{self.preset['name']}/temp/{self.filename}/")

    def check_if_exists(self):
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            '-show_entries', 'format=duration', f"{self.out_path}/{self.preset['name']}/{self.filename}"
        ]
        if os.path.isfile(f"{self.out_path}/{self.preset['name']}/{self.filename}"):
            duration = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if duration.returncode == 0:
                duration = round(float(duration.stdout.strip()), 3)
                if seconds_to_frames(duration, self.framerate) == self.frames_total:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def tally_completed_segments(self, filename, mux_info_queue):
        self.segments_completed += 1
        self.completed_segment_filename_list.append(filename)
        if self.segments_completed == self.num_segments:
            self.completed_segment_filename_list = sorted(self.completed_segment_filename_list,
                key=lambda x: float(re.search(r'/(\d+\.\d+)-', x).group(1)))
            mux = mux_worker(self.preset, self.out_path, self.filename, self.completed_segment_filename_list,
                             self.additional_content, self.file_index, mux_info_queue)
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
    def __init__(self, hostname, current_user):
        super().__init__()
        self.hostname = hostname
        self.current_user = current_user
        self.current_segment = mp.Manager().Value(typecode=None, value=None)
        self.is_running = mp.Value('b', False)
        self.test_queue = mp.Queue()
        self.stderr_queue = mp.Queue()
        self.stdout_queue = mp.Queue()

    def return_hostname(self):
        return self.hostname

    def run(self):
        while True:
            if not self.current_segment.value is None and self.is_running.value == True:
                returncode, stderr = self.execute_encode(segment_to_encode=self.current_segment.value)
                self.test_queue.put((self.current_segment.value, returncode, stderr))
                self.is_running.value = False
            time.sleep(1)

    def execute_encode(self, segment_to_encode):
        returncode, stderr = segment_to_encode.encode(self.hostname, self.current_user, stdout_queue = self.stdout_queue)
        return returncode, stderr

def job_handler(segment_list, worker_list):

    mux_info_queue = mp.Queue()

    segment_list[:] = [segment for segment in segment_list if not segment.check_if_exists(mux_info_queue)]

    progress_bar_worker = progress_bar(worker_list, segment_list, mux_info_queue)
    progress_bar_worker.start()

    segment_index = 0
    while len(segment_list) > 0:
        while segment_list[segment_index].assigned:
            segment_index += 1

        for worker in worker_list:
            if worker.current_segment.value is None or worker.is_running.value == False:
                if not worker.test_queue.empty():
                    results = worker.test_queue.get()
                    if results[1] == 0:
                        for segment in segment_list:
                            if segment.uuid == results[0].uuid:
                                segment.encode_job.tally_completed_segments(segment.file_output_fstring, mux_info_queue)
                                segment_list.remove(segment)
                                break
                    else:
                        worker.stderr_queue.put(results)
                        for segment in segment_list:
                            if segment.uuid == results[0].uuid:
                                segment.assigned = False

                    worker.current_segment.value = None

                else:
                    worker.current_segment.value = segment_list[segment_index]
                    worker.is_running.value = True
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

    for path in config['additional_content']:
        config['additional_content'][path]['file_list'] = sorted(os.listdir(path))

    for x, file in enumerate(sorted(os.listdir(args.in_path)), start=0):
        file_fullpath = os.path.join(args.in_path, file)
        for preset in config['presets']:
            job_list += [encode_job(proper_name=f"{preset['name']}_{file}", input_file=file_fullpath, preset=preset,
                out_path=args.out_path, filename=file, additional_content=config['additional_content'],
                file_index=x)]

    job_list = sorted(job_list, key=lambda x: x.proper_name)
    job_list[:] = [job for job in job_list if not job.check_if_exists()]
    job_segment_list = []
    for jobs in job_list:
        job_segment_list += jobs.create_segment_encode_list()

    for worker in config['nodes']:
        worker_list += [encode_worker(hostname=worker['hostname'], current_user=current_user)]
        worker_list[len(worker_list)-1].start()

    job_handler(job_segment_list, worker_list)

if __name__ == "__main__":
    main()