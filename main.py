from __future__ import annotations
import yaml
import argparse
import os
import subprocess
import uuid
import multiprocessing as mp
import time
import re
import math
import tqdm
import select
from rich.live import Live  # type: ignore
from rich.table import Table  # type: ignore
from rich.layout import Layout  # type: ignore
from rich.panel import Panel  # type: ignore
from rich import print  # type: ignore
import paramiko
from datetime import datetime
from typing import Any, cast


def load_config(file_path: str) -> dict[str, Any]:
    with open(file_path, 'r') as file:
        return cast(dict[str, Any], yaml.safe_load(file))


def get_segments_and_framerate(file: str, frames_per_segment: int) -> tuple[int, float, int]:
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
    duration_seconds: float = float(subprocess.run(duration_seconds_command, stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE, text=True).stdout.strip())
    return math.ceil(int((framerate_num / framerate_denom) * duration_seconds) / frames_per_segment), \
        float(framerate_num / framerate_denom), round(duration_seconds * (framerate_num / framerate_denom))


def frame_to_timestamp(frame: int, framerate: float) -> float:
    return round(float(frame / framerate), 3)


def seconds_to_frames(seconds: float, framerate: float) -> int:
    return round(framerate * seconds)


class MuxWorker(mp.Process):
    def __init__(self, preset: dict[str, str], out_path: str, filename: str, completed_segment_filename_list: list[str],
                 additional_content: dict[str, Any], file_index: int, mux_info_queue: mp.Queue[str]) -> None:
        super().__init__()
        self.preset = preset
        self.out_path = out_path
        self.filename = filename
        self.completed_segment_filename_list = completed_segment_filename_list
        self.additional_content = additional_content
        self.file_index = file_index
        self.mux_info_queue = mux_info_queue

    def run(self) -> None:
        with open(f"{self.out_path}/{self.preset['name']}/temp/{self.filename}/mux.txt", 'w') as mux:
            for x in self.completed_segment_filename_list:
                mux.write(f"file '{x.replace("'", "'\\''")}'\n")

        # we ssh into ourselves here to get unbuffered stdout from ffmpeg...
        # I can't figure out how to get it without ssh!?
        # same BS running it twice to get stderr for ffmpeg
        ffmpeg_concat_string = (
            f'ffmpeg -f concat -safe -0 -i '
            f'\"{self.out_path}/{self.preset['name']}/temp/{self.filename}/mux.txt\" -c copy '
            f'\"{self.out_path}/{self.preset['name']}/temp/{self.filename}/{self.filename}\" -y'
        )

        return_code, stderr = execute_cmd_ssh(ffmpeg_concat_string, "localhost", os.getlogin(),
                                              self.mux_info_queue, get_pty=True, prefix="Muxing segments: ")

        # paramiko can only get ffmpeg stdout with get_pty=True... But it can only get stderr with get_pty=False...
        if not return_code == 0:
            return_code, stderr = execute_cmd_ssh(ffmpeg_concat_string, "localhost", os.getlogin(),
                                                  self.mux_info_queue, get_pty=False)
            self.mux_info_queue.put(cast(str, stderr))
            self.close()

        self.mux_info_queue.put("Muxing segments: Done! Audio transcode soon...")

        mux_video_only = "-A -S -B -T --no-chapters --no-attachments --no-global-tags"
        mux_audio_only = "-D -S -B -T --no-chapters --no-attachments --no-global-tags"
        mux_attachments_only = "-A -D -S -B -T --no-chapters --no-global-tags"
        mux_subtitles_only = "-A -D -B -T --no-chapters --no-attachments --no-global-tags"
        mux_chapters_only = "-A -D -S -B -T --no-attachments --no-global-tags"

        mkvmerge_mux_string = (f'mkvmerge -v --title "{os.path.splitext(self.filename)[0]}" -o "{self.out_path}/'
                               f'{self.preset['name']}/{self.filename}" {mux_video_only} --video-tracks 0 "'
                               f'{self.out_path}/{self.preset['name']}/temp/{self.filename}/{self.filename}" ')

        for path in self.additional_content:
            for content_type in self.additional_content[path]:
                if 'chapters' in self.additional_content[path][content_type]:
                    mkvmerge_mux_string += (f'{mux_chapters_only} --chapters "'
                                            f'{path}{self.additional_content[path]['file_list'][self.file_index]}" ')

                if 'attachments' in self.additional_content[path][content_type]:
                    mkvmerge_mux_string += f'{mux_attachments_only} --attachments -1:all "{path}{self.additional_content[path]['file_list'][self.file_index]}" '

                if 'audio' in self.additional_content[path][content_type]:
                    for track_id in self.additional_content[path][content_type]['audio']:
                        if self.preset['name'] in self.additional_content[path][content_type]['audio'][track_id]['presets']:
                            ffmpeg_cmd = (f'ffmpeg -i \"{path}{self.additional_content[path]['file_list'][self.file_index]}\" '
                                          f'-map 0:{track_id} -map_metadata -{track_id} -map_chapters -{track_id} '
                                          f'{self.preset['ffmpeg_audio_string']} \"{self.out_path}/'
                                          f'{self.preset['name']}/temp/{self.filename}/audio_{track_id}_'
                                          f'{self.additional_content[path][content_type]['audio'][track_id]['lang']}_'
                                          f'{self.additional_content[path][content_type]['audio'][track_id]['track_name']}_{self.filename}\" -y'
                            )

                            return_code, stderr = execute_cmd_ssh(ffmpeg_cmd, "localhost", os.getlogin(),
                                                                  self.mux_info_queue, get_pty=True,
                                                                  prefix="Transcode/copy audio: ")

                            if not return_code == 0:
                                return_code, stderr = execute_cmd_ssh(ffmpeg_cmd, "localhost", os.getlogin(),
                                                                      self.mux_info_queue, get_pty=False)
                                self.mux_info_queue.put(cast(str, stderr))
                                self.close()

                            self.mux_info_queue.put("Transcode/copy audio: Done!")

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

        with subprocess.Popen(mkvmerge_mux_string, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              shell=True) as process:
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("Failed to read stdout and stderr from muxing subprocess!")

            while process.poll() is None:
                stdout = process.stdout.readline()
                self.mux_info_queue.put(f"Final Mux: {stdout.strip()}")
                err = select.select([process.stderr], [], [], 0.01)[0]
                if err:
                    stderr = process.stderr.read()
                time.sleep(0.01)

        if not process.returncode == 0:
            self.mux_info_queue.put(f"Bad mux! {stderr}")
            self.close()

        self.mux_info_queue.put(f"Good mux: {self.preset['name']}/{self.filename}")

        self.close()


class EncodeWorker(mp.Process):
    def __init__(self, hostname: str, current_user: str) -> None:
        super().__init__()
        self.hostname = hostname
        self.current_user = current_user
        self.manager = mp.Manager()
        self.current_segment = self.manager.Namespace()
        self.current_segment.value = None
        self.is_running = mp.Value('b', False)
        # by "any" of course, I mean, mp.Manager.Value, but I couldn't figure out how to make mypy happy
        self.stdout_queue: mp.Queue[str] = mp.Queue()
        self.stderr_queue: mp.Queue[tuple[Any, int, str | Exception]] = mp.Queue()
        self.shutdown = mp.Value('b', False)
        self.err_cooldown = mp.Value('b', False)
        self.err_timestamp = 0.0

    def return_hostname(self) -> str:
        return self.hostname

    def run(self) -> None:
        while not self.shutdown.value:
            if time.time() - self.err_timestamp > 30:
                self.err_cooldown.value = False

            if not self.current_segment.value is None and self.is_running.value == True:
                returncode, stderr = self.execute_encode(segment_to_encode=self.current_segment.value)
                self.stderr_queue.put((self.current_segment.value, returncode, stderr))
                self.is_running.value = False

            time.sleep(1)
        self.close()

    def execute_encode(self, segment_to_encode: EncodeSegment) -> tuple[int, str | Exception]:
        returncode, stderr = segment_to_encode.encode(self.hostname, self.current_user, stdout_queue=self.stdout_queue)
        if not returncode == 0:
            self.err_timestamp = time.time()
        return returncode, stderr


class RichHelper:
    def __init__(self, worker_list: list[EncodeWorker], segment_list: list[EncodeSegment],
                 mux_info_queue: mp.Queue[str]) -> None:
        self.worker_list = worker_list
        self.segment_list = segment_list
        self.total_frames = 0
        self.cumulative_frames = 0
        self.init_time = time.time()
        self.mux_info_queue = mux_info_queue
        self.worker_cum_values: dict[EncodeWorker, dict[str, float | int]] = {}
        self.worker_last_values: dict[EncodeWorker, dict[str, float | int]] = {}
        self.worker_avg_values: dict[EncodeWorker, dict[str, float]] = {}
        self.worker_cur_values: dict[EncodeWorker, dict[str, float | int | str]] = {}
        # was I planning on using these?
        # self.worker_status_header: dict[EncodeWorker, str] = {}
        # self.worker_status_stderr: dict[EncodeWorker.stderr_queue] = {}
        self.mux_strings_list = [""] * 8
        self.stderr_strings_list = [""] * 8

        for worker in self.worker_list:
            self.worker_cum_values[worker] = {'Frame': 0, 'FPS': 0.0, '%RT': 0.0, 'accumulation': 0}
            self.worker_last_values[worker] = {'Frame': 0, 'FPS': 0.0, '%RT': 0.0}
            self.worker_avg_values[worker] = {'FPS': 0.0, '%RT': 0.0}
            self.worker_cur_values[worker] = {'Frame': 0, 'FPS': 0.0, '%RT': 0.0, 'Status': "Idle"}

        for segment in self.segment_list:
            self.total_frames += segment.num_frames

        # use case for this variable?
        # worker_stdout_strings = [""] * len(self.worker_list)

    def update_stderr(self, hostname: str, stderr: tuple[Any, int, str | Exception]) -> str:
        stderr_text = ""
        self.stderr_strings_list.append(f"{datetime.now().strftime('%H:%M:%S')} {hostname}: {str(stderr[2]).splitlines()[-1]}")

        if len(self.stderr_strings_list) > 8:
            self.stderr_strings_list.pop(0)

        for x in range(len(self.stderr_strings_list)):
            stderr_text += self.stderr_strings_list[x]
            stderr_text += "\n"

        return stderr_text

    def update_mux_info(self, mux_info_queue: mp.Queue[str]) -> str:
        mux_text = ""
        new_mux_string = mux_info_queue.get()
        new_mux_prefix = new_mux_string.split(':')[0]
        # print(new_mux_prefix)
        # print(self.mux_strings_list[7])
        if self.mux_strings_list[7].startswith(new_mux_prefix):
            self.mux_strings_list[7] = new_mux_string
        else:
            self.mux_strings_list.append(new_mux_string)

        if len(self.mux_strings_list) > 8:
            self.mux_strings_list.pop(0)

        for x in range(len(self.mux_strings_list)):
            mux_text += self.mux_strings_list[x]
            mux_text += "\n"

        return mux_text


    def create_node_table(self) -> Table:
        table = Table(title=tqdm.tqdm.format_meter(n=self.cumulative_frames, total=self.total_frames, elapsed=time.time() - self.init_time,
                                                   unit='frames'))
        table.add_column(header="Hostname", min_width=16)
        table.add_column(header="Status", min_width=5)
        table.add_column(header="# Frames", min_width=8)
        table.add_column(header="Avg FPS", min_width=7)
        table.add_column(header="Avg %RT", min_width=7)

        for worker in self.worker_list:
            table.add_row(worker.hostname, self.worker_cur_values[worker]['Status'], str(self.worker_cum_values[worker]['Frame']),
                          str(self.worker_avg_values[worker]['FPS']), f"{self.worker_avg_values[worker]['%RT']}x")

        return table

    def update_worker_status(self, worker: EncodeWorker, status: str) -> None:
        match = re.search(pattern=r'frame=\s*(\d+)\s*fps=\s*([\d\.]+)\s*q=.*?size=\s*([\d\.]+\s*[KMG]i?B)\s*time=.*?bitrate=\s*([\d\.]+kbits/s)\s*speed=\s*([\d\.x]+)', string=status)
        if match:
            self.worker_cur_values[worker]['Status'] = "OK"
            self.worker_cur_values[worker]['Frame'] = int(match.group(1))
            self.worker_cur_values[worker]['FPS'] = float(match.group(2))
            self.worker_cur_values[worker]['%RT'] = float(match.group(5)[:-1])
            for stat in self.worker_cur_values[worker]:
                try:
                    if stat == 'Frame':
                        if not cast(int, self.worker_cur_values[worker][stat]) == cast(int, self.worker_last_values[worker][stat]):
                            if cast(int, self.worker_cur_values[worker][stat]) > cast(int, self.worker_last_values[worker][stat]):
                                self.cumulative_frames += cast(int, self.worker_cur_values[worker][stat]) - cast(int, self.worker_last_values[worker][stat])
                                self.worker_cum_values[worker][stat] += cast(int, self.worker_cur_values[worker][stat]) - cast(int, self.worker_last_values[worker][stat])
                    elif not stat == 'Status':
                        self.worker_avg_values[worker][stat] = round((cast(int | float, self.worker_cur_values[worker][stat]) +
                                                                      self.worker_cum_values[worker][stat]) /
                                                                     self.worker_cum_values[worker]['accumulation'], 3)
                        self.worker_cum_values[worker][stat] += cast(int | float, self.worker_cur_values[worker][stat])

                except ZeroDivisionError:
                    continue

                self.worker_last_values[worker][stat] = cast(float | int, self.worker_cur_values[worker][stat])

            self.worker_cum_values[worker]['accumulation'] += 1

        elif re.search(pattern=r'Press \[q\] to stop, \[\?\] for help', string=status):
            self.worker_cur_values[worker]['Status'] = "Seek"


class EncodeSegment:
    def __init__(self, framerate: float, file_fullpath: str, out_path: str, segment_start: int, segment_end: int,
                 ffmpeg_video_string: str, preset: dict[str, str], filename: str, encode_job: EncodeJob, num_frames: int) -> None:
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

    def check_if_exists(self, mux_info_queue: mp.Queue[str]) -> bool:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            '-show_entries', 'format=duration', self.file_output_fstring
        ]
        if os.path.isfile(self.file_output_fstring):
            duration_proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if duration_proc.returncode == 0:
                duration = round(float(duration_proc.stdout.strip()), 3)
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

    def encode(self, hostname: str, current_user: str, stdout_queue: mp.Queue[str]) -> tuple[int, str | Exception]:
        cmd = (
            f'ffmpeg -ss {self.segment_start} -to {self.segment_end} -i '
            f'\"{self.file_fullpath}\" '
            f"{self.ffmpeg_video_string} "
            f'\"{self.file_output_fstring}\"'
        )

        return_code, stderr = execute_cmd_ssh(cmd, hostname, current_user, stdout_queue, get_pty=True)

        # paramiko can only get ffmpeg stdout with get_pty=True... But it can only get stderr with get_pty=False...
        if not return_code == 0:
            return_code, stderr = execute_cmd_ssh(cmd, hostname, current_user, stdout_queue, get_pty=False)

        return return_code, stderr


def execute_cmd_ssh(cmd: str, hostname: str, username: str, stdout_queue: mp.Queue[str], get_pty: bool,
                    prefix: str = "") -> tuple[int, str | Exception]:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname=hostname, username=username, key_filename=f"/home/{username}/.ssh/id_rsa.pub")

    except Exception as e:
        client.close()
        return -1, e

    client.invoke_shell()
    stdin, stdout, stderr = client.exec_command(cmd, get_pty=get_pty)
    try:
        line = b''
        while not stdout.channel.exit_status_ready():
            for byte in iter(lambda: stdout.read(1), b""):
                line += byte
                if byte == b'\n' or byte == b'\r':
                    stdout_queue.put(f"{prefix}{line.decode('utf-8', errors='replace').rstrip()}")
                    line = b''
    except KeyboardInterrupt:
        kill_cmd = (
            f'pkill -9 -f {cmd}'
        )
        client.exec_command(kill_cmd)
        client.close()

    client.close()
    return stdout.channel.recv_exit_status(), stderr.read().decode('utf-8')



class EncodeJob:
    def __init__(self, proper_name: str, input_file: str, preset: dict[str, str], out_path: str, filename: str,
                 additional_content: dict[str, Any], file_index: int) -> None:
        self.input_file = input_file
        self.proper_name = proper_name
        self.frames_per_segment = 2000
        self.num_segments, self.framerate, self.frames_total = get_segments_and_framerate(self.input_file,
                                                                                          self.frames_per_segment)
        self.preset = preset
        self.out_path = out_path
        self.filename = filename
        self.segments_completed = 0
        self.completed_segment_filename_list: list[str] = []
        self.additional_content = additional_content
        self.file_index = file_index

        if not os.path.isdir(f"{self.out_path}/{self.preset['name']}/temp/{self.filename}/"):
            os.makedirs(f"{self.out_path}/{self.preset['name']}/temp/{self.filename}/")

    def check_if_exists(self) -> bool:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            '-show_entries', 'format=duration', f"{self.out_path}/{self.preset['name']}/{self.filename}"
        ]
        if os.path.isfile(f"{self.out_path}/{self.preset['name']}/{self.filename}"):
            duration_proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if duration_proc.returncode == 0:
                if not duration_proc.stdout.strip() == "N/A":
                    duration = round(float(duration_proc.stdout.strip()), 3)
                    if seconds_to_frames(duration, self.framerate) == self.frames_total:
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            return False

    def tally_completed_segments(self, filename: str, mux_info_queue: mp.Queue[str]) -> None:
        self.segments_completed += 1
        self.completed_segment_filename_list.append(filename)
        if self.segments_completed == self.num_segments:
            self.completed_segment_filename_list = sorted(self.completed_segment_filename_list,
                                                          key=lambda x: float(match.group(1)) if
                                                          (match := re.search(r'/(\d+\.\d+)-', x)) else 0.0)

            mux = MuxWorker(self.preset, self.out_path, self.filename, self.completed_segment_filename_list,
                            self.additional_content, self.file_index, mux_info_queue)
            mux.start()

    def create_segment_encode_list(self) -> list[EncodeSegment]:
        segment_list = []
        frames_assigned = 0
        last_segment_compensation = self.frames_per_segment
        for x in range(self.num_segments):
            if self.frames_total - frames_assigned < self.frames_per_segment:
                last_segment_compensation = self.frames_total - frames_assigned
            segment_list += [EncodeSegment(framerate=self.framerate, file_fullpath=self.input_file,
                                           out_path=self.out_path,
                                           ffmpeg_video_string=self.preset['ffmpeg_video_string'],
                                           segment_start=x * self.frames_per_segment,
                                           segment_end=(x + 1) * self.frames_per_segment,
                                           preset=self.preset, filename=self.filename, encode_job=self,
                                           num_frames=last_segment_compensation)]
            frames_assigned += self.frames_per_segment
        return segment_list


def job_handler(segment_list: list[EncodeSegment], worker_list: list[EncodeWorker]) -> None:

    mux_info_queue: mp.Queue[str] = mp.Queue()

    segment_list[:] = [segment for segment in segment_list if not segment.check_if_exists(mux_info_queue)]

    tui = RichHelper(worker_list, segment_list, mux_info_queue)
    layout = Layout()
    # layout.split_column(Layout(name="header", size=4), Layout(name="table"), Layout(name="footer", size=8))
    layout.split_column(Layout(name="table"), Layout(name="stderr", size=10), Layout(name="Mux Info", size=10))
    layout['table'].update(tui.create_node_table())

    layout['stderr'].update(Panel("Nothing here yet!", title="Errors"))
    layout['Mux Info'].update(Panel("Nothing here yet...", title="Mux Info"))

    segment_index = 0
    # I put this here for a reason, maybe, but I don't remember what
    # mux_string_arr = []
    with Live(layout, refresh_per_second=1, screen=True):
        while len(segment_list) > 0:
            while segment_list[segment_index].assigned and segment_index + 1 < len(segment_list):
                segment_index += 1
            for worker in worker_list:
                if (worker.current_segment.value is None or not worker.is_running.value) and not \
                        worker.err_cooldown.value:
                    if not worker.stderr_queue.empty():
                        results = worker.stderr_queue.get()
                        if results[1] == 0:
                            for segment in segment_list:
                                if segment.uuid == results[0].uuid:
                                    segment.encode_job.tally_completed_segments(segment.file_output_fstring,
                                                                                mux_info_queue)
                                    segment_list.remove(segment)
                                    tui.worker_cur_values[worker]['Status'] = "Idle"
                                    break
                        else:
                            for segment in segment_list:
                                if segment.uuid == results[0].uuid:
                                    segment.assigned = False
                                    worker.err_cooldown.value = True
                            layout['stderr'].update(Panel(tui.update_stderr(worker.hostname, results), title="stderr"))
                            tui.worker_cur_values[worker]['Status'] = "Error"
                        worker.current_segment.value = None
                    else:
                        try:
                            if not segment_list[segment_index].assigned:
                                worker.current_segment.value = segment_list[segment_index]
                                worker.is_running.value = True
                                segment_list[segment_index].assigned = True
                                segment_index += 1
                        except IndexError:
                            continue
                elif not worker.stdout_queue.empty():
                    tui.update_worker_status(worker, worker.stdout_queue.get())
            segment_index = 0
            layout['table'].update(tui.create_node_table())
            if not mux_info_queue.empty():
                layout['Mux Info'].update(Panel(tui.update_mux_info(mux_info_queue), title="Mux Info"))
            time.sleep(0.01)

    for worker in worker_list:
        worker.shutdown.value = True
        worker.join()

    mux_workers = [x for x in mp.active_children() if x.name.startswith('mux_worker')]

    for mux in mux_workers:
        while mux.is_alive():
            if not mux_info_queue.empty():
                print(mux_info_queue.get())
            time.sleep(0.01)


def main() -> None:
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
            job_list += [EncodeJob(proper_name=f"{preset['name']}_{file}", input_file=file_fullpath, preset=preset,
                                   out_path=args.out_path, filename=file,
                                   additional_content=config['additional_content'], file_index=x)]

    job_list = sorted(job_list, key=lambda job: job.proper_name)
    job_list[:] = [job for job in job_list if not job.check_if_exists()]
    job_segment_list = []
    for jobs in job_list:
        job_segment_list += jobs.create_segment_encode_list()

    for worker in config['nodes']:
        worker_list += [EncodeWorker(hostname=worker['hostname'], current_user=current_user)]
        worker_list[len(worker_list)-1].start()

    job_handler(job_segment_list, worker_list)


if __name__ == "__main__":
    main()
