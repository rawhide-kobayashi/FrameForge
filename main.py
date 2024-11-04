from __future__ import annotations

import random

import yaml
import argparse
import os
import subprocess
import multiprocessing as mp
# importing this separately makes mypy happy for some reason
import multiprocessing.managers
import time
import re
import math
import select
from rich.live import Live  # type: ignore
from rich.table import Table  # type: ignore
from rich.layout import Layout  # type: ignore
from rich.panel import Panel  # type: ignore
from rich import print  # type: ignore
from rich.progress import Progress, ProgressColumn, SpinnerColumn, TextColumn, MofNCompleteColumn  # type: ignore
from rich.progress import TimeRemainingColumn, Text, Task
import paramiko
from datetime import datetime
from typing import Any, cast
import json
from threading import Lock as TypedLock
import sys
import shutil


class TransferSpeedColumnFPS(ProgressColumn):  # type: ignore
    """Renders human-readable transfer speed... in FPS!"""

    def render(self, task: "Task") -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        data_speed = round(speed, 2)
        return Text(f"{data_speed} FPS", style="progress.data.speed")


class TextMarquee:
    def __init__(self, string: str, width: int) -> None:
        self.string = list(string)
        self.marquee_scroller: list[int] = [i - 1 for i in range(width)]
        self.marquee_string = ""

    def advance(self) -> str:
        for x in range(len(self.marquee_scroller)):
            if self.marquee_scroller[x] + 1 < len(self.string):
                self.marquee_scroller[x] += 1
            else:
                self.marquee_scroller[x] = 0

        marquee_string = ""
        for x in range(len(self.marquee_scroller)):
            marquee_string += self.string[self.marquee_scroller[x]]

        return marquee_string


def load_config(file_path: str) -> dict[str, Any]:
    with open(file_path, 'r') as file:
        return cast(dict[str, Any], yaml.safe_load(file))


def get_framerate(file: str) -> tuple[int, int]:
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        '-show_entries', 'stream=avg_frame_rate', file
    ]

    return cast(tuple[int, int], map(int, subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                         text=True).stdout.strip().split('/')))


def get_duration(file: str) -> float:
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        '-show_entries', 'format=duration', file
    ]

    return float(subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip())


def frame_to_timestamp(frame: int, framerate: float) -> float:
    return round(float(frame / framerate), 6)


def seconds_to_frames(seconds: float, framerate: float) -> int:
    return round(framerate * seconds)


class MuxWorker(mp.Process):
    def __init__(self, preset: dict[str, str], out_path: str, filename: str, completed_segment_filename_list: list[str],
                 additional_content: dict[str, Any], file_index: int, mux_info_queue: mp.Queue[str],
                 ssh_username: str, encode_job: EncodeJob) -> None:
        super().__init__()
        self.preset = preset
        self.out_path = out_path
        self.filename = filename
        self.completed_segment_filename_list = completed_segment_filename_list
        self.additional_content = additional_content
        self.file_index = file_index
        self.mux_info_queue = mux_info_queue
        self.ssh_username = ssh_username
        self.encode_job = encode_job
        self.mux_txt = f'{self.out_path}/temp/{self.preset['name']}/{self.filename}/mux.txt'
        self.mux_video_path = f'{self.out_path}/temp/{self.preset['name']}/{self.filename}'
        self.start()

    def run(self) -> None:
        with open(self.mux_txt, 'w') as mux:
            for x in self.completed_segment_filename_list:
                mux.write(f"file '{x.replace("'", "'\\''")}'\n")

        # we ssh into ourselves here to get unbuffered stdout from ffmpeg...
        # I can't figure out how to get it without ssh!?
        # same BS running it twice to get stderr for ffmpeg
        ffmpeg_concat_string = (
            f'ffmpeg -f concat -safe -0 -i \"{self.mux_txt}\" -c copy \"{self.mux_video_path}/{self.filename}\" -y'
        )

        return_code, stderr = execute_cmd_ssh(cmd=ffmpeg_concat_string, hostname="localhost",
                                              ssh_username=self.ssh_username, stdout_queue=self.mux_info_queue,
                                              get_pty=True, prefix=f'Concatenating segments for '
                                                                   f'{self.preset['name']}/{self.filename}: ')

        # paramiko can only get ffmpeg stdout with get_pty=True... But it can only get stderr with get_pty=False...
        if not return_code == 0:
            return_code, stderr = execute_cmd_ssh(cmd=ffmpeg_concat_string, hostname="localhost",
                                                  ssh_username=self.ssh_username, stdout_queue=self.mux_info_queue,
                                                  get_pty=False)
            self.mux_info_queue.put(cast(str, stderr))
            self.close()

        self.mux_info_queue.put(f'Concatenating segments for {self.preset['name']}/{self.filename}: '
                                f'Done! Audio transcode soon...')

        mux_video_only = "-A -S -B -T --no-chapters --no-attachments --no-global-tags"
        mux_audio_only = "-D -S -B -T --no-chapters --no-attachments --no-global-tags"
        mux_attachments_only = "-A -D -S -B -T --no-chapters --no-global-tags"
        mux_subtitles_only = "-A -D -B -T --no-chapters --no-attachments --no-global-tags"
        mux_chapters_only = "-A -D -S -B -T --no-attachments --no-global-tags"

        # mkvmerge_mux_string = (f'mkvmerge -v --title "{os.path.splitext(self.filename)[0]}" -o "{self.out_path}/'
        #                        f'{self.preset['name']}/{self.filename}" {mux_video_only} --default-duration 0:'
        #                        f'{self.encode_job.framerate_numerator}/{self.encode_job.framerate_denominator}fps '
        #                        f'--video-tracks 0 \"{self.out_path}/{self.preset['name']}/temp/{self.filename}/'
        #                        f'{self.filename}\" ')

        mkvmerge_mux_string = (f'mkvmerge -v --title "{os.path.splitext(self.filename)[0]}" -o "{self.out_path}/'
                               f'output/{self.preset['name']}/{self.filename}" {mux_video_only} '
                               f'--video-tracks 0 \"{self.mux_video_path}/{self.filename}\" ')

        for path in self.additional_content:
            file_path = f"{path}{self.additional_content[path]['file_list'][self.file_index]}"
            for content_type in self.additional_content[path]:
                if 'chapters' in self.additional_content[path][content_type]:
                    mkvmerge_mux_string += f'{mux_chapters_only} --chapters "{file_path}" '

                if 'attachments' in self.additional_content[path][content_type]:
                    mkvmerge_mux_string += f'{mux_attachments_only} --attachments -1:all "{file_path}" '

                if 'audio' in self.additional_content[path][content_type]:
                    for track_id in self.additional_content[path][content_type]['audio']:
                        if (self.preset['name'] in
                                self.additional_content[path][content_type]['audio'][track_id]['presets']):
                            track_lang = self.additional_content[path][content_type]['audio'][track_id]['lang']
                            track_name = self.additional_content[path][content_type]['audio'][track_id]['track_name']

                            ffmpeg_cmd = (f'ffmpeg -i \"{file_path}\" -map 0:{track_id} -map_metadata -{track_id} '
                                          f'-map_chapters -{track_id} {self.preset['ffmpeg_audio_params']} '
                                          f'\"{self.mux_video_path}/audio_{track_id}_{track_lang}_{track_name}_'
                                          f'{self.filename}\" -y')

                            return_code, stderr = execute_cmd_ssh(cmd=ffmpeg_cmd, hostname="localhost",
                                                                  ssh_username=self.ssh_username,
                                                                  stdout_queue=self.mux_info_queue,
                                                                  get_pty=True, prefix=f'Transcode/copy audio: '
                                                                                       f'{track_lang} - {track_name}: ')

                            if not return_code == 0:
                                return_code, stderr = execute_cmd_ssh(cmd=ffmpeg_cmd, hostname="localhost",
                                                                      ssh_username=self.ssh_username,
                                                                      stdout_queue=self.mux_info_queue,
                                                                      get_pty=False)
                                self.mux_info_queue.put(cast(str, stderr))
                                self.close()

                            self.mux_info_queue.put("Transcode/copy audio: Done!")

                            if self.additional_content[path][content_type]['audio'][track_id].get('default_flag',
                                                                                                  False):
                                mkvmerge_mux_string += f'--default-track-flag 0 '
                            if self.additional_content[path][content_type]['audio'][track_id].get('original_language',
                                                                                                  False):
                                mkvmerge_mux_string += f'--original-flag 0 '

                            mkvmerge_mux_string += (f'--audio-tracks 0 --language 0:{track_lang} --track-name 0:\"'
                                                    f'{track_name}\" {mux_audio_only} \"{self.mux_video_path}/audio_'
                                                    f'{track_id}_{track_lang}_{track_name}_{self.filename}\" ')

                if 'subtitles' in self.additional_content[path][content_type]:
                    for track_id in self.additional_content[path][content_type]['subtitles']:
                        if self.preset['name'] in (self.additional_content[path][content_type]['subtitles'][track_id]
                                                   ['presets']):
                            sub_lang = self.additional_content[path][content_type]['subtitles'][track_id]['lang']
                            sub_name = self.additional_content[path][content_type]['subtitles'][track_id]['track_name']

                            if self.additional_content[path][content_type]['subtitles'][track_id].get('default_flag',
                                                                                                      False):
                                mkvmerge_mux_string += f'--default-track-flag {track_id} '
                            if self.additional_content[path][content_type]['subtitles'][track_id].get('forced', False):
                                mkvmerge_mux_string += f'--forced-display-flag {track_id} '
                            if self.additional_content[path][content_type]['subtitles'][track_id].get(
                                    'original_language', False):
                                mkvmerge_mux_string += f'--original-flag {track_id} '
                            mkvmerge_mux_string += (f"--subtitle-tracks {track_id} --language {track_id}:"
                                                    f"{sub_lang} --track-name {track_id}:\"{sub_name}\" "
                                                    f"{mux_subtitles_only} \"{file_path}\" ")

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

        if not process.returncode == 0 and not self.encode_job.check_if_exists():
            self.mux_info_queue.put(f"Bad mux! {stderr}")
            self.close()

        else:
            self.mux_info_queue.put(f"Good mux: {self.preset['name']}/{self.filename}")
            shutil.rmtree(self.mux_video_path)

        self.close()


class NodeManager(mp.Process):
    def __init__(self, hostname: str, ssh_username: str,
                 job_list: mp.managers.ListProxy[EncodeJob], job_list_lock: TypedLock,
                 segment_list: mp.managers.DictProxy[VideoSegment, dict[str, bool]],
                 segment_list_lock: TypedLock, stderr_queue: mp.Queue[tuple[str, tuple[int, str | Exception]]],
                 color: str, segment_progress_update_queue: mp.Queue[tuple[VideoSegment, str, int, Any]]) -> None:
        super().__init__()
        self.hostname = hostname
        self.ssh_username = ssh_username
        self.stdout_queue: mp.Queue[str] = mp.Queue()
        self.stderr_queue = stderr_queue
        self.job_list = job_list
        self.job_list_lock = job_list_lock
        self.segment_list = segment_list
        self.segment_list_lock = segment_list_lock
        self.avx512: bool | Exception | None = None
        self.manager = mp.Manager()
        self.init_timestamp = time.time()
        self.color = color
        self.segment_progress_update_queue = segment_progress_update_queue

        self.stats: dict[str, dict[str, Any]] = {
            'cumulative_values': {
                'Frames': self.manager.Value('i', 0),
                'FPS': 0,
                '%RT': 0.0,
                'duration': 0.0,
                'last_update_timestamp': time.time()
            },
            'last_values': {
                'Frames': 0,
                'FPS': 0,
                '%RT': 0.0
            },
            'avg_values': {
                'FPS': self.manager.Value('f', 0),
                '%RT': self.manager.Value('f', 0)
            }
        }

        self.start()

    def run(self) -> None:
        mapped_threads = None
        worker_list: set[NodeWorker] = set()

        while not isinstance(self.avx512, bool) or not isinstance(mapped_threads, dict):
            self.avx512, mapped_threads = get_cpu_info(self.hostname, self.ssh_username, self.stdout_queue)

        shareable_threads = self.manager.dict()

        shareable_threads['numa_nodes'] = self.manager.dict()

        for numa_node in mapped_threads['numa_nodes']:
            shareable_threads['numa_nodes'][numa_node] = self.manager.dict()
            shareable_threads['numa_nodes'][numa_node]['cores'] = self.manager.list()
            shareable_threads['numa_nodes'][numa_node]['cores_available'] = 0
            for core in mapped_threads['numa_nodes'][numa_node]['cores']:
                shareable_threads['numa_nodes'][numa_node]['cores'].append(
                    CPUCore(core, mapped_threads['numa_nodes'][numa_node]['cores'][core]['threads'], self.manager))
                shareable_threads['numa_nodes'][numa_node]['cores_available'] += 1

        last_optimal_core_count = 0

        while True:
            # YOU WILL NOT CHANGE THE LENGTH OF A LIST WHILE ITERATING OVER IT
            worker_del_list = []

            for worker in worker_list:
                while not worker.stdout_queue.empty():
                    stdout = worker.stdout_queue.get()
                    match = re.search(pattern=r'frame=\s*(\d+)\s*fps=\s*([\d\.]+)\s*q=.*?size=\s*([\d\.]+\s*[KMG]i?'
                                              r'B)\s*time=.*?bitrate=\s*([\d\.]+kbits/s)\s*speed=\s*([\d\.x]+)',
                                      string=stdout)
                    if match:
                        worker.stats['cur_values']['Frames'] = int(match.group(1))
                        worker.stats['cur_values']['FPS'] = float(match.group(2))
                        worker.stats['cur_values']['%RT'] = float(match.group(5)[:-1])

                        current_time = time.time()
                        delta_time_worker = current_time - worker.stats['last_values']['last_update_timestamp']
                        delta_time_total = current_time - self.init_timestamp

                        if not worker.stats['cur_values']['Frames'] == worker.stats['last_values']['Frames']:
                            self.stats['cumulative_values']['Frames'].value += (worker.stats['cur_values']['Frames'] -
                                                                                worker.stats['last_values']['Frames'])
                            self.segment_progress_update_queue.put((worker.segment, 'update',
                                                                    worker.stats['cur_values']['Frames'] -
                                                                    worker.stats['last_values']['Frames'], self.color))
                            worker.stats['last_values']['Frames'] = worker.stats['cur_values']['Frames']

                        for metric in ['FPS', '%RT']:
                            self.stats['cumulative_values'][metric] += (worker.stats['cur_values'][metric] *
                                                                        delta_time_worker)
                            self.stats['avg_values'][metric].value = round(self.stats['cumulative_values'][metric] /
                                                                           delta_time_total, 3)

                            worker.stats['last_values'][metric] = worker.stats['cur_values'][metric]
                        worker.stats['last_values']['last_update_timestamp'] = current_time
                        self.stats['cumulative_values']['last_update_timestamp'] = current_time

                if not worker.is_alive():
                    results = worker.results_queue.get()
                    self.segment_list_lock.acquire()
                    if results[1][0] == 0:
                        if worker.segment.check_if_exists(current_user=self.ssh_username):
                            self.segment_list.pop(worker.segment)

                        else:
                            self.segment_list[worker.segment] = {'assigned': False}

                    else:
                        self.segment_list[worker.segment] = {'assigned': False}
                        self.stderr_queue.put(results)

                    self.segment_list_lock.release()
                    worker_del_list.append(worker)

            for worker in worker_del_list:
                worker_list.remove(worker)

            # you have to use .keys for a managed list... Because... Because you just do, okay?!
            for numa_node in shareable_threads['numa_nodes'].keys():
                if last_optimal_core_count <= shareable_threads['numa_nodes'][numa_node]['cores_available']:
                    self.segment_list_lock.acquire()
                    for segment in self.segment_list.keys():
                        if not self.segment_list[segment]['assigned']:
                            if segment.encode_job.preset['optimize_jobs']:
                                # optimized core counts for other codecs coming when I care (or if someone tells me)
                                if segment.encode_job.preset['video_encoder'] == 'libx265':
                                    optimal_cores = 4

                                else:
                                    optimal_cores = len(shareable_threads['numa_nodes'][numa_node]['cores'])

                                last_optimal_core_count = optimal_cores

                                if shareable_threads['numa_nodes'][numa_node]['cores_available'] >= optimal_cores:
                                    core_list = self.manager.list()
                                    for core in shareable_threads['numa_nodes'][numa_node]['cores']:
                                        if len(core_list) < optimal_cores:
                                            if not core.assigned.value:
                                                core_list.append(core)
                                                core.assigned.value = True
                                                shareable_threads['numa_nodes'][numa_node]['cores_available'] -= 1
                                        else:
                                            break

                                    worker_list.add(NodeWorker(node_info=self, shareable_threads=shareable_threads,
                                                               core_list=core_list, segment=segment))
                                    # reassign the entire dict value rather than the bool alone because... that's what
                                    # works with shared dicts
                                    self.segment_list[segment] = {
                                        'assigned': True
                                    }
                                    self.segment_progress_update_queue.put((segment, 'create', 0, self.color))

                                else:
                                    break

                    self.segment_list_lock.release()
            time.sleep(0.1)


class NodeWorker(mp.Process):
    def __init__(self, node_info: NodeManager, shareable_threads: mp.managers.DictProxy[str, Any],
                 core_list: mp.managers.ListProxy[CPUCore], segment: VideoSegment) -> None:
        super().__init__()
        self.stdout_queue: mp.Queue[str] = mp.Queue()
        self.results_queue: mp.Queue[tuple[str, tuple[int, str | Exception]]] = mp.Queue()
        self.node_info = node_info
        self.shareable_threads = shareable_threads
        self.core_list = core_list
        self.segment = segment

        self.stats = {
            'last_values': {
                'Frames': 0,
                'FPS': 0.0,
                '%RT': 0.0,
                'last_update_timestamp': time.time()
            },
            'cur_values': {
                'Frames': 0,
                'FPS': 0.0,
                '%RT': 0.0
            }
        }

        self.start()

    def run(self) -> None:
        if self.segment.encode_job.preset['optimize_jobs']:
            pool_threads = 0
            taskset_threads = "taskset --cpu-list "
            for core in self.core_list:
                for thread in core.threads:
                    taskset_threads += f"{thread},"
                    pool_threads += 1

            taskset_threads = taskset_threads[:-1]

            self.results_queue.put((self.node_info.hostname,
                                    self.segment.encode(hostname=self.node_info.hostname,
                                                        ssh_username=self.node_info.ssh_username,
                                                        stdout_queue=self.stdout_queue, pool_threads=pool_threads,
                                                        avx512=cast(bool, self.node_info.avx512),
                                                        taskset_threads=taskset_threads)))

        for numa_node in self.shareable_threads['numa_nodes'].keys():
            for core in self.core_list:
                if core in self.shareable_threads['numa_nodes'][numa_node]['cores']:
                    core.assigned.value = False
                    self.shareable_threads['numa_nodes'][numa_node]['cores_available'] += 1

        self.close()


class RichHelper:
    def __init__(self, node_list: list[NodeManager], segment_list: mp.managers.DictProxy[VideoSegment, dict[str, bool]],
                 mux_info_queue: mp.Queue[str], global_frames: multiprocessing.managers.ValueProxy[int]) -> None:
        self.node_list = node_list
        self.segment_list = segment_list
        self.manager = mp.Manager()
        self.global_frames = global_frames
        self.cumulative_frames = 0
        self.init_time = time.time()
        self.mux_info_queue = mux_info_queue
        self.node_last_frames_value = {}

        for node in self.node_list:
            self.node_last_frames_value[node] = node.stats['cumulative_values']['Frames'].value

        self.mux_strings_list = [""] * 8
        self.stderr_strings_list = [""] * 8

        self.global_progress = Progress(SpinnerColumn(), *Progress.get_default_columns(), MofNCompleteColumn(),
                                        TransferSpeedColumnFPS())
        self.global_progress_bar = self.global_progress.add_task("Total Progress",
                                                                 total=self.global_frames.value)

    def update_stderr(self, stderr_queue: mp.Queue[tuple[str, tuple[int, str | Exception]]]) -> str:
        stderr_text = ""
        stderr = stderr_queue.get()
        self.stderr_strings_list.append(f"{datetime.now().strftime('%H:%M:%S')} {stderr[0]}: "
                                        f"{str(stderr[1][1]).splitlines()[-1]}")
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
        for node in self.node_list:
            if node.stats['cumulative_values']['Frames'].value > self.node_last_frames_value[node]:
                frame_diff = node.stats['cumulative_values']['Frames'].value - self.node_last_frames_value[node]
                self.cumulative_frames += frame_diff
                self.node_last_frames_value[node] = node.stats['cumulative_values']['Frames'].value
                self.global_progress.update(self.global_progress_bar, advance=frame_diff,
                                            total=self.global_frames.value)

        table = Table(title=cast(str, self.global_progress))
        table.add_column(header="Hostname", min_width=16)
        table.add_column(header="# Frames", min_width=8)
        table.add_column(header="Avg FPS", min_width=7)
        table.add_column(header="Avg %RT", min_width=7)
        for node in self.node_list:
            table.add_row(f'[{node.color}]{node.hostname}',
                          str(node.stats['cumulative_values']['Frames'].value),
                          str(node.stats['avg_values']['FPS'].value), f"{node.stats['avg_values']['%RT'].value}x")
        return table

    def create_segment_progress_table(self, segment_progress_update_queue: mp.Queue[tuple[VideoSegment, str, int, str]],
                                      segment_progress_bar_dict: dict[VideoSegment, dict[str, Any]]
                                      ) -> Table:

        table = Table.grid(expand=True)
        table.add_column()
        table.add_column()
        table.add_column()
        table.add_column()

        while not segment_progress_update_queue.empty():
            segment, command, frame_input, color = segment_progress_update_queue.get()
            if command == 'create':
                segment_progress_bar_dict.update({
                    segment: {
                        'bar_obj': Progress(SpinnerColumn(), TextColumn("{task.description}"),
                                            TimeRemainingColumn(), MofNCompleteColumn(), TransferSpeedColumnFPS()),
                        'bar_marquee': TextMarquee(segment.marquee_string, 8),
                        'bar_color': color
                    }
                })

                segment_progress_bar_dict.update({
                    segment: {
                        'bar_obj': segment_progress_bar_dict[segment]['bar_obj'],
                        'bar_id': segment_progress_bar_dict[segment]['bar_obj'].add_task(
                            description=f'[{segment_progress_bar_dict[segment]['bar_color']}]'
                                        f'{segment_progress_bar_dict[segment]['bar_marquee'].advance()}',
                            total=segment.num_frames),
                        'bar_marquee': segment_progress_bar_dict[segment]['bar_marquee'],
                        'bar_color': segment_progress_bar_dict[segment]['bar_color']
                    }
                })

            elif command == 'update':
                segment_progress_bar_dict[segment]['bar_obj'].update(
                    segment_progress_bar_dict[segment]['bar_id'],
                    description=f'[{segment_progress_bar_dict[segment]['bar_color']}]'
                                f'{segment_progress_bar_dict[segment]['bar_marquee'].advance()}', advance=frame_input)

        temp_column_list = []

        for segment in self.segment_list.keys():
            if self.segment_list[segment]['assigned']:
                # table.add_row(segment_progress_bar_dict[segment]['bar_obj'])
                temp_column_list.append(segment_progress_bar_dict[segment]['bar_obj'])
                if len(temp_column_list) == 3:
                    table.add_row(*temp_column_list)
                    temp_column_list = []

        if len(temp_column_list) > 0:
            table.add_row(*temp_column_list)

        return table


class VideoSegment:
    def __init__(self, encode_job: EncodeJob, source_fullpath: str) -> None:
        self.encode_job = encode_job
        self.source_fullpath = source_fullpath
        self.source_filename = os.path.basename(self.source_fullpath)
        # self.num_frames: int = num_frames  # why? ---------------------------------------------> ^ ???
        self.file_output_fstring = (f'{self.encode_job.out_path}/temp/{self.encode_job.preset['name']}/'
                                    f'{self.encode_job.filename}/{self.source_filename}')

        self.marquee_string: str = (f'   <-- {self.encode_job.preset['name']} - {self.encode_job.filename} - '
                                    f'{self.source_fullpath} -->   ')

        self.duration = get_duration(self.source_fullpath)
        self.num_frames: int = round(self.duration * self.encode_job.framerate)

    # required to index objects in managed dictionaries without exe crashing to desktop
    def __hash__(self) -> int:
        return hash(self.file_output_fstring)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VideoSegment) and self.file_output_fstring == other.file_output_fstring

    def check_if_exists(self, current_user: str) -> bool:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            '-show_entries', 'format=duration', self.file_output_fstring
        ]
        if os.path.isfile(self.file_output_fstring):
            duration_proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if duration_proc.returncode == 0:
                duration = round(float(duration_proc.stdout.strip()), 3)
                frame_diff = seconds_to_frames(duration, self.encode_job.framerate) - self.num_frames

                # fudge factor for fuckin bullshit ffmpeg timestamps
                if -3 <= frame_diff <= 3:
                    self.encode_job.tally_completed_segments(self.file_output_fstring, current_user)
                    return True
                else:
                    # print(f"Duration not equal: Expected {self.num_frames}, got "
                    #       f"{seconds_to_frames(duration, self.encode_job.framerate)}")
                    # print(f"{self.encode_job.framerate}")
                    # print(f"File: {self.file_output_fstring}")
                    os.remove(self.file_output_fstring)
                    return False
            else:
                # print(f"Malformed file: {self.file_output_fstring}")
                os.remove(self.file_output_fstring)
                return False
        else:
            return False

    def encode(self, hostname: str, ssh_username: str, stdout_queue: mp.Queue[str],
               pool_threads: int, avx512: bool, taskset_threads: str = "") -> tuple[int, str | Exception]:
        cmd = (f'{taskset_threads} ffmpeg -i \"{self.source_fullpath}\" -c:v {self.encode_job.preset['video_encoder']} '
               f'{self.encode_job.preset['ffmpeg_video_params']} ')

        if self.encode_job.preset['video_encoder'] == 'libx265':
            cmd += f'-x265-params \"{self.encode_job.preset['encoder_params']}'

            if avx512:
                cmd += f':asm=avx512'

            if self.encode_job.preset['optimize_jobs']:
                cmd += f':pmode=1:frame-threads=1:pools=\'{pool_threads}\'\" \"{self.file_output_fstring}\"'

            else:
                cmd += f'\" \"{self.file_output_fstring}\"'

        return_code, stderr = execute_cmd_ssh(cmd=cmd, hostname=hostname, ssh_username=ssh_username,
                                              stdout_queue=stdout_queue, get_pty=True)

        # # paramiko can only get ffmpeg stdout with get_pty=True... But it can only get stderr with get_pty=False...
        if not return_code == 0:
            return_code, stderr = execute_cmd_ssh(cmd=cmd, hostname=hostname, ssh_username=ssh_username,
                                                  stdout_queue=stdout_queue, get_pty=False)

        return return_code, stderr


def execute_cmd_ssh(cmd: str, hostname: str, ssh_username: str, get_pty: bool, stdout_queue: mp.Queue[str] = mp.Queue(),
                    prefix: str = "") -> tuple[int, str | Exception]:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname=hostname, username=ssh_username, key_filename=f"/home/{ssh_username}/.ssh/id_rsa.pub")

    except Exception as e:
        client.close()
        return -1, e

    client.invoke_shell()
    stdin, stdout, stderr = client.exec_command(cmd, get_pty=get_pty)
    line = b''
    while not stdout.channel.exit_status_ready() or stdout.channel.recv_ready():
        for byte in iter(lambda: stdout.read(1), b""):
            line += byte
            if byte == b'\n' or byte == b'\r':
                stdout_queue.put(f"{prefix}{line.decode('utf-8', errors='replace').rstrip()}")
                line = b''

    client.close()
    return stdout.channel.recv_exit_status(), stderr.read().decode('utf-8')


class SegmenterExecutor(mp.Process):
    def __init__(self, cmd_string: str, stdout_queue: mp.Queue[str], ssh_username: str) -> None:
        super().__init__()
        self.cmd_string = cmd_string
        self.stdout_queue = stdout_queue
        self.ssh_username = ssh_username
        self.start()

    def run(self) -> None:
        execute_cmd_ssh(cmd=self.cmd_string, hostname='localhost', ssh_username=self.ssh_username, get_pty=True,
                        stdout_queue=self.stdout_queue)

        self.close()


class EncodeJob:
    def __init__(self, input_file: str, preset: dict[str, str], out_path: str, filename: str,
                 additional_content: dict[str, Any], file_index: int, mux_info_queue: mp.Queue[str],
                 segment_tally_counter: mp.managers.ValueProxy[int],
                 completed_segment_filename_list: mp.managers.ListProxy[str], ssh_username: str) -> None:
        self.input_file = input_file
        self.job_name = f'{preset['name']}_{filename}'
        self.framerate_numerator, self.framerate_denominator = get_framerate(self.input_file)
        self.framerate = self.framerate_numerator / self.framerate_denominator
        self.preset = preset
        self.out_path = out_path
        self.filename = filename
        self.segments_completed = segment_tally_counter
        self.completed_segment_filename_list = completed_segment_filename_list
        self.additional_content = additional_content
        self.mux_info_queue = mux_info_queue
        self.file_index = file_index
        self.temp_segment_dir = f'{self.out_path}/temp/source_segments/{self.filename}/'
        self.segment_txt = f'{self.temp_segment_dir}list.txt'
        self.ssh_username = ssh_username
        self.num_segments = 0
        self.duration = get_duration(self.input_file)
        self.frames_total = round(self.duration * self.framerate)

        if not os.path.isdir(f'{self.out_path}/temp/source_segments/{self.filename}/'):
            os.makedirs(f'{self.out_path}/temp/source_segments/{self.filename}/')

        if not os.path.isdir(f'{self.out_path}/temp/{self.preset['name']}/{self.filename}/'):
            os.makedirs(f'{self.out_path}/temp/{self.preset['name']}/{self.filename}/')

        if not os.path.isdir(f'{self.out_path}/output/{self.preset['name']}/'):
            os.makedirs(f'{self.out_path}/output/{self.preset['name']}/')

    def check_if_exists(self) -> bool:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            '-show_entries', 'format=duration', f'{self.out_path}/output/{self.preset['name']}/{self.filename}'
        ]

        if os.path.isfile(f'{self.out_path}/output/{self.preset['name']}/{self.filename}'):
            duration_proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if duration_proc.returncode == 0:
                if not duration_proc.stdout.strip() == "N/A":
                    duration = round(float(duration_proc.stdout.strip()), 3)
                    if seconds_to_frames(duration, self.framerate) == self.frames_total:
                        return True
                    else:
                        # print(f'expected {self.frames_total}, got {seconds_to_frames(duration, self.framerate)}')
                        # print(f'{duration}, {self.framerate}')
                        return False
                else:
                    return False
            else:
                return False
        else:
            return False

    def tally_completed_segments(self, filename: str, current_user: str) -> None:
        self.segments_completed.value += 1
        self.completed_segment_filename_list.append(filename)

        if self.segments_completed.value == self.num_segments:
            # no longer needs to be shared, so ignore!
            self.completed_segment_filename_list = sorted(self.completed_segment_filename_list,  # type: ignore
                                                          key=lambda x: float(match.group(1)) if
                                                          (match := re.search(r'/(\d+\.\d+)-', x)) else 0.0)

            MuxWorker(self.preset, self.out_path, self.filename, cast(list[str], self.completed_segment_filename_list),
                      self.additional_content, self.file_index, self.mux_info_queue, current_user, self)

    def create_segment_encode_list(self, segment_list: mp.managers.DictProxy[VideoSegment, dict[str, bool]],
                                   segment_list_lock: TypedLock,
                                   global_frames: multiprocessing.managers.ValueProxy[int]) -> None:
        last_line = ""
        if os.path.exists(f'{self.segment_txt}'):
            with open(self.segment_txt, 'r') as file:
                lines = file.readlines()
                if lines:
                    last_line = lines[-1].strip()

        if last_line != 'completed':
            segment_cmd = (f'ffmpeg -i \"{self.input_file}\" -map 0 -c copy -f segment -reset_timestamps 1 '
                           f'-segment_list_type flat -segment_time 30 -segment_list \"{self.segment_txt}\" '
                           f'\"{self.temp_segment_dir}%05d_{self.filename}\"')

            # the ssh stdout struggles will continue until morale improves
            stdout_queue: mp.Queue[str] = mp.Queue()
            segmenter = SegmenterExecutor(segment_cmd, stdout_queue, self.ssh_username)

            while segmenter.is_alive():
                while not stdout_queue.empty():
                    stdout = stdout_queue.get().strip()
                    match = re.search(r"Opening '(.+?)' for writing", stdout)

                    if match:
                        line = match.group(1)
                        if line.endswith('list.txt'):
                            last_line = line

                        elif not last_line.endswith('list.txt'):
                            self.add_segment(source_fullpath=last_line, segment_list=segment_list,
                                             segment_list_lock=segment_list_lock, global_frames=global_frames)
                            last_line = line
                            self.num_segments += 1

                        else:
                            last_line = line

            with open(self.segment_txt, 'a') as file:
                file.write('completed')

        else:
            lines = lines[:-1]
            self.num_segments = len(lines)

            for line in lines:
                line = line.strip()
                self.add_segment(source_fullpath=f'{self.temp_segment_dir}{line}', segment_list=segment_list,
                                 segment_list_lock=segment_list_lock, global_frames=global_frames)

    def add_segment(self, source_fullpath: str, segment_list: mp.managers.DictProxy[VideoSegment, dict[str, bool]],
                    segment_list_lock: TypedLock, global_frames: multiprocessing.managers.ValueProxy[int]) -> None:
        segment = VideoSegment(encode_job=self, source_fullpath=source_fullpath)
        if not segment.check_if_exists(current_user=self.ssh_username):
            segment_list_lock.acquire()
            segment_list.update({
                segment: {
                    'assigned': False
                }
            })
            segment_list_lock.release()

            global_frames.value += segment.num_frames


def get_cpu_info(hostname: str, username: str,
                 stdout_queue: mp.Queue[str]) -> tuple[bool | Exception, dict[Any, Any] | Exception]:
    avx512_dict: dict[Any, Any] = {}
    core_info_dict: dict[Any, Any] = {}

    # tragic: for no apparent reason, paramiko randomly returns a very, very small fraction of the full stdout
    # so, we simply retry until we get uncorrupted json :)
    while len(avx512_dict) == 0:
        try:
            avx512_info = execute_cmd_ssh(cmd="lscpu --json", hostname=hostname, ssh_username=username,
                                          stdout_queue=stdout_queue, get_pty=True)
            avx512_stdout = ""
            if avx512_info[0] == 0:
                while not stdout_queue.empty():
                    avx512_stdout += stdout_queue.get()

            elif avx512_info[0] == -1:
                # print(avx512_info[1])
                # print(f"Failed to connect to {hostname}.")
                return cast(Exception, avx512_info[1]), core_info_dict

            avx512_dict = json.loads(avx512_stdout)

        except json.decoder.JSONDecodeError:
            continue

    # I can't even with this data structure. This retrieves the CPU feature flag str to determine avx512 availability.
    for child in avx512_dict['lscpu'][2]['children'][0]['children']:
        if child['field'] == 'Flags:':
            avx512 = True if 'avx512' in child['data'] else False
            # print(f"AVX512 Capability Detected: {avx512}")

    while len(core_info_dict) == 0:
        try:
            core_info = execute_cmd_ssh(cmd="lscpu --json --extended", hostname=hostname, ssh_username=username,
                                        stdout_queue=stdout_queue, get_pty=True)
            core_info_stdout = ""
            if core_info[0] == 0:
                while not stdout_queue.empty():
                    core_info_stdout += stdout_queue.get()

            elif core_info[0] == -1:
                # print(core_info[1])
                # print(f"Failed to connect to {hostname}.")
                break

            core_info_dict = json.loads(core_info_stdout)

        except json.decoder.JSONDecodeError:
            continue

    # Restructure the returned thread data. This is the only way to properly deal with asymmetrical CPUs that have
    # cores with variable thread counts, and it makes NUMA easier.

    mapped_threads: dict[str, dict[str, dict[str, dict[str, dict[str, list[str]]]]]] = {
        'numa_nodes': {
            '0': {
                'cores': {
                    '0': {
                        'threads': ['0']
                    }
                }
            }
        }
    }

    for thread in core_info_dict['cpus']:
        node = str(thread['node']) if 'node' in thread else '0'

        if node not in mapped_threads['numa_nodes']:
            mapped_threads['numa_nodes'].update({
                str(node): {
                    'cores': {
                        str(thread['core']): {
                            'threads': [
                                str(thread['cpu'])
                            ]
                        }
                    }
                }
            })

        if avx512_dict['lscpu'][2]['data'] == 'Qualcomm':
            # nonobad ugly hack due to lscpu misreporting core/thread relationships on my snapdragon x wsl install
            # no multithreading so no need to check 2x. no clue if it would work properly on a native linux install.
            if str(thread['cpu']) not in mapped_threads['numa_nodes'][node]['cores']:
                mapped_threads['numa_nodes'][node]['cores'].update({
                    str(thread['cpu']): {
                        'threads': [
                            str(thread['cpu'])
                        ]
                    }
                })

        else:
            if str(thread['core']) not in mapped_threads['numa_nodes'][node]['cores']:
                mapped_threads['numa_nodes'][node]['cores'].update({
                    str(thread['core']): {
                        'threads': [
                            str(thread['cpu'])
                        ]
                    }
                })
            if str(thread['cpu']) not in mapped_threads['numa_nodes'][node]['cores'][str(thread['core'])]['threads']:
                mapped_threads['numa_nodes'][node]['cores'][str(thread['core'])]['threads'].append(str(thread['cpu']))

    # noinspection PyUnboundLocalVariable
    return avx512, mapped_threads


class CPUCore:
    def __init__(self, core_id: str, threads: tuple[int, ...], manager: mp.managers.SyncManager) -> None:
        self.core_id = core_id
        self.threads = threads
        self.assigned = manager.Value('b', False)

    def __hash__(self) -> int:
        return hash(self.core_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CPUCore) and self.core_id == other.core_id


class JobExecutor(mp.Process):
    def __init__(self, job_list: mp.managers.ListProxy[EncodeJob], job_list_lock: TypedLock,
                 segment_list: mp.managers.DictProxy[VideoSegment, dict[str, bool]],
                 segment_list_lock: TypedLock, in_path: str, out_path: str, ssh_username: str,
                 segment_tally_list: mp.managers.ListProxy[mp.managers.ValueProxy[int]],
                 manager: multiprocessing.managers.SyncManager, config: dict[str, Any],
                 mux_info_queue: mp.Queue[str], global_frames: multiprocessing.managers.ValueProxy[int]) -> None:
        super().__init__()
        self.job_list = job_list
        self.job_list_lock = job_list_lock
        self.segment_list = segment_list
        self.segment_list_lock = segment_list_lock
        self.in_path = in_path
        self.out_path = out_path
        self.ssh_username = ssh_username
        self.segment_tally_list = segment_tally_list
        self.manager = manager
        self.config = config
        self.mux_info_queue = mux_info_queue
        self.global_frames = global_frames
        self.start()

    def run(self) -> None:
        completed_segment_filename_list = self.manager.list()

        for x, file in enumerate(sorted(os.listdir(self.in_path)), start=0):
            file_fullpath = os.path.join(self.in_path, file)
            for preset in self.config['presets']:
                self.segment_tally_list.append(self.manager.Value('i', 0))
                completed_segment_filename_list.append(self.manager.list())
                job = EncodeJob(input_file=file_fullpath, preset=preset, out_path=self.out_path, filename=file,
                                additional_content=self.config['additional_content'], file_index=x,
                                mux_info_queue=self.mux_info_queue, segment_tally_counter=self.segment_tally_list[-1],
                                completed_segment_filename_list=completed_segment_filename_list[-1],
                                ssh_username=self.ssh_username)

                if not job.check_if_exists():
                    self.job_list.append(job)
                    job.create_segment_encode_list(self.segment_list, self.segment_list_lock, self.global_frames)

                time.sleep(5)

        self.close()


def main() -> None:
    ssh_username = os.getlogin()

    parser = argparse.ArgumentParser(description="Load a YAML configuration file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument('--in_path', type=str, required=True, help="Path containing input video files.")
    parser.add_argument('--out_path', type=str, required=True, help="Path to output video files.")

    args = parser.parse_args()
    args.out_path = args.out_path.rstrip('/')

    config = load_config(args.config)
    print("Loaded configuration:")
    print(config)
    print(args.in_path)
    print(args.out_path)

    manager = mp.Manager()

    global_frames = manager.Value('i', 0)

    # "typing" hell
    stderr_queue: mp.Queue[tuple[str, tuple[int, str | Exception]]] = manager.Queue()  # type: ignore
    mux_info_queue: mp.Queue[str] = manager.Queue()  # type: ignore

    job_list = manager.list()
    job_list_lock = mp.Manager().Lock()

    segment_list = manager.dict()
    segment_list_lock = mp.Manager().Lock()

    for path in config['additional_content']:
        config['additional_content'][path]['file_list'] = sorted(os.listdir(path))

    # shared list that contains shareable values for counting segments that are finished encoding
    segment_tally_list = manager.list()

    segment_progress_update_queue: mp.Queue[tuple[VideoSegment, str, int, Any]] = manager.Queue()  # type: ignore
    segment_progress_bar_dict: dict[VideoSegment, dict[str, Any]] = {}

    job_executor = JobExecutor(job_list=job_list, job_list_lock=job_list_lock, segment_list=segment_list,
                               segment_list_lock=segment_list_lock, in_path=args.in_path, out_path=args.out_path,
                               ssh_username=ssh_username, segment_tally_list=segment_tally_list, manager=manager,
                               config=config, mux_info_queue=mux_info_queue, global_frames=global_frames)

    node_list = []

    # nobody would ever use this with more than... fifteen... nodes, right? right? kinda based tho ngl...
    terminal_colors = ["red", "green", "yellow", "blue", "magenta", "cyan", "white", "bright_black", "bright_red",
                       "bright_green", "bright_yellow", "bright_blue", "bright_magenta", "bright_cyan", "bright_white"]

    for worker_hostname in config['nodes']:
        randcolor = terminal_colors[random.randint(0, len(terminal_colors) - 1)]
        node_list += [NodeManager(hostname=worker_hostname, ssh_username=ssh_username,
                                  job_list=job_list, job_list_lock=job_list_lock, segment_list=segment_list,
                                  segment_list_lock=segment_list_lock, stderr_queue=stderr_queue, color=randcolor,
                                  segment_progress_update_queue=segment_progress_update_queue)]
        terminal_colors.remove(randcolor)

    tui = RichHelper(node_list, segment_list, mux_info_queue, global_frames)
    layout = Layout()
    # layout.split_column(Layout(name="header", size=4), Layout(name="table"), Layout(name="footer", size=8))
    layout.split_column(Layout(name="table"), Layout(name="segment_tracker"), Layout(name="stderr", size=10),
                        Layout(name="Mux Info", size=10))
    layout['table'].update(tui.create_node_table())
    layout['segment_tracker'].update(tui.create_segment_progress_table(segment_progress_update_queue,
                                                                       segment_progress_bar_dict))
    layout['stderr'].update(Panel("Nothing here yet!", title="Errors"))
    layout['Mux Info'].update(Panel("Nothing here yet...", title="Mux Info"))

    try:
        with Live(layout, refresh_per_second=1, screen=True):
            while True:
                layout['table'].update(tui.create_node_table())
                segment_list_lock.acquire()
                layout['segment_tracker'].update(tui.create_segment_progress_table(segment_progress_update_queue,
                                                                                   segment_progress_bar_dict))
                segment_list_lock.release()
                while not mux_info_queue.empty():
                    layout['Mux Info'].update(Panel(tui.update_mux_info(mux_info_queue), title="Mux Info"))
                while not (stderr_queue.empty()):
                    layout['stderr'].update(Panel(tui.update_stderr(stderr_queue), title="stderr"))
                time.sleep(1)

        # while True:
        #     time.sleep(1)
        #     while not mux_info_queue.empty():
        #         print(mux_info_queue.get())

    except KeyboardInterrupt:
        kill_cmd = (
            f'pgrep -f \"ffmpeg.*{args.in_path}.*{args.out_path}\" | xargs kill -9'
        )
        for process in mp.active_children():
            process.join()
            process.close()

        for node in node_list:
            execute_cmd_ssh(cmd=kill_cmd, hostname=node.hostname, ssh_username=node.ssh_username, get_pty=True)

        sys.exit()


if __name__ == "__main__":
    main()
