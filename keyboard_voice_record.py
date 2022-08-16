#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2022, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@gmail.com
@file: keyboard_voice_record.py
@time: 2022/8/12 15:30
@desc:
"""
import collections
import sys
import time
from io import BytesIO
import threading
from typing import Union
import webrtcvad
import keyboard
import pyaudio
import queue

enable_trigger_record = True


class VadQueue:

    def __init__(self, read_queue: queue.Queue, event: threading.Event, write_queue: queue.Queue, wav_size: int,
                 move_size:int,
                 stop_event: threading.Event, model_state,
                 sr=16000, duration=30, bit_size=2):
        self._vad = webrtcvad.Vad(2)
        self._read_queue = read_queue
        self._write_queue = write_queue
        self._stop_event = stop_event
        self._event = event
        self._sr = sr
        self.frame_size = int(duration / 1000 * sr)
        self._bit_size = bit_size
        self.buffer_size: int = self.frame_size * self._bit_size
        self.wav_size = wav_size
        self.move_size = move_size

        self._window_size = 20
        self._ring_flag = collections.deque(maxlen=self._window_size)
        self._ring_buffer = collections.deque()
        self._cache = bytes()
        self.model_state = model_state
        self._pre_state = False

    def run(self):
        while not self._stop_event.is_set():
            try:
                chunk = self._read_queue.get_nowait()
                self.append_and_input(chunk)
            except queue.Empty:
                time.sleep(0.1)
        print("vad queue end")

    def append_and_input(self, chunk):
        if len(self._ring_flag) == self._window_size:
            c = self._ring_buffer.popleft()
            self._cache += c
            if sum(self._ring_flag) <= 0.2 * self._window_size:
                self._write_queue.put(b'')
            # if sum(self._ring_flag) >= 0.9 * self._window_size:
            #     if not self._pre_state:
            #         print("[检测到声音]")
            #         self._pre_state = True
            #     c = self._ring_buffer.popleft()
            #     self._cache += c
            # else:
            #     if self._pre_state:
            #         print("[检测到中断]")
            #         self._pre_state = False
            #         last_buffer = self._cache + b''.join(self._ring_buffer)
            #         self._cache = bytes()
            #         self._ring_buffer.clear()
            #         last_buffer = self.padding_size(last_buffer)
            #         self._write_queue.put(last_buffer)
            #         self._write_queue.put(b'')
            #         self._ring_flag.clear()
        flag = self._vad.is_speech(chunk, sample_rate=self._sr)
        self._ring_flag.append(flag)
        self._ring_buffer.append(chunk)

        if len(self._cache) >= self.wav_size:
            self._write_queue.put(self._cache[:self.wav_size])
            self._cache = self._cache[self.move_size:]

    def padding_size(self, chunk: bytes) -> bytes:
        if len(chunk) >= self.wav_size:
            return chunk[:self.wav_size]
        else:
            lack_size = self.wav_size - len(chunk)
            return chunk + b'\x00' * lack_size


#
# class FEFF:
#     def __init__(self):
#         self._b = BytesIO()
#         self._tail_index = 0
#         self._read_index = 0
#         self._lock = threading.Lock()
#         self._q = queue.Queue()
#         self._pre = b''
#         self.vad_queue = VadQueue(self._q)
#
#     def write(self, b: bytes):
#         self._pre += b
#         while len(self._pre) > self.vad_queue.buffer_size:
#             chunk, self._pre = self._pre[:self.vad_queue.buffer_size], self._pre[self.vad_queue.buffer_size:]
#             self._q.put(chunk)
#
#         # while b:
#         #     chunk, b = b[:30]
#         # with self._lock:
#         #     self._b.seek(self._tail_index)
#         #     self._b.write(b)
#         # self._tail_index += len(b)
#
#     def read(self, size: int, move: int) -> Union[bytes]:
#         chunk = self.vad_queue.read()
#
#         with self._lock:
#             self._b.seek(self._read_index)
#             if size > 0:
#                 res = self._b.read(size)
#             else:
#                 res = self._b.read()
#                 return res
#
#             if len(res) == size:
#                 self._read_index += move
#                 return res
#
#     def read_all(self) -> bytes:
#         self._b.seek(0)
#         return self._b.read()
#
#     def __len__(self) -> int:
#         return self._tail_index
#
#     def rest(self) -> int:
#         return self._tail_index - self._read_index
#
#     def clean(self):
#         pass


class KeyboardVoiceRecorder:

    def __init__(self, is_recording: threading.Event, queue: queue.Queue, model_queue: queue.Queue, chunk_size: int):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                                  stream_callback=self._stream_callback)
        self.is_recording = is_recording
        self.queue = queue
        self.model_queue = model_queue
        self.chunk_size = chunk_size
        self._pre = bytes()

    def _stream_callback(self, in_data: bytes, frame_count, time_info, status):
        if self.is_recording.is_set():
            self._pre += in_data
            while len(self._pre) > self.chunk_size:
                chunk, self._pre = self._pre[:self.chunk_size], self._pre[self.chunk_size:]
                self.queue.put(chunk)
        else:
            if self._pre:
                self._pre = bytes()
        return (in_data, pyaudio.paContinue)

    def _keyboard_listen_callback(self, x):
        press = keyboard.KeyboardEvent('down', 28, 'm')
        release = keyboard.KeyboardEvent('up', 28, 'n')
        if x.event_type == 'down' and x.name == press.name:
            if not self.is_recording.is_set():
                sys.stdout.write("\nStart Recording ... \n")
                sys.stdout.flush()
                self.is_recording.set()

        if x.event_type == 'up' and x.name == release.name:
            if self.is_recording.is_set():
                sys.stdout.write("\nEnd Recording ... \n")
                sys.stdout.flush()
                self.is_recording.clear()
                self.model_queue.put(b'')

    def run(self):
        print("[Init Start]")
        self.stream.start_stream()

        while 1:
            keyboard.hook(self._keyboard_listen_callback)
            if keyboard.record('esc'):
                break
        self.close()

    def close(self):
        print("[Bye End]")
        self.stream.stop_stream()
        print("stop stream")
        self.stream.close()
        print("close stream")
        self.p.terminate()
        print("pyaudio  terminate")
