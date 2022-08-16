#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2022, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@gmail.com
@file: run.py.py
@time: 2022/8/12 15:31
@desc:
"""
import queue
import threading
import time
from io import BytesIO

import torch
from keyboard_voice_record import KeyboardVoiceRecorder, VadQueue
from conv_emformer import get_model
from corpus_map import ELEMENTS

model = get_model()
input_length, move_length = model.input_size()
print(f"{input_length=} {move_length=}")
state = None


def show(res):
    if res.sum() == 0:
        return
    a = [ELEMENTS[i] for i in res if i != 0]
    if a:
        print(res, a)
    else:
        print(res)


def check_data(read_queue: queue.Queue, event: threading.Event):
    global state
    while not event.is_set():
        try:
            chunk = read_queue.get(block=False)
            if chunk == b'':
                state = None
                continue
            float_chunk = torch.frombuffer(chunk, dtype=torch.int16) / 65536
            res, state = model.stream_forward(float_chunk.unsqueeze(0), state)
            show(res)
        except queue.Empty:
            time.sleep(0.1)
    print("model predit end")


def run():
    is_recorder = threading.Event()
    record_to_vad_queue = queue.Queue()
    vad_to_model_queue = queue.Queue()
    stop_flag = threading.Event()

    vad_queue = VadQueue(record_to_vad_queue, is_recorder, vad_to_model_queue, input_length * 2, move_length * 2, stop_flag, state)
    recorder = KeyboardVoiceRecorder(is_recorder, record_to_vad_queue, vad_to_model_queue, vad_queue.buffer_size)
    task = threading.Thread(target=recorder.run)
    t2 = threading.Thread(target=check_data, args=(vad_to_model_queue, stop_flag))
    t2.start()
    t3 = threading.Thread(target=vad_queue.run)
    t3.start()
    task.start()
    task.join()
    stop_flag.set()
    t2.join()
    t3.join()


if __name__ == "__main__":
    run()
