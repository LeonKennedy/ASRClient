#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2022, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@gmail.com
@file: main.py
@time: 2022/8/11 10:09
@desc:
"""
import argparse
import sys
import torch
import keyboard
import pyaudio
from net import ConvEmformer
from corpus_map import ELEMENTS
import threading

model = ConvEmformer(len(ELEMENTS), 1)
state = torch.load("model.pt", map_location=torch.device('cpu'))
model.load_state_dict(state["model_state_dict"])
model.eval()
print("loss: ", state["loss"])
input_length, move_length = model.input_size() #b=6480 5120
print(f"{input_length=} {move_length=}")

is_recording = False
enable_trigger_record = True


def on_press_release(x):
    """Keyboard callback function."""
    global is_recording, enable_trigger_record, state
    press = keyboard.KeyboardEvent('down', 28, 'space')
    release = keyboard.KeyboardEvent('up', 28, 'space')
    if x.event_type == 'down' and x.name == press.name:
        if (not is_recording) and enable_trigger_record:
            sys.stdout.write("Start Recording ... ")
            sys.stdout.flush()
            is_recording = True

    if x.event_type == 'up' and x.name == release.name:
        if is_recording:
            is_recording = False
            state = None


data_list = bytearray()

chunk_size = input_length
bytes_chunk_size = chunk_size * 2
bytes_move_length = move_length * 2
state = None


def show(res):
    a = [ELEMENTS[i] for i in res if i != 0]
    if a:
        print(res, a)
    else:
        print(res)


def predict(data: bytearray):
    global state
    chunk = data[:bytes_chunk_size]
    float_chunk = torch.frombuffer(bytes(chunk), dtype=torch.int16) / 65536
    res, state = model.stream_forward(float_chunk.unsqueeze(0), state)
    show(res)
    return data[bytes_move_length:]


def callback(in_data: bytes, frame_count, time_info, status):
    print(f"thread count: {threading.active_count()}")
    global data_list, is_recording, enable_trigger_record
    if is_recording:
        data_list.extend(in_data)
    #     enable_trigger_record = False
    # enable_trigger_record = True
    while len(data_list) > bytes_chunk_size:
        data_list = predict(data_list)
    return (in_data, pyaudio.paContinue)


def main():
    # prepare audio recorder
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        stream_callback=callback)
    stream.start_stream()
    print("OK:")

    # prepare keyboard listener
    while (1):
        keyboard.hook(on_press_release)
        if keyboard.record('esc'):
            break

    # close up
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    main()
