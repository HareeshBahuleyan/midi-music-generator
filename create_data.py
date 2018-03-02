import os
from config import config
from utils import *


def create_seq_data(config):
    midi_files = os.listdir(config['midi_dir'])
    print("[INFO] Number of MIDI files:", len(midi_files))

    print("[INFO] Reading MIDI Files and converting to text")
    for file_name in midi_files:
        midi_chunk_str = midi_to_text(file_name, config['midi_dir'])

        with open(config['txt_dir'] + file_name[:-3] + 'txt', 'w') as f:
            f.write(midi_chunk_str)


if __name__ == '__main__':
    create_seq_data(config)
