import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
sess = tf.Session(config=tf_config)

import numpy as np
import pandas as pd
import utils
from config import config
from rnn_model import RNNLanguageModel


txt_files = os.listdir(config['txt_dir'])
np.random.seed(10)
txt_files = np.random.choice(txt_files, size = config['num_files'], replace = False)

midi_sequence = utils.text_file_to_seq(txt_files, config['txt_dir'])
print("[INFO] Length of MIDI sequence:", len(midi_sequence))
print("[INFO] Number of unique tokens in MIDI sequence:", len(np.unique(midi_sequence)))

midi_sequence = pd.Series(midi_sequence)
token_counts = midi_sequence.value_counts()
discard_tokens = list(token_counts[token_counts <= config['min_token_freq']].index)
keep_tokens = list(token_counts[token_counts > config['min_token_freq']].index)
print("[INFO] Tokens that occur </= " + str(config['min_token_freq']) + " times:", len(discard_tokens))

# Array with the list of sequences
midi_sequence = np.array(midi_sequence[midi_sequence.isin(keep_tokens)])
seq_len = len(midi_sequence)
print("[INFO] Length of MIDI sequence after removal of such tokens:", seq_len)

X, y, note_index, idx_to_note = utils.tokenize_midi_seq(midi_sequence, config['prev_n_tokens'])
config['vocab_size'] = np.min([config['vocab_size'], len(note_index)])

if config['load_checkpoint'] != 0:
    checkpoint = config['model_checkpoint_dir'] + str(config['load_checkpoint']) + '.ckpt'
else:
    checkpoint = tf.train.get_checkpoint_state(os.path.dirname('models/checkpoint')).model_checkpoint_path

model = RNNLanguageModel(config)

sample_gen_seq = model.generate_midi_array(X, checkpoint)

back_convert = np.vectorize(lambda x: idx_to_note[x])
sample_gen_seq = back_convert(sample_gen_seq)

idx = np.random.choice(range(config['batch_size']))
utils.list_to_midi(sample_gen_seq[idx], config['generated_dir'], 'generated')
