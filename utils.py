import midi
import numpy as np
from keras.preprocessing.text import Tokenizer


def get_batches(X, y, batch_size, prev_n_tokens):
    for batch_i in range(len(X) // batch_size):
        start_i = batch_i * batch_size
        x_ = X[start_i:start_i + batch_size]
        y_ = y[start_i:start_i + batch_size]

        source_length = np.repeat([prev_n_tokens], repeats=batch_size)

        yield x_, y_, source_length


def text_file_to_seq(txt_files, txt_dir):
    sequence = []
    for file_name in txt_files:
        with open(txt_dir + file_name, 'r') as f:
            chunk_str = f.read()

        # Split by space and take all tokens starting from index 1
        # Index 0 has the token that specifies the MIDI file resolution
        # MIDI file resolution can be added artificially to the start of the generated sequence
        sequence.extend(chunk_str.split()[1:])

    return sequence


def tokenize_midi_seq(midi_sequence, prev_n_tokens):
    seq_len = len(midi_sequence)
    X = []
    y = []
    for i in range(seq_len - prev_n_tokens):
        # Relevant sub-sequence
        sub_seq = midi_sequence[i: i + prev_n_tokens + 1]
        X.append(sub_seq[:prev_n_tokens])
        y.append(sub_seq[prev_n_tokens])

    X = np.array(X)
    y = np.array(y)

    # Shuffle X and y in unison
    p = np.random.permutation(len(y))
    X = X[p]
    y = y[p]

    # Convert word sequence to a sequence of ints
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(midi_sequence)

    # Re-index to make it start from 0 instead of 1
    note_index = dict()
    for i, note in enumerate(dict(tokenizer.word_index).keys()):
        note_index[note] = i

    tokenizer.word_index = note_index

    X = [' '.join(x) for x in X]
    X = tokenizer.texts_to_sequences(X)
    X = np.array(X, dtype='int32')

    y = tokenizer.texts_to_sequences(y)
    y = np.array(y, dtype='int32')
    y = np.reshape(y, (y.shape[0],))

    idx_to_note = dict((i, note) for note, i in note_index.items())

    return X, y, tokenizer.word_index, idx_to_note


def midi_to_text(file_name, midi_dir):
    pattern = midi.read_midifile(midi_dir + file_name)
    chunk_str_list = []

    chunk_str = "rs_" + str(pattern.resolution)
    chunk_str_list.append(chunk_str)

    max_idx = np.argmax([len(p) for p in pattern])

    for i, chunk in enumerate(pattern[max_idx]):

        chunk_str = ""

        if chunk.name == "Note On":
            chunk_str = chunk_str + str(chunk.tick) + "_" + "no" + "_" + str(chunk.pitch) + "_" + str(chunk.velocity)

            chunk_str_list.append(chunk_str)

        elif chunk.name == "Set Tempo":
            chunk_str = chunk_str + str(chunk.tick) + "_" + "st" + "_" + str(int(chunk.bpm)) + "_" + str(
                int(chunk.mpqn))
            chunk_str_list.append(chunk_str)

        elif chunk.name == "Control Change":
            chunk_str = chunk_str + str(chunk.tick) + "_" + "cc" + "_" + str(chunk.channel) + "_" + str(
                chunk.data[0]) + "_" + str(chunk.data[1])
            chunk_str_list.append(chunk_str)

    return ' '.join(chunk_str_list)


def list_to_midi(chunk_str_list, generated_dir, filename):
    pattern = midi.Pattern(resolution=480)

    track = midi.Track()
    pattern.append(track)

    for chunk in chunk_str_list:
        chunk_info = chunk.split("_")
        event_type = chunk_info[1]

        if event_type == "no":
            tick = int(chunk_info[0])
            pitch = int(chunk_info[2])
            velocity = int(chunk_info[3])

            e = midi.NoteOnEvent(tick=tick, channel=0, velocity=velocity, pitch=pitch)
            track.append(e)

        elif event_type == "st":
            tick = int(chunk_info[0])
            bpm = int(chunk_info[2])
            mpqn = int(chunk_info[3])
            ev = midi.SetTempoEvent(tick=tick, bpm=bpm, mpqn=mpqn)
            track.append(ev)

        elif event_type == "cc":
            control = int(chunk_info[3])
            value = int(chunk_info[4])
            e = midi.ControlChangeEvent(channel=0, control=control, value=value)
            track.append(e)

    end_event = midi.EndOfTrackEvent(tick=1)
    track.append(end_event)

    midi.write_midifile(generated_dir + filename + '.mid', pattern)


def sample(y_pred):
    preds = []
    for pred in y_pred:
        temperatures = [0.1,  0.3, 1.0, 3.0]
        weight = [0.2, 0.3, 0.3, 0.2]
        temp = np.random.choice(a = temperatures, p = weight)
        pred = np.asarray(pred).astype('float64')
        pred = pred / temp
        exp_pred = np.exp(pred)
        pred = exp_pred / np.sum(exp_pred)
        probas = np.random.multinomial(n = 1, pvals = pred, size = 1)
        preds.append(np.argmax(probas))

    return np.array(preds)
