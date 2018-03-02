config = dict(

    midi_dir = 'midi-data/',
    txt_dir = 'midi-txt/',
    generated_dir = 'generated-midi/',
    model_checkpoint_dir = 'models/lstm-midi-lm-',
    logs_dir = 'logs/lstm-midi-lm',

    prev_n_tokens = 20,
    min_token_freq = 2, # Minimum number of occurences for a token to be retained
    num_files = 10, # Number of MIDI files from the dataset to be used for training

    n_epochs = 5,
    learning_rate = 0.01,
    batch_size = 128,
    rnn_size = 512,
    num_layers = 1,
    dropout_keep = 0.6,
    n_steps = 400, # Number of time steps of future prediction (during test time - music generation)
    vocab_size = 10000,
    load_checkpoint = 5,

)