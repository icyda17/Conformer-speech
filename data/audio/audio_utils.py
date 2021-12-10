import librosa

def load_audio(audio_path, sample_rate):
    assert audio_path.endswith('wav') or audio_path.endswith(
        'flac'), "only wav/flac files"
    signal, sr = librosa.load(audio_path, sr=sample_rate)

    return signal