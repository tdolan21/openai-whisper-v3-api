import soundfile as sf
import librosa

def convert_to_16000hz(input_path, output_path):
    # Load the audio file with librosa
    audio, sample_rate = librosa.load(input_path, sr=None)  # sr=None to preserve the original sample rate
    if sample_rate != 16000:
        # Resample the audio to 16000 Hz
        audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    else:
        audio_16k = audio

    # Save the resampled audio
    sf.write(output_path, audio_16k, 16000)

# Example usage:
input_audio_path = '/path/to/your/file/' # Replace with your input file path
output_audio_path = 'file_16000hz.mp3'  # Replace with your desired output file path

# Convert the input file
convert_to_16000hz(input_audio_path, output_audio_path)
