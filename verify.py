from AudioAnalysis.speech import ProcessAudio
from GeminiModel.test import main
from EEGSignals.eeg import eeg
import pyaudio
import wave
import keyboard
import serial
import numpy as np
from scipy import signal
import pandas as pd
import time
import pickle
import pyautogui

from collections import deque 

# Suppress sklearn warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#audio--------------------------------------------------------------------------------

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
OUTPUT_FILE = "output.wav"

def record_audio():
    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording... Press 'v' to stop.")
    frames = []

    try:
        while True:
            if keyboard.is_pressed('v'):
                print("Stopping recording...")
                break
            data = stream.read(CHUNK)
            frames.append(data)
    except Exception as e:
        print(f"An error occurred: {e}")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recording to a WAV file
    with wave.open(OUTPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Recording saved to {OUTPUT_FILE}")

#eeg--------------------------------------------------------------------------------

    


if __name__ == "__main__":
    #record_audio()
    # m=ProcessAudio("output.wav")
    # print(r)
    # print("\n")
    # print(m)
    eeg()
    # print(main(m))

    

