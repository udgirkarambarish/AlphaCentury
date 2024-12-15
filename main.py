from AudioAnalysis.speech import ProcessAudio
from GeminiModel.test import main
from EyeGazing.eye import eyeGazing
from EEGSignals.eeg import EEG
import pyaudio
import wave
import keyboard
import threading

from collections import deque 

# Suppress sklearn warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)




zeros=0
one=0
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

    print("Recording... Press 'q' to stop.")
    frames = []

    try:
        while True:
            if keyboard.is_pressed('q'):
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

#--------------------------------------------------------------------------------


#audio-----------------------------q---------------------------------------------------------------------------


# print("\n")
# print(m)
    # eeg()



def run_model1():
    eyeGazing()
    #print("Model 1 running")

def run_model2():
    record_audio()
    #print("Model 2 running")

def run_model3():
    # Code for model 3
    zeros,one=EEG()
    #print("Model 3 running")

# Create threads
thread1 = threading.Thread(target=run_model1)
thread2 = threading.Thread(target=run_model2)
thread3 = threading.Thread(target=run_model3)

# Start threads
thread1.start()
thread2.start()
thread3.start()

# Wait for all threads to complete
thread1.join()
thread2.join()
thread3.join()

r,m=ProcessAudio("output.wav")

if int(zeros)>int(one):
        print("Concentration")
        emot="Concentration"

else:
        print("Relaxation") 
        emot="Relaxation"  

print("Emotion:",r)
main(r,m,emot)

print("All models finished execution.")
