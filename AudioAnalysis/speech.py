import google.generativeai as genai
import speech_recognition as sr
import pyaudio
import wave
from pynput import keyboard


genai.configure(api_key="AIzaSyAyfGYKjlBgamqssVJ-ekkMtKG67-H2Njs")
model1 = genai.GenerativeModel("gemini-1.5-flash")


def summarize(text):
    input_prompt = (
    f"Summarize the speech '{text}'. "
    )

    response = model1.generate_content(input_prompt).text.strip()
    return response

def emotion(text):
    input_prompt = (
    f"List only one emotion analyzed from text '{text}' among (happy, fear, sad, anger, neutral). Do not describe the emotion."
    )

    response = model1.generate_content(input_prompt).text.strip()
    return response




def problem(out,emot):
    input_prompt=(

        f"'{out}' is the summary of text that was spoked by person."
        f"'{emot}' are the emotion analyzed from text ."
        
        f" analyze the the above statements and  give me the short context about the situation of person  but dont include any type of punctuations,astrix, column,etc"
        
         

    )

    response = model1.generate_content(input_prompt).text.strip()
    return response




# Settings for recording
audio_filename = "output.wav"
chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16-bit audio format
channels = 1
rate = 44100  # Sample rate

# Flag to stop recording
# recording = True

# def on_press(key):
#     global recording
#     try:
#         if key.char == 'v':
#             recording = False
#             return False  # Stop listener
#     except AttributeError:
#         pass

# def record_audio():
#     global recording
#     print("Recording... Press 'v' to stop.")

#     p = pyaudio.PyAudio()
#     stream = p.open(format=sample_format,
#                     channels=channels,
#                     rate=rate,
#                     frames_per_buffer=chunk,
#                     input=True)

#     frames = []

#     # Record until 'v' is pressed
#     with keyboard.Listener(on_press=on_press) as listener:
#         while recording:
#             data = stream.read(chunk)
#             frames.append(data)
#         listener.join()

#     # Stop and close the stream
#     stream.stop_stream()
#     stream.close()
#     p.terminate()

#     # Save the recorded audio
#     with wave.open(audio_filename, 'wb') as wf:
#         wf.setnchannels(channels)
#         wf.setsampwidth(p.get_sample_size(sample_format))
#         wf.setframerate(rate)
#         wf.writeframes(b''.join(frames))

#     print(f"Audio recorded and saved as {audio_filename}")

def convert_speech_to_text(voice):
    recognizer = sr.Recognizer()
    
    # Load the recorded audio
    # Load the recorded audio
    with sr.AudioFile(voice) as source:
        #print("Converting speech to text...")
        audio_data = recognizer.record(source)
    

    try:
        # Recognize speech using Google Web Speech API
        text = recognizer.recognize_google(audio_data)
        #print("Transcription:", text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def ProcessAudio(audio):
    text= convert_speech_to_text(audio)
    out=summarize(text)
    emot=emotion(text)
    keywords=problem(out,emot)

    return emot,keywords
    
            

if __name__ == "__main__":
    # record_audio()
    ProcessAudio(audio)
    

    # text=convert_speech_to_text()

    # out=summarize(text)

    # print(out)

    # emot=emotion(text)

    # print(emot)

    # print(problem(out,emot))
