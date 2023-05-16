from webvtt import WebVTT, Caption
from tkinter import filedialog
import tkinter as tk
import threading
import queue
import whisper
import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
import shutil


# A simple app that generate transcript out of audio.
class AudioCaptureApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Transcript Generator")

        # Button that captures audio.
        self.capture_button = tk.Button(
            self.root,
            text="Capture Audio",
            command=self.toggle_audio_capture
        )
        self.capture_button.pack()

        # Button that export the recorded audio and transcript.
        self.transcript_button = tk.Button(
            self.root,
            text="Export Audio and Transcript",
            state=tk.DISABLED,  # Initially disabled
            command=self.export_transcript
        )
        self.transcript_button.pack()

        self.is_capturing = False
        self.audio_queue = queue.Queue()

        # Load the Whisper model.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.whisper = whisper.load_model("small", device=device)

    # Start/stop capturing audio.
    def toggle_audio_capture(self):
        if not self.is_capturing:
            self.is_capturing = True
            self.capture_button.config(text="Stop Capturing")
            self.start_audio_capture()
        else:
            self.is_capturing = False
            self.capture_button.config(text="Capture Audio")

    # Start a new thread for capturing audio.
    def start_audio_capture(self):
        capture_thread = threading.Thread(target=self.capture_audio)
        capture_thread.start()

    def capture_audio(self):
        while self.is_capturing:
            # Capture audio.
            audio_data = capture_audio()
            # Put audio data in the queue
            self.audio_queue.put(audio_data)

        # Enable the "export" button when there is audio to process
        if self.audio_queue.qsize() > 0:
            self.transcript_button.config(state=tk.NORMAL)
        else:
            self.transcript_button.config(state=tk.DISABLED)

    def export_transcript(self):
        # Get all the audio data from the queue.
        self.transcript_button.config(state=tk.DISABLED)
        audio_data = self.audio_queue.get()
        while not self.audio_queue.empty():
            audio_data = np.concatenate((audio_data, self.audio_queue.get()))

        # Create a temporary WAV file for Whisper.
        sf.write("gen-trans-temp.wav", audio_data, 44100, 'PCM_24')

        # Generate transcript out of the temporary WAV file using Whisper
        transcript = self.whisper.transcribe("gen-trans-temp.wav")

        save_transcript(transcript)

    def run(self):
        self.root.mainloop()


# Record audio every 5 seconds
def capture_audio():
    duration = 5  # Duration of audio capture in seconds
    srate = 44100  # Sample rate
    channels = 2  # Channels

    # Start capturing audio.
    audio_data = sd.rec(int(duration * srate),
                        samplerate=srate,
                        blocking=True,
                        channels=channels)

    return audio_data


# Format time recorded by Whisper.
def format_time(seconds):
    s = seconds % 60
    m = seconds // 60
    h = m // 60
    m %= 60
    return "{:02d}:{:02d}:{:02d}.000".format(h, m, s)


# Save audio and transcript as a WAV and a VTT file.
def save_transcript(transcript):
    vtt = WebVTT()
    for seg in transcript["segments"]:
        start_time = format_time(int(seg['start']))
        end_time = format_time(int(seg['end']))
        caption = Caption(
            start_time,
            end_time,
            [seg['text']]
        )
        vtt.captions.append(caption)

    root = tk.Tk()
    root.withdraw()

    audio_path = filedialog.asksaveasfilename(
        initialdir="./",  # Set initial directory if needed
        title="Save Audio",
        defaultextension="wav",
        filetypes=(("WAV files", "*.wav"),)
    )
    print("audio path:", audio_path)

    shutil.copyfile("gen-trans-temp.wav", audio_path)

    # Prompt the user to choose the output location and file name
    file_path = filedialog.asksaveasfilename(
        initialdir="./",  # Set initial directory if needed
        title="Save Transcript",
        defaultextension="vtt",
        filetypes=(("VTT files", "*.vtt"),)
    )

    # Check if the user canceled the file dialog
    if not file_path:
        print("Save canceled.")
        return

    try:
        vtt.save(file_path)
        print("Transcript saved successfully.")
    except Exception as e:
        print("Error saving transcript:", str(e))


if __name__ == "__main__":
    app = AudioCaptureApp()
    app.run()
