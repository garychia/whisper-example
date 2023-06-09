# Transcript Generator

This is a simple Python application that allows you to capture audio and generate a transcript from the captured audio. The application provides a graphical user interface (GUI) built using the Tkinter library.

## Python Packages Required
Make sure to have webvtt, tkinter, whisper, torch, sounddevice, soundfile, numpy and shutil installed.

## Usage
* Execute the following command to run the app.
```
python main.py
```
* Click the "Capture Audio" button to start capturing audio. The button text will change to "Stop Capturing". The application will record audio and repeat the process until you click the "Stop Capturing" button.
* Once you have captured audio, the "Export Audio and Transcript" button will become enabled. Click the button to export the captured audio and generate a transcript.
* A file dialog will prompt you to choose the location and file name to save the audio.
* Another file dialog will prompt you to choose the location and file name to save the transcript.
* The captured audio will be saved as a WAV file, and the transcript will be saved as a VTT file.
