import speech_recognition as sr

class AudioToTextConverter:
    def __init__(self):
        # Initialize the recognizer
        self.recognizer = sr.Recognizer()
        
    def convert(self, audio_file_path: str) -> str:
        # Load audio file
        with sr.AudioFile(audio_file_path) as source:
            audio_data = self.recognizer.record(source)
        
        try:
            # Use CMU Sphinx (offline)
            text = self.recognizer.recognize_sphinx(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Recognition error; {e}"