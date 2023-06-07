from src.text_emotion import Detector

detector = Detector(emotion_language="fr")

print(detector.detect(["Hello, I am so happy!", "sono felice!"] * 7))
