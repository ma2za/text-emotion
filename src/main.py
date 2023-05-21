# from src.emotion_classification.multilang_emotion import emotion
#
# result = emotion(["wow, cosa stai facendo?", "è stato divertente", "i like you"],
#                  emotion_language="it")
#
# print(result)

from data.preprocessing import load_emotion_datasets

dataset = load_emotion_datasets(["emotion", "go_emotions"],
                                ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise'])

print()
