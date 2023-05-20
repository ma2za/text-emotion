# from src.emotion_classification.multilang_emotion import emotion
#
# result = emotion(["wow, cosa stai facendo?", "è stato divertente", "i like you"],
#                  emotion_language="it")
#
# print(result)
from transformers import RobertaTokenizer

from data.preprocessing import load_emotion_datasets

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

dataset = load_emotion_datasets(["emotion", "daily_dialog", "go_emotions"], tokenizer)

print()
