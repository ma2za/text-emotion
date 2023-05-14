from multilang_emotion import emotion

result = emotion(["wow, cosa stai facendo?", "è stato divertente", "i like you"],
                 emotion_language="it")

print(result)
