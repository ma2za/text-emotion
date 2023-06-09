# Text Emotion

# Introduction

### Supported Languages

The following languages are supported by the finetuned
xlm-roberta model:

- English
- French
- Spanish
- German
- Italian

All other languages are translated to English
using the EasyNMT library. If the language is not
supported by EasyNMT, then it is not supported.

# Installation

You can install emotion using:

    $ pip install text-emotion

# Usage

```python
from text_emotion import Detector

detector = Detector(emotion_language="fr")

print(detector.detect("Hello, I am so happy!"))
```

### XLM-Roberta

The underlying model is xlm-roberta-large. You can test it at:

https://huggingface.co/ma2za/xlm-roberta-emotion

### References

[Unsupervised Cross-lingual Representation Learning at Scale](https://aclanthology.org/2020.acl-main.747) (Conneau et
al., ACL 2020)