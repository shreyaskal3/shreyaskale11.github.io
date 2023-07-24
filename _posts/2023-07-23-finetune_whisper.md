---
title: fine-tuning whisper
date: 2023-07-23 00:00:00 +0800
categories: [Blogging, Tutorial]
tags: [favicon]
---

Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers

Wav2Vec 2.0, which is pre-trained on the unsupervised task of masked prediction. Here, the model is trained to learn an intermediate mapping from speech to hidden states from un-labelled audio only data. 
While unsupervised pre-training yields high-quality representations of speech, it does not learn a speech-to-text mapping. This mapping is only learned during fine-tuning, thus requiring more fine-tuning to yield competitive performance.

--- 

IMPORT 


LOAD DATASET



Prepare Feature Extractor, Tokenizer and Data


Load WhisperFeatureExtractor


Speech is represented by a 1-dimensional array that varies with time.The value of the array at any given time step is the signal's amplitude at that point.
From the amplitude information alone, we can reconstruct the frequency spectrum of the audio and recover all acoustic features.

Since speech is continuous, it contains an infinite number of amplitude values. This poses problems for computer devices which expect finite arrays. Thus, we discretise our speech signal by sampling values from our signal at fixed time steps. The interval with which we sample our audio is known as the sampling rate and is usually measured in samples/sec or Hertz (Hz). Sampling with a higher sampling rate results in a better approximation of the continuous speech signal, but also requires storing more values per second.

The Mel channels (frequency bins) are standard in speech processing and chosen to approximate the human auditory range. All we need to know for Whisper fine-tuning is that the spectrogram is a visual representation of the frequencies in the speech signal. For more detail on Mel channels, refer to Mel-frequency cepstrum.

 Transformers Whisper feature extractor performs both the padding and spectrogram conversion in just one line of code! Let's go ahead and load the feature extractor from the pre-trained checkpoint to have ready for our audio data:
```python
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

```

Load WhisperTokenizer

Now let's look at how to load a Whisper tokenizer. The Whisper model outputs text tokens that indicate the index of the predicted text among the dictionary of vocabulary items. The tokenizer maps a sequence of text tokens to the actual text string (e.g. [1169, 3797, 3332] -> "the cat sat").

Traditionally, when using encoder-only models for ASR, we decode using Connectionist Temporal Classification (CTC). Here we are required to train a CTC tokenizer for each dataset we use. One of the advantages of using an encoder-decoder architecture is that we can directly leverage the tokenizer from the pre-trained model.

Combine To Create A WhisperProcessor



Training and Evaluation




https://huggingface.co/blog/fine-tune-whisper 

https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb#scrollTo=c7b07f9b-ae0e-4f89-98f0-0c50d432eab6 
---