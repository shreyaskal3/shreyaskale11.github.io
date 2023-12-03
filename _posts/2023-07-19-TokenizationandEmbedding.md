---
title: Tokenization and Embedding
date: 2023-07-19 00:00:00 +0800
categories: [NLP, Basics]
tags: [GPT]
---

## Tokenization

 There are many ways of tokenization of words, most common method is to use vocabulary of words and map each words to a unique number.This is called vocabulary mapping. 

For eg: 

    sentence = "I love to eat apples" 
    vocabulary = ["I", "love", "to", "eat", "apples"] 
    vocabulary_mapping = {"I": 0, "love": 1, "to": 2, "eat": 3, "apples": 4}

    Now we can map each word in the sentence to a unique number using the vocabulary mapping.

    sentence_tokenized = [0, 1, 2, 3, 4]

But the tokenized sentence is not useful for the model, as it does not have any information and relation between the words. So we need to convert the tokenized sentence to embedding.


## Embedding

 After tokenization, we need to convert the tokenized sentence to embedding.

 For eg: 
 
    sentence_tokenized = [0, 1, 2, 3, 4] 
    word_embedding = {"I": [0.1, 0.2, 0.3, 0.4], "love": [0.5, 0.6, 0.7, 0.8], "to": [0.9, 0.10, 0.11, 0.12], "eat": [0.13, 0.14, 0.15, 0.16], "apples": [0.17, 0.18, 0.19, 0.20]}

    Now we can map each word in the sentence to a unique embedding using the embedding.

    sentence_embedding = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.10, 0.11, 0.12], [0.13, 0.14, 0.15, 0.16], [0.17, 0.18, 0.19, 0.20]]

- How to get the embedding of a word?

    There are few ways to get the embedding of a word.

    - Use pre-trained word embedding model like word2vec, GloVe, or BERT.These models have been trained on large corpora to generate vector representations for words. You can download the models and use them in your code. For example:
        ```python
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format('/path/to/model')  
        vector = model['word']
        ```
    - Train your own word embedding model on your own text corpus using techniques like word2vec or GloVe. Then get vectors for words from your trained model.

- How model learns the embedding of a word?
           
    Models learn word embeddings using different techniques, but generally through an unsupervised learning process over large text corpora:

    - Word2Vec: Word2vec uses techniques like skip-gram or continuous bag of words to predict context words around target words. As it trains on the corpus to improve these predictions, it adjusts internal vector representations for each word. These vectors effectively embed semantic information about each word based on its usage contexts.

    - GloVe: GloVe creates word vectors by essentially doing matrix factorization on a co-occurrence count matrix for all words in a corpus. Words that appear in similar contexts have similar co-occurrence counts, which leads to similar embeddings.

    - BERT: BERT is trained on masked language modeling and next sentence prediction tasks on large text. As it trains on these tasks, the internal representations that it learns for each word contain contextual semantic information, which can be used as dynamic word embeddings.

    - ELMo: ELMo uses internal BiLSTM layers trained on a language modeling objective to capture contextual information, including syntax and semantics. The internal representations capture this useful embedding information about words in context.

    The key difference across models is the exact training techniques. But they all rely on the distributional hypothesis - words appearing in similar contexts have similar meanings. The models learn vector representations capturing these semantic relationships between words across their large training corpora.




