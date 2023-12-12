---
title: NLP Tweet Split
date: 2023-12-01 00:00:00 +0800
categories: [NLP, Tweetsplit]
tags: [nlpproblem]
---

# Tweet Split
Given a set of Twitter hashtags, split each hashtag into its constituent words. For example:

wearethepeople is split into we are the people
mentionyourfaves is split into mention your faves

```python
import re

def segment(tweet, cword):
    # Sort cword in decreasing string order
    cword.sort(key=len, reverse=True)
    return tokenize(tweet, cword, "")

def tokenize(tweet, cword, token):
    # Are we done yet?
    if not tweet:
        return [token]
    # Find all possible prefixes
    for pref in cword:
        if tweet.startswith(pref):
            res = tokenize(tweet[len(pref):], cword, pref)
            return res + [token] if res else False
    # Not possible
    return False

def main():
    num = int(input())
    tweets = [input().strip() for _ in range(num)]
    # Sample word list (you can replace it with a larger, more comprehensive list)
    word_list = ["we", "are", "the", "people", "mention", "your", "faves", "now", "playing", "the", "walking", "dead", "follow", "me"]

    for tweet in tweets:
        result = segment(tweet, word_list)
        print(' '.join(result) if result else tweet)

if __name__ == "__main__":
    main()

```