---
title: Regular Expression
date: 2023-09-03 00:00:00 +0800
categories: [NLP, Regular Expression]
tags: [Regular Expression]
---

# Python Regular Expression Cheat Sheet

## Basics

- `.`: Matches any character except a newline.
- `^`: Anchors the regex at the start of the string.
- `$`: Anchors the regex at the end of the string.
- `[]`: Matches any one of the enclosed characters.
- `|`: Acts like a logical OR.

## Character Classes

- `\d`: Matches any digit (0-9).
- `\D`: Matches any non-digit.
- `\w`: Matches any alphanumeric character (word character).
- `\W`: Matches any non-alphanumeric character.
- `\s`: Matches any whitespace character.
- `\S`: Matches any non-whitespace character.

## Quantifiers

- `*`: Matches 0 or more occurrences.
- `+`: Matches 1 or more occurrences.
- `?`: Matches 0 or 1 occurrence.
- `{n}`: Matches exactly n occurrences.
- `{n,}`: Matches n or more occurrences.
- `{n,m}`: Matches between n and m occurrences.

## Anchors

- `\b`: Matches a word boundary.
- `\B`: Matches a non-word boundary.
- `^`: Matches the start of a string.
- `$`: Matches the end of a string.

## Groups and Capturing

- `()`: Groups patterns together.
- `(?:...)`: Non-capturing group.

## Character Escapes

- `\\`: Escapes a special character.
- `\n`, `\t`, etc.: Newline, tab, etc.

## Examples

- `\d{3}-\d{2}-\d{4}`: Matches a Social Security Number.
- `^\w+@\w+\.\w+$`: Matches a basic email address.

## Flags

- `re.IGNORECASE` or `re.I`: Case-insensitive matching.
- `re.MULTILINE` or `re.M`: ^ and $ match the start/end of each line.

## Methods

- `re.search(pattern, string)`: Searches for the first occurrence of the pattern.
- `re.match(pattern, string)`: Matches the pattern only at the beginning of the string.
- `re.fullmatch(pattern, string)`: Matches the entire string against the pattern.
- `re.findall(pattern, string)`: Returns a list of all occurrences of the pattern.
- `re.finditer(pattern, string)`: Returns an iterator of match objects for all occurrences.
- `re.sub(pattern, replacement, string)`: Replaces occurrences of the pattern with the replacement.

```python
import re

# Example 1: Check if a string contains a number
result = bool(re.search(r'\d', 'Hello123World'))
print(result)  # Output: True

# Example 2: Extract all words from a string
words = re.findall(r'\b\w+\b', 'This is a sample sentence.')
print(words)  # Output: ['This', 'is', 'a', 'sample', 'sentence']

# Example 3: Replace all vowels with '*'
new_string = re.sub(r'[aeiou]', '*', 'Hello World')
print(new_string)  # Output: H*ll* W*rld

# Example 4: Extract domain from an email address
domain = re.search(r'@\w+\.\w+', 'user@example.com').group()
print(domain)  # Output: @example.com
```

### Split a paragraph into sentences

```python
input = input()
split = input.split('.')
for i in split:
    print(f"{(i).strip()}.")
```

## Perplexity Calculation

Perplexity is a measure of how well a probability distribution or probability model predicts a sample. It is often used in the context of language modeling.

The formula for perplexity in the case of a unigram model is:

$$ \text{Perplexity} = 2^{H(p)} $$

$$ H(p) = - \sum\_{x \in \mathcal{X}} p(x) \log_2 p(x) $$

where $ H(p) $ is the cross-entropy of the unigram model.
