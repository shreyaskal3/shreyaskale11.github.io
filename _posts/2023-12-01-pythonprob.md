---
title: Python Problem
date: 2023-12-01 00:00:00 +0800
categories: [pythonprob]
tags: [pythonprob]
---




## Basic

### sort a list of elements using the bubble sort algorithm.
```python
nums = [2,0,1,2,0,1]

for i in range(1,len(nums)):
    for j in range(i):
        if nums[i]<nums[j]:
            nums[j],nums[i] = nums[i],nums[j]

print(nums)
```

### To print all possible permutations of a given string. 
```python
def generate_permutations(s, start=0):
    if start == len(s) - 1:
        print(''.join(s))
        return

    for i in range(start, len(s)):
        # Swap characters at indices start and i
        s[start], s[i] = s[i], s[start]

        # Recursively generate permutations for the remaining part of the string
        generate_permutations(s, start + 1)

        # Backtrack: undo the swap for the next iteration
        s[start], s[i] = s[i], s[start]

# Example usage
input_string = "loop"
input_list = list(input_string)
generate_permutations(input_list)
```

### To get the factorial of a non-negative integer.
```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

result = factorial(5)
print("Factorial:", result)
```


### To get the Fibonacci series between 0 to 50.
```python
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
nterms = 10
if nterms <= 0:
    print("Please enter a positive integer")
else:
    print("Fibonacci sequence:")
    for i in range(nterms):
        print(fibonacci(i))
```

### To find duplicates in a string
```python
input_str = "hello world"
seen_chars = set()
duplicates = []

for char in input_str:
    if char in seen_chars:
        duplicates.append(char)
    else:
        seen_chars.add(char)

print("Duplicates:", duplicates)
```


## Advanced


