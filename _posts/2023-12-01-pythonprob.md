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


### HCF and LCM
```python
def hcf(x, y):
    """This function implements the Euclidian algorithm
    to find H.C.F. of two numbers"""
    while(y):
        x, y = y, x % y
    return x
```
```python
def find_hcf(x, y, z):
    # Function to find the HCF of two numbers using Euclid's Algorithm
    def find_gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    # Finding the HCF of three numbers
    hcf_xyz = find_gcd(x, find_gcd(y, z))
    return hcf_xyz
```

```python
def lcm(x, y):
    """This function takes two
    integers and returns the L.C.M."""
    lcm = (x*y)//hcf(x,y)
    return lcm
```

### To find the factors of a number
```python
def print_factors(x):
    """This function takes a
    number and prints the factors"""
    print("The factors of",x,"are:")
    for i in range(1, x + 1):
        if x % i == 0:
            print(i)
```




## Medium


### HCF

Sagar, a mathematics and computer science student, needs to find the largest possible HCF (Highest Common Factor) that can be obtained using three distinct numbers less than or equal to a given number N. Help him solve this problem.

Constraints
- 1 <= T <= 10
- 1 <= N <= 10^18

Input
- The first line contains an integer T, denoting the number of test cases.
- For each test case, there is a line containing an integer N.

Output
- Output the answer for each test case on the next line.
- If multiple answers are possible, print the smallest three numbers in increasing order.
- If no solution is possible, print -1.

| Sample Input | Sample Output |
| --- | --- |
| 2| -1 |
2 | 1 2 3 |
5

```
Explanation :
n = 6 , hcf(2,4,6) = 2
n = 9 , hcf(3,6,9) = 3
n = 12 , hcf(4,8,12) = 4
```
```python
n = int(input())                # number of test cases

for i in range(n):
    inp = int(input())         # input for each test case
    if (inp==0)|(inp-1 ==0)|(inp-2==0):
        print(-1)
        continue
    print(inp//3,(inp//3)*2,(inp//3)*3)
```



### Intersection of 3 Sorted Arrays

You are given three sorted arrays `arr1`, `arr2`, and `arr3`. The task is to return a sorted array of elements that are common to all three arrays.

Input:

- `n1`, `n2`, `n3`: The lengths of `arr1`, `arr2`, and `arr3` respectively.
- `arr1`, `arr2`, `arr3`: Three sorted arrays.

Output:

A sorted array of numbers common to `arr1`, `arr2`, and `arr3`.

Constraints:

- 1 <= `arr1.length`, `arr2.length`, `arr3.length` <= 1000
- 1 <= `arr1[i]`, `arr2[i]`, `arr3[i]` <= 2000
- There will be at least one element common in all three arrays.

Example:

**Input:**
```
5
5
5
1 2 3 4 5
1 3 4 5 8
1 2 5 7 9
```

**Output:**
```
1 5
```
```python
from collections import Counter
def array_intersection (n1, n2, n3, arr1, arr2, arr3):
    # Write your code here
    # print(set(arr1)&set(arr2)&set(arr3))
    counter1 = Counter(arr1)
    counter2 = Counter(arr2)
    counter3 = Counter(arr3)

    intersection = counter1 & counter2 & counter3

    result = []
    for k,i in intersection.items():
        result.extend([k]*i)
    return sorted(result)

n1 = int(input())
n2 = int(input())
n3 = int(input())
arr1 = list(map(int, input().split()))
arr2 = list(map(int, input().split()))
arr3 = list(map(int, input().split()))

out_ = array_intersection(n1, n2, n3, arr1, arr2, arr3)
print (' '.join(map(str, out_)))
```
---

### Missingno!
You are given an integer n, along with an array of n-1 numbers in the range 1 to n, with no duplicates. One number is missing from the array. Find that number.

Input:

n - An integer

arr - An array of size n-1

Output:

The single missing number

Constraints:

1 <= n <= 1000

1 <= arr[i] <= n

All elements in arr are unique

**Input:**
```
4
4 1 2
```

**Output:**
```
3
```
```
Explanation:
sum of n numbers = n*(n+1)/2
missing number = sum of n numbers - sum of array elements
```
```python
def find_missing_number(n, arr):
    # Write your code here
    return (n*(n+1)//2) - sum(arr)
```


### Stuff Them Candies In!
You are given an array candies, where candies[i] defines how many candies the i-th kid has. You are also given an integer, extra_candies, which can be distributed among the kids.

For each kid, check if there is a way to distribute extra_candies such that that kid has the maximum number of candies. Multiple kids can have maximum candies.

Input:

n - The number of elements in the candies array

candies -The array itself

extra_candies - An integer containing the number of extra candies

Output:

An array of 1s and 0s, where 1 denotes that the kid can have the maximum number of candies, and 0 denotes that the kid cannot.

Constraints:

2 <= candies.length <= 100

1 <= candies[i] <= 100

1 <= extra_candies <= 50
```
SAMPLE INPUT 
5
2 3 5 1 3
3
SAMPLE OUTPUT 
1 1 1 0 1
```

```python
def extra_candy (n, candies, extra_candies):
    # Write your code here
    diff = max(candies)-extra_candies
    return [1 if i >= diff else 0 for i in candies ]
    
n = int(input())
candies = list(map(int, input().split()))
extra_candies = int(input())

out_ = extra_candy(n, candies, extra_candies)
print (' '.join(map(str, out_)))
``` 






### Shuffle the Array!
You are given an array nums consisting if 2n elements in the form [x1,x2,x3...xn,y1,y2,y3...yn].

Return an array in the form [x1,y1,x2,y2,x3,y3...,xn,yn].

Input:

n -  Half the size of the array arr

arr - The array itself

Output:

An array of the form [x1,y1,x2,y2,x3,y3...,xn,yn]

Constraints:

1 <= n <= 500

nums.length = 2n

1 <= nums[i] <= 1000
```
SAMPLE INPUT 
3
2 5 1 3 4 7
SAMPLE OUTPUT 
2 3 5 4 1 7
```

```python
def shuffle (n, arr):
    # Write your code here
    arr1 = arr[:n]
    arr2 = arr[n:]
    out = []
    [out.extend([arr1[i],arr2[i]]) for i in range(n) ]
    return out
    
n = int(input())
arr = list(map(int, input().split()))

out_ = shuffle(n, arr)
print (' '.join(map(str, out_)))
```

### Destroy Those Pairs!

You are given a string str of lower case letters. If this string has any adjacent pairs of the same characters, that pair must be removed. The new string must again be checked for adjacent pairs. Repeat these steps until no pairs exist.

Input:

str - the string

Output:

A string without adjacent pairs

Constraints:

1 <= str.length <= 10000

str consists of only lower case alphabets
```
SAMPLE INPUT 
abcddce
SAMPLE OUTPUT 
abe
```
```
Explanation
First, the 2 'd's are deleted from the string "abcddce", to make it "abcce"

Then, the two adjacent 'c's are removed, to make it "abe"

Now, there are no longer any adjacent pairs. Hence, the result is "abe"
```


```python
def remove_pair (str):
    # Write your code here

    stack = []
    i = 0
    
    while i <len(str):
        if (len(stack)!=0) and (str[i] == stack[-1]):
            stack.pop()
        else:
            stack.append(str[i])
        i +=1

    return "".join(stack)
str = input()

out_ = remove_pair(str)
print (out_)
```

### Good Pairing
You are given an array arr in which a good pair is defined as a pair of numbers in the array which satisfy the following conditions:

arr[i] = arr[j] (The two numbers must be equal)

i<j

Find the number of good pairs in the array.

Input:

n - The number of elements in arr

arr - The array itself.

Output:

The number of good pairs.

Conditions:

1 <= arr.length <= 100

1 <= arr[i] <= 100
```
SAMPLE INPUT 
6
1 2 3 1 1 3
SAMPLE OUTPUT 
4
```
```python
from collections import Counter
def good_pairs (n, arr):
    # Write your code here
    """
    11 - 1
    111 - 2+1
    1111 - 3+2+1
    """
    s = Counter(arr)
    out = 0
    for k,i in s.items():
        out+= i*(i-1)//2
    return out

n = int(input())
arr = list(map(int, input().split()))

out_ = good_pairs(n, arr)
print (out_)
```

### Golden Letters
You are given a string key that contains a list of golden letters. You are also given another string str. Find out how many characters in str are golden letters.

Input:

key - a string of golden letters

str - the string to be checked

Output:

The number of golden letters in str

Constraints:

1 <= key.length <= 52

1 <= str.length <= 10000

key and str are made up of only upper-case and lower-case alphabets
```
SAMPLE INPUT 
wxYZ
lmnoWwwxyZ
SAMPLE OUTPUT 
4
```
```python
from collections import Counter
def golden_char (key, str):
    # Write your code here
    c = set(key)&set(str)
    s = Counter(str)
    out = 0
    for k in c:
        out+=s[k]
    return out

key = input()
str = input()

out_ = golden_char(key, str)
print (out_)
```


### Longest String Without Repeating Characters
You are given a string str. Find the length of the longest possible substring in str without ANY repeating characters.

Input:

str - A string

Output:

The length of the longest possible substring in str without repeating characters

Constraits:

1 <= str.length <= 10000
```
SAMPLE INPUT 
abcbcde
SAMPLE OUTPUT 
4
```
```
Explanation
"bcde" is the longest string in the input without any repeating characters, ie, each of the characters appears only once. The length of this string is 4, hence that is the output.
```
```python

def no_dups (str):
    # Write your code here
    # Use len(str) to generate the length of str
    n = len(str)
    m = 0
    marr = []
    for i in range(n):
        j=i+1
        arr = []
        arr.append(str[i])
        while (j<n):
            if str[j] in arr:
                break
            else:
                arr.append(str[j])
                j+=1
        if len(arr)>m:
            m = len(arr)
            marr = arr
    # print(marr)   
    return m

str = input()

out_ = no_dups(str)
print (out_)
```
```python
def no_dups(s):
    n = len(s)
    max_length = 0
    current_length = 0
    char_index = {}

    for i in range(n):
        if s[i] in char_index and char_index[s[i]] >= i - current_length:
            current_length = i - char_index[s[i]]
        else:
            current_length+=1
        char_index[s[i]]=i
        max_length = max(max_length,current_length) 
    return max_length

str_input = input()
output = no_dups(str_input)
print(output)
```


## Hard

### Verify the Alien Dictionary

You are given a string key that contains the 26 English alphabets, jumbled in some order.

You are also given n words stored in an array str.

Check if the words in str are sorted lexicographically according to string key.

Input:

key - A string that denotes the correct order

n - The number of words in str

str - An array of words that must follow the key string lexicograhically

Output:

1 if all the words in str are lexicographically sorted according to key, 0 if they are not

Constraints:

key.length = 26

2 <= n <= 100

1 <= str[i].length <= 100 (Length of each word)

key and str consist of only uppercase alphabets

key has all 26 uppercase alphabets, jumbled in some order

```
SAMPLE INPUT 
HECABDFGIJKLMNOPQRSTUVWXYZ
3
HACKER
EARTHS

SAMPLE OUTPUT 
1
```
```
Explanation
In the first sample above, the key is "HECABDFGIJKLMNOPQRSTUVWXYZ". The first characters of the words "HACKER", "EARTHS" and "CODEXPLAINED" are 'H', 'E' and 'C'. These are sorted correctly according to key, hence the output is 1.

In the second sample below, the two words are "AGENTS" and "AGENCY". The first 4 characters are equal. The 5th characters are 'N' and 'C'. However, if we look at the key, 'N' comes AFTER 'C'. Hence, the first string is lexicographically GREATER than the second, and so the output is 0.

In the third sample below, the two words are "XAVIER" and "XAVIE". Their first 5 characters are equal. At this point, the second string is terminated. However, the length of the first string is greater than that of the second. The first string is lexicographically GREATER than the second, and so the output is 0.
```

```python 
    def is_lexicographic (key, n, str):
    # Write your code here
    d = {k:i for i,k in enumerate(key)}

    for i in range(n-1):
        m = 0
        j = 0
        while j!=len(str[i]) and j!=len(str[i+1]) and str[i][j]==str[i+1][j]:
            j+=1
        if j==len(str[i]):
            continue
        if j==len(str[i+1]):
            return 0
        # print(d[str[i][j]] , d[str[i+1][j]])
        if d[str[i][j]] > d[str[i+1][j]]:
            return 0
    return 1

key = input()
n = int(input())
str = []
for _ in range(n):
    str.append(input())

out_ = is_lexicographic(key, n, str)
print (out_)
    
```


### Three Sum

You are given an array arr consisting of unique numbers. Find the total number of triplets in the array such that:

arr[i] + arr[j] + arr[k] = 0,

where i != j != k

In other words, identify how many sets of three different elements in the array add up to zero.

Input:

n - the size of the array

arr - the array itself

Output:

A number denoting how many triplets add up to 0.

Constaints:

1 <= n <= 100

-10000 <= arr[i] <= 10000
```
SAMPLE INPUT 
6
0 8 6 2 -2 -14
SAMPLE OUTPUT 
2
```
```
Explanation
The two sets are (6,8,-14) and (-2,0,2).
```
```python
def no_of_triplets (n, arr):
    # Write your code here
    arr = sorted(arr)
    out = 0
    for i in range(n - 3):
        l = i+1
        r = n-1
        while (l!=r):
            # print(arr[l],arr[r])
            if arr[l]+arr[r]+arr[i]==0:
                out+=1
                # print(arr[l],arr[r],arr[i])
            if r == l+1:
                r = n
                l += 1
            r -= 1
    return out

n = int(input())
arr = list(map(int, input().split()))

out_ = no_of_triplets(n, arr)
print (out_)

```


### Letter Combinations of a Phone Number
You are given a string str containing only digits from 2 to 9 (including 2 and 9). You are also given the following mapping of digits, such as they are on a telephone. Return all possible letter combinations that the number could represent. The resultant array must be in ascending order.

Input:

str - A string 

Output:

A string array containing all possible combinations, sorted in ascending order

Constraints:

1 <= str.length <= 10

str consists of only digits from 2 to 9

```
SAMPLE INPUT 
68
SAMPLE OUTPUT 
mt
mu
mv
nt
nu
nv
ot
ou
ov
```

```python
def letter_combinations(digits):
    d = {'2':'abc', '3':'def', '4':'ghi', '5':'jkl', '6':'mno', '7':'pqrs', '8':'tuv', '9':'wxyz'}
    
    def backtrack(index, current):
        if index == len(digits):
            combinations.append(current)
            return

        for letter in d[digits[index]]:
            backtrack(index + 1, current + letter)
    combinations = []
    if digits:
        backtrack(0, '')
    return sorted(combinations)

str = input()
output = letter_combinations(str)
for i in output:
    print(i)
```


