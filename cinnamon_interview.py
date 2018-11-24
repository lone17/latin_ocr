from nltk.corpus import words
import re
words = words.words()
s = 'Ilikesandwich'
all = []

def foo(str, cur):
    # print(cur)
    n = len(str)
    if n == 0:
        all.append(cur)
    for i in range(1, n+1):
        if str[:i] in words:
            foo(str[i:], cur + [str[:i]])

max_len = max([len(w) for w in words])
scores = [0]
for i in range(1, max_len+1):
    scores.append(len([w for w in words if len(w) == i]) / len(words))

def get_score(sentence):
    score = 0
    for w in sentence:
        score += scores[len(w)]

    return score

foo(s, [])
top5 = sorted(all, key=lambda sen: get_score(sen))[-5:]
