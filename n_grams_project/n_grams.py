r"""
CS 20P n-grams!

This was a project assigned to me for the CS20p class at Cabrillo College. The objective was
to define three functions: n_grams, most_frequent_n_grams, and main. n_grams returns all 
n-grams of a certain length that appear at least a certain number of times in a text. 
most_frequent_n_grams returns the top most frequent n-grams of a range of lengths based on a
limit (top 5, top 10, etc.) in a text. The main function prints the behavior of n_grams. 

For the purposes of this assignment, a word was defined as a sequence of non-whitespace
characters, with the following characters stripped from either end: !"#%&'()*,-./:;?@\_¡§¶·¿
"""
__author__ = 'Axel V. Morales Sanchez'

from collections import Counter
from collections import defaultdict
import sys


def n_grams(text: str, n_gram_len: int, min_count: int = 2) -> dict[int, list[tuple[str]]]:
  """
  Finds and returns all word n-grams of length `n_gram_len`
  occurring at least `min_count` times in `text`.

  :param text: the text to analyze
  :param n_gram_len: the desired length of n-grams (e.g. 2 for 2-grams)
  :param min_count: the minimum number of times a n-gram must appear in the text to be counted
  :return a dictionary mapping n-gram occurrence counts to a list of the n-grams occurring that
          number of times, as a list of n_gram_len-tuples of strings in ascending
          lexicographic/alphabetical order of the n-gram words.
  """
  strip = r'!\"#%&\'()*,-./:;?@\_¡§¶·¿'
  text = [word.lower().strip(strip) for i, word in enumerate(text.split())]
  text_sl = [word for i, word in enumerate(text) if len(word) != 0]
  lim = len(text_sl) - (n_gram_len - 1)
  n_grams_list = [tuple(text_sl[i: i + n_gram_len]) for i, word in enumerate(text_sl) if i < lim]
  count_list = list(sorted(Counter(n_grams_list).items()))
  count_dict = defaultdict(list)
  for n_gram, count in count_list:
    if count >= min_count:
      count_dict[count].append(n_gram)
  return dict(count_dict)


def most_frequent_n_grams(text: str,
                          min_len: int = 1,
                          max_len: int = 10,
                          limit: int = 5) -> dict[int, list[tuple[tuple[str], int]]]:
  """
  Returns a dictionary mapping n-gram lengths to a list of the most frequently occurring word
  n-grams of that length, along with their occurrence counts, for n-grams appearing at least twice.

  :param text: the text to analyze
  :param min_len: the minimum n-gram length
  :param min_len: the maximum n-gram length
  :param limit: the maximum number of n-grams to display for each length
  :return a dictionary mapping n-gram lengths to a list of the most frequently occurring n-grams
          of that length, along with their occurrence counts, as a list of 2-tuples, where
          each tuple contains a tuple of the words in an n-gram and the n-gram's occurrence count.
          The list shall be sorted in descending order of occurrence count, with ties broken in
          ascending lexicographic/alphabetical order of the n-gram words.
  """
  top_dict = {}
  for n_gram_len in range(min_len, max_len + 1):
    n_grams_dict = n_grams(text, n_gram_len)
    sorted_keys = sorted(list(n_grams_dict.keys()))[-limit:][::-1]
    mp = [n_grams_dict[key] for key in sorted_keys]
    pairs = [(n, grams[-1]) for grams in mp for i, n in enumerate(grams) if i < len(grams) - 1]
    top_dict[n_gram_len] = pairs[: limit]
  return top_dict


def main():
  """
  Expects one or two command-line arguments:
  sys.argv[1]: A length of n-gram (e.g. 2 for bigrams)
  sys.argv[2] (optional): A minimum occurrence count (2 if unspecified)
  Then prints, in descending order of occurrence count, lines containing (a) the occurrence count
  and (b) a comma-separated list of all n-grams with that occurrence count,
  in ascending alphabetical/lexicographic order.
  """
  text = sys.stdin.read()
  n_gram_len = int(sys.argv[1])
  if len(sys.argv) >= 3:
    minimum_occurence = int(sys.argv[2])
  else:
    minimum_occurence = 2
  n_grams_dict = n_grams(text, n_gram_len, minimum_occurence)
  sorted_keys = sorted(list(n_grams_dict.keys()))[::-1]
  for key in sorted_keys:
    print(key, ', '.join([' '.join(word) for word in n_grams_dict[key]]))


if __name__ == '__main__':
  main()
