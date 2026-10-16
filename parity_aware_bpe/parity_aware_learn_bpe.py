#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Rico Sennrich, Negar Foroutan

""" Parity-aware BPE learns a tokenization that ensures parity in token lengths across languages on a multi-parallel development set.
Unlike standard BPE, which optimizes merges based on a single corpus, this approach explicitly considers cross-lingual fairness during the tokenization process.
"""

from __future__ import unicode_literals

import os
import sys
import inspect
import codecs
import re
import copy
import argparse
import warnings
import tempfile
import functools
import operator
import numpy
import logging
import csv

from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter, deque
from contextlib import contextmanager

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add handler to logger
logger.addHandler(ch)

from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers import pre_tokenizers

BYTELEVEL_PRETOKENIZER_ENABLED = True


try:
    from tqdm import tqdm
    tqdm.monitor_interval = 0
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator

def create_parser(subparsers=None):

    if subparsers:
        parser = subparsers.add_parser('parity-aware-learn-bpe',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn Parity-aware BPE-based word segmentation")
    else:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn Parity-aware BPE-based word segmentation")

    parser.add_argument(
        '--variant', type=str, default='base',
        help="Partiy-aware BPE variant, either 'base' or 'window'. Default: %(default)s")
    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=[sys.stdin], nargs='*',
        metavar='PATHS',
        help="Input texts (default: standard input).")
    parser.add_argument(
        '--dev', '-d', type=argparse.FileType('r'), nargs='*',
        metavar='PATHS',
        help="Development texts (are used for parity computation).")
    parser.add_argument(
        '--ratio', '-r', type=float, nargs='*',
        help="Desired ratio of compression (comparing to pre-tokenized length) per input language. Can be used for parity computation in lieu of development set.")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file for BPE codes (default: standard output)")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s)")
    parser.add_argument(
        '--global-merges', '-g', type=int, default=0,
        help="For first INT merge operations, do merge based on global statistics instead of parity-driven language-specifc ones (default: %(default)s)")
    parser.add_argument(
        '--min-frequency', type=int, default=2, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s)')
    parser.add_argument(
        '--preload', type=argparse.FileType('r'), default=None,
        metavar='PATH',
        help="Preload merges from BPE file (default: None). Can be used to continue learning with different settings (e.g. without whitespace pre-tokenization for SuperBPE).")
    parser.add_argument(
        '--pretokenize', type=str, default=['whitespace', 'bytelevel'], nargs='*',
        # metavar='STR',
        choices=['whitespace', 'bytelevel'],
        help="Huggingface pre-tokenizer(s) to apply. (default: %(default)s)")
    parser.add_argument('--dict-input', action="store_true",
        help="If set, input file is interpreted as a dictionary where each line contains a word-count pair")
    parser.add_argument(
        '--total-symbols', '-t', action="store_true",
        help="subtract number of characters from the symbols to be generated (so that '--symbols' becomes an estimate for the total number of symbols needed to encode text).")
    parser.add_argument(
        '--window-size', '-w', type=int, default=100,
        help="Size of the context window for the moving-window balancingvariant of parity-aware BPE (default: %(default)s)")
    parser.add_argument(
        '--alpha', type=int, default=2,
        help="Ratio of the context window for the moving-window balancing variant of parity-aware BPE (default: %(default)s)")
    parser.add_argument(
        '--num-workers', type=int, default=1,
        help="Number of processors to process texts, only supported in Python3. If -1, set `multiprocessing.cpu_count()`. (default: %(default)s)")
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")
    parser.add_argument(
        '--utf8-merge-check-strategy', type=str, default='none',
        choices=['none', 'max', 'rescan', 'rule1',
                 'block3', 'block4', 'block5', 'block34', 'block35', 'block45', 'block345',
                 'char_first'],
        help="UTF-8 merge filtering strategy. Case-blocking strategies (block3, block4, block5, "
             "block34, block35, block45, block345) reject merges matching the specified case "
             "numbers: 3=F+P, 4=S+F, 5=S+P. 'char_first' blocks cases 3+4+5 and forces "
             "character assembly first. 'rule1' blocks bare prefix+suffix merges. "
             "'max'/'rescan' are selection-time filters. Default: %(default)s")
    parser.add_argument(
        '--merge-log', type=str, default=None,
        help="Optional CSV path for logging selected merges and their UTF-8 classification."
    )
    return parser

def get_vocabulary(fobj, is_dict=False, num_workers=1):
    """ Reads text and return dictionary that encodes vocabulary.
    Args:
        fobj (file-like object): The input file object to read from.
        is_dict (bool): If True, the input is treated as a dictionary file.
        num_workers (int): The number of worker processes to use for parallel processing.
    Returns:
        Counter: A Counter object mapping words to their frequencies.
    """
    vocab = Counter()

    strip_chars = '\r\n '
    split_char = ' '

    if is_dict:
        for i, line in enumerate(fobj):
            try:
                word, count = line.strip(strip_chars).split(split_char)
            except:
                print('Failed reading vocabulary file at line {0}: {1}'.format(i, line))
                sys.exit(1)
            vocab[word] += int(count)
    elif num_workers == 1 or fobj.name == '<stdin>':
        if num_workers > 1:
            warnings.warn("In parallel mode, the input cannot be STDIN. Using 1 processor instead.")
        for i, line in enumerate(fobj):
            # spliting the line using huggingface bpe-pre_tokenizer
            split_line = [item[0] for item in pre_tokenizer.pre_tokenize_str(line)]
            for word in split_line:
                if word:
                    vocab[word] += 1
            
    elif num_workers > 1:

        with open_file(fobj.name, 'r') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = int(size / num_workers)
            offsets = [0 for _ in range(num_workers + 1)]
            for i in range(1, num_workers):
                f.seek(chunk_size * i)
                pos = f.tell()
                while True:
                    try:
                        line = f.readline()
                        break
                    except UnicodeDecodeError:
                        pos -= 1
                        f.seek(pos)
                offsets[i] = f.tell()
                assert 0 <= offsets[i] < 1e20, "Bad new line separator, e.g. '\\r'"

        vocab_files = []
        pool = Pool(processes=num_workers)
        for i in range(num_workers):
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.close()
            vocab_files.append(tmp)
            pool.apply_async(_get_vocabulary, (fobj.name, tmp.name, offsets[i], offsets[i + 1]))
        
        pool.close()
        pool.join()
        import pickle
        for i in range(num_workers):
            with open(vocab_files[i].name, 'r') as f:
                vocab += pickle.load(f)
            os.remove(vocab_files[i].name)
    else:
        raise ValueError('`num_workers` is expected to be a positive number, but got {}.'.format(num_workers))
    return vocab

def _get_vocabulary(infile, outfile, begin, end):
    import pickle
    vocab = Counter()
    with open_file(infile, 'r') as f:
        f.seek(begin)
        line = f.readline()
        while line:
            pos = f.tell()
            assert 0 <= pos < 1e20, "Bad new line separator, e.g. '\\r'"
            if end > 0 and pos > end:
                break
            split_line = [item[0] for item in pre_tokenizer.pre_tokenize_str(line)]
            for word in split_line:
                if word:
                    vocab[word] += 1
            line = f.readline()
    with open(outfile, 'w') as f:
        pickle.dump(vocab, f)

def pre_merge(vocab, bpe_codes):
    """Apply list of BPE merge operations to each item in vocab
    """

    new_vocab = Counter()

    for orig in vocab:

        if len(orig) == 1:
            new_vocab[orig] = vocab[orig]

        word = list(orig[:-1]) + [orig[-1]]

        while len(word) > 1:

            # get list of symbol pairs; optionally apply dropout
            pairs = [(bpe_codes[pair],i,pair) for (i,pair) in enumerate(zip(word, word[1:])) if pair in bpe_codes]

            if not pairs:
                break

            #get first merge operation in list of BPE codes
            bigram = min(pairs)[2]

            # find start position of all pairs that we want to merge
            positions = [i for (rank,i,pair) in pairs if pair == bigram]

            i = 0
            new_word = []
            bigram = ''.join(bigram)
            for j in positions:
                # merges are invalid if they start before current position. This can happen if there are overlapping pairs: (x x x -> xx x)
                if j < i:
                    continue
                new_word.extend(word[i:j]) # all symbols before merged pair
                new_word.append(bigram) # merged pair
                i = j+2 # continue after merged pair
            new_word.extend(word[i:]) # add all symbols until end of word
            word = new_word

        word = tuple(word)
        new_vocab[word] = vocab[orig]

    return new_vocab

@functools.lru_cache(maxsize=1)
def get_bytelevel_char_to_byte():
    """Return the inverse GPT-2 byte-level mapping used by Hugging Face ByteLevel."""
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))

    codepoints = byte_values[:]
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1

    return {chr(codepoint): byte for byte, codepoint in zip(byte_values, codepoints)}

def symbol_to_raw_bytes(symbol):
    decoder = get_bytelevel_char_to_byte()
    return bytes(decoder[ch] for ch in symbol)


def valid_utf8_char_len(raw, i):
    for length in (1, 2, 3, 4):
        if i + length <= len(raw):
            try:
                raw[i:i+length].decode("utf-8")
                return length
            except UnicodeDecodeError:
                pass
    return 0


def is_valid_utf8_char_prefix(raw):
    """
    True iff raw is a proper prefix of exactly one UTF-8 character.
    """
    n = len(raw)
    if n == 0 or n > 3:
        return False

    b0 = raw[0]

    # 1-byte ASCII cannot be an unfinished prefix
    if 0x00 <= b0 <= 0x7F:
        return False

    # len 1: valid lead byte of a multi-byte char
    if n == 1:
        return (0xC2 <= b0 <= 0xDF) or (0xE0 <= b0 <= 0xEF) or (0xF0 <= b0 <= 0xF4)

    b1 = raw[1]

    def cont(x):
        return 0x80 <= x <= 0xBF

    # len 2
    if n == 2:
        # prefix of a 3-byte char
        if b0 == 0xE0:
            return 0xA0 <= b1 <= 0xBF
        if 0xE1 <= b0 <= 0xEC or 0xEE <= b0 <= 0xEF:
            return cont(b1)
        if b0 == 0xED:
            return 0x80 <= b1 <= 0x9F

        # prefix of a 4-byte char
        if b0 == 0xF0:
            return 0x90 <= b1 <= 0xBF
        if 0xF1 <= b0 <= 0xF3:
            return cont(b1)
        if b0 == 0xF4:
            return 0x80 <= b1 <= 0x8F

        return False

    # len 3: prefix of a 4-byte char only
    if n == 3:
        b2 = raw[2]
        if not cont(b2):
            return False

        if b0 == 0xF0:
            return 0x90 <= b1 <= 0xBF
        if 0xF1 <= b0 <= 0xF3:
            return cont(b1)
        if b0 == 0xF4:
            return 0x80 <= b1 <= 0x8F

        return False

    return False


@functools.lru_cache(maxsize=200000)
def utf8_symbol_state(symbol):
    """
    Returns:
      'complete'  -> valid UTF-8 string
      'prefix'    -> full UTF-8 chars plus one unfinished char lead at the end
      'suffix'    -> starts with a continuation byte (cannot start a character)
    """
    if not BYTELEVEL_PRETOKENIZER_ENABLED:
        return "complete"

    try:
        raw = symbol_to_raw_bytes(symbol)
    except KeyError:
        return "complete"

    i = 0
    while i < len(raw):
        length = valid_utf8_char_len(raw, i)
        if length > 0:
            i += length
            continue

        # whatever remains must be a valid unfinished prefix at the end
        return "prefix" if is_valid_utf8_char_prefix(raw[i:]) else "suffix"

    return "complete"


def is_admissible_utf8_symbol(symbol):
    return utf8_symbol_state(symbol) in ("complete", "prefix")


def is_admissible_utf8_pair(pair):
    return is_admissible_utf8_symbol("".join(pair))


# ---------------------------------------------------------------------------
# Generic case-blocking admissibility
# ---------------------------------------------------------------------------

# Map from strategy name to the set of cases it blocks.
# Cases:  3 = F+P,  4 = S+F,  5 = S+P
STRATEGY_BLOCKED_CASES = {
    'block3':     frozenset({3}),
    'block4':     frozenset({4}),
    'block5':     frozenset({5}),
    'block34':    frozenset({3, 4}),
    'block35':    frozenset({3, 5}),
    'block45':    frozenset({4, 5}),
    'block345':   frozenset({3, 4, 5}),
    'char_first': frozenset({3, 4, 5}),
}

# All strategy names that use local-style filtering (checked in get_pair_statistics
# and update_pair_statistics to gate whether new pairs are added to the pool).
LOCAL_STRATEGIES = frozenset(STRATEGY_BLOCKED_CASES.keys()) | {'rule1'}


def get_block_reason(pair, strategy):
    """Single source of truth for why a pair is rejected under a local strategy.

    Returns one of:
        'case3' / 'case4' / 'case5'  -- blocked by case-blocking strategy
        'rule1' / 'utf8'             -- blocked by rule1 strategy
        None                         -- pair is admissible

    Cases:
        3: F+P  (complete + prefix)   -- composite recoverable prefix
        4: S+F  (suffix  + complete)  -- produces suffix-starting token
        5: S+P  (suffix  + prefix)    -- produces suffix-starting token

    Case-blocking strategies (block3, block4, block5, block34, ...) are TRUE
    INDEPENDENT ABLATIONS: they block ONLY the listed cases and apply no other
    well-formedness check. Pairs whose merged symbol would be a suffix (e.g.
    case-4/5 not in the list, F+S, S+S) are ADMITTED, so the resulting BPE may
    contain tokens that start with continuation bytes. This is intentional --
    it isolates the cost of each individual case.

    Rule1 retains its original behavior, including a UTF-8 well-formedness
    fallback, because it is a separate research direction.
    """
    if strategy == 'rule1':
        if utf8_symbol_state(pair[0]) == 'prefix' and utf8_symbol_state(pair[1]) == 'suffix':
            raw = symbol_to_raw_bytes(pair[0])
            if valid_utf8_char_len(raw, 0) == 0:
                return 'rule1'
        if not is_admissible_utf8_symbol("".join(pair)):
            return 'utf8'
        return None

    if strategy in STRATEGY_BLOCKED_CASES:
        blocked_cases = STRATEGY_BLOCKED_CASES[strategy]
        left = utf8_symbol_state(pair[0])
        right = utf8_symbol_state(pair[1])
        if 3 in blocked_cases and left == 'complete' and right == 'prefix':
            return 'case3'
        if 4 in blocked_cases and left == 'suffix' and right == 'complete':
            return 'case4'
        if 5 in blocked_cases and left == 'suffix' and right == 'prefix':
            return 'case5'
        return None

    return None


def is_admissible_for_strategy(pair, strategy):
    """Return True if pair is admissible under the given local-style strategy."""
    if strategy in LOCAL_STRATEGIES:
        return get_block_reason(pair, strategy) is None
    return is_admissible_utf8_pair(pair)
    
@functools.lru_cache(maxsize=200000)
def is_valid_utf8_symbol(symbol):
    """Check whether a byte-level symbol is a standalone valid UTF-8 byte sequence."""
    if not BYTELEVEL_PRETOKENIZER_ENABLED:
        return True

    decoder = get_bytelevel_char_to_byte()
    try:
        raw_bytes = bytes(decoder[char] for char in symbol)
    except KeyError:
        return True

    try:
        raw_bytes.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def is_valid_utf8_pair(pair):
    return is_valid_utf8_symbol(''.join(pair))

def symbol_utf8_type(symbol):
    return 'complete' if is_valid_utf8_symbol(symbol) else 'incomplete'


def symbol_to_hex(symbol):
    if not BYTELEVEL_PRETOKENIZER_ENABLED:
        return ''
    decoder = get_bytelevel_char_to_byte()
    try:
        return ' '.join(f'{decoder[ch]:02X}' for ch in symbol)
    except KeyError:
        return ''


def classify_merge_pair(pair):
    left_state = utf8_symbol_state(pair[0])
    right_state = utf8_symbol_state(pair[1])
    merged_state = utf8_symbol_state(''.join(pair))
    merged_valid = is_valid_utf8_pair(pair)

    if merged_state == "complete":
        category = "good"
    elif merged_state == "prefix":
        category = "recoverable"
    else:
        category = "bad"

    return {
        "left_state": left_state,
        "right_state": right_state,
        "merged_state": merged_state,
        "merged_valid": merged_valid,
        "category": category,
        "left_hex": symbol_to_hex(pair[0]),
        "right_hex": symbol_to_hex(pair[1]),
        "merged_hex": symbol_to_hex(''.join(pair)),
    }


def open_merge_log(path):
    if not path:
        return None, None

    f = open(path, 'w', encoding='utf-8', newline='')
    writer = csv.writer(f)
    writer.writerow([
        'step',
        'status',
        'selected_language_index',
        'left_symbol',
        'right_symbol',
        'merged_symbol',
        'left_hex',
        'right_hex',
        'merged_hex',
        'left_state',
        'right_state',
        'merged_state',
        'merged_valid_utf8',
        'category',
        'count',
        'selected_frequency',
        'frequency_vector'
    ])
    return f, writer


def write_merge_log(writer, step, max_index, pair, pair_stats):
    if writer is None:
        return

    info = classify_merge_pair(pair)
    writer.writerow([
        step,
        'accepted',
        max_index,
        pair[0],
        pair[1],
        ''.join(pair),
        info['left_hex'],
        info['right_hex'],
        info['merged_hex'],
        info['left_state'],
        info['right_state'],
        info['merged_state'],
        info['merged_valid'],
        info['category'],
        1,
        int(pair_stats[max_index]),
        ';'.join(map(str, pair_stats.tolist()))
    ])


def buffer_blocked_merge(buffer, pair, freq, blocked_reason):
    """Accumulate a blocked-merge event into the per-step aggregation buffer.

    Buffer key is (left, right, reason). Value is {'count': N, 'freq': vec}.
    """
    if buffer is None:
        return
    key = (pair[0], pair[1], blocked_reason)
    entry = buffer.get(key)
    if entry is None:
        buffer[key] = {'count': 1, 'freq': numpy.array(freq, copy=True)}
    else:
        entry['count'] += 1
        entry['freq'] = entry['freq'] + freq


def flush_blocked_buffer(writer, step, buffer):
    """Write one aggregated row per unique (pair, reason) at `step`, then clear buffer.

    Rows are sorted by total frequency descending so the most impactful blocks
    come first within each step.
    """
    if writer is None or buffer is None or not buffer:
        return

    def total_freq(entry):
        f = entry['freq']
        return int(sum(f.tolist() if hasattr(f, 'tolist') else list(f)))

    items = sorted(buffer.items(), key=lambda kv: -total_freq(kv[1]))
    for (left, right, reason), entry in items:
        info = classify_merge_pair((left, right))
        freq = entry['freq']
        freq_list = freq.tolist() if hasattr(freq, 'tolist') else list(freq)
        writer.writerow([
            step,
            f'blocked_{reason}',
            -1,
            left,
            right,
            left + right,
            info['left_hex'],
            info['right_hex'],
            info['merged_hex'],
            info['left_state'],
            info['right_state'],
            info['merged_state'],
            info['merged_valid'],
            info['category'],
            entry['count'],
            int(sum(freq_list)),
            ';'.join(map(str, freq_list)),
        ])
    buffer.clear()

def invalidate_pair(pair, stats, big_stats, indices, array_length):
    """Permanently zero a rejected pair so it is no longer considered for merging."""
    if pair in stats:
        stats[pair] = numpy.zeros(array_length, dtype=int)
    if big_stats is not None and pair in big_stats:
        big_stats[pair] = numpy.zeros(array_length, dtype=int)
    if indices is not None and pair in indices:
        indices[pair] = defaultdict(int)


def invalidate_pairs_globally(stats, big_stats, indices, array_length):
    """Rescan the full candidate pool and zero any pair that is not UTF-8-valid."""
    pairs_to_check = set(stats.keys())
    if big_stats is not None:
        pairs_to_check.update(big_stats.keys())

    for pair in pairs_to_check:
        if not is_admissible_utf8_pair(pair):
            invalidate_pair(pair, stats, big_stats, indices, array_length)


def select_most_frequent_pair(stats, max_index, merge_check_strategy, big_stats, indices, array_length):
    """Select the current best pair, optionally rejecting invalid pairs at selection time."""
    most_frequent = max(stats, key=lambda x: (stats[x][max_index], x))
    if merge_check_strategy != 'max':
        return most_frequent

    while stats[most_frequent][max_index] > 0 and not is_admissible_utf8_pair(most_frequent):
        invalidate_pair(most_frequent, stats, big_stats, indices, array_length)
        most_frequent = max(stats, key=lambda x: (stats[x][max_index], x))

    return most_frequent


def update_pair_statistics(pair, changed, stats, indices, merge_check_strategy='none',
                           blocked_buffer=None):
    """ Minimally updates the indices and frequency of symbol pairs.
    If we merge a pair of symbols, only pairs that overlap with occurrences
    of this pair are affected, and need to be updated.

    Args:
        pair (tuple): A tuple of two characters (first, second) that form the pair to be updated.
        changed (list): A list of changes made, where each change is a tuple containing the index of the word, the new word, the old word, and its frequency.
        stats (defaultdict): A dictionary mapping symbol pairs (tuples) to their frequencies (numpy arrays).
        indices (defaultdict): A dictionary mapping symbol pairs (tuples) to their indices (defaultdicts).
        blocked_buffer: optional dict for accumulating blocked-pair events at this step (flushed by caller).
    Returns:
        None: The function updates the `stats` and `indices` dictionaries in place.
    """
    stats[pair] = 0
    indices[pair] = defaultdict(int)
    first, second = pair
    new_pair = first+second
    for j, word, old_word, freq in changed:

        # find all instances of pair, and update frequency/indices around it
        i = 0
        while True:
            # find first symbol
            try:
                i = old_word.index(first, i)
            except ValueError:
                break
            # if first symbol is followed by second symbol, we've found an occurrence of pair (old_word[i:i+2])
            if i < len(old_word)-1 and old_word[i+1] == second:
                # assuming a symbol sequence "A B C", if "B C" is merged, reduce the frequency of "A B"
                if i:
                    prev = old_word[i-1:i+1]
                    stats[prev] -= freq
                    indices[prev][j] -= 1
                if i < len(old_word)-2:
                    # assuming a symbol sequence "A B C B", if "B C" is merged, reduce the frequency of "C B".
                    # however, skip this if the sequence is A B C B C, because the frequency of "C B" will be reduced by the previous code block
                    if old_word[i+2] != first or i >= len(old_word)-3 or old_word[i+3] != second:
                        nex = old_word[i+1:i+3]
                        stats[nex] -= freq
                        indices[nex][j] -= 1
                i += 2
            else:
                i += 1

        i = 0
        while True:
            try:
                # find new pair
                i = word.index(new_pair, i)
            except ValueError:
                break
            # assuming a symbol sequence "A BC D", if "B C" is merged, increase the frequency of "A BC"
            if i:
                prev = word[i-1:i+1]
                reason = get_block_reason(prev, merge_check_strategy) if merge_check_strategy in LOCAL_STRATEGIES else None
                if reason is None:
                    stats[prev] += freq
                    indices[prev][j] += 1
                else:
                    buffer_blocked_merge(blocked_buffer, prev, freq, reason)
            # assuming a symbol sequence "A BC B", if "B C" is merged, increase the frequency of "BC B"
            # however, if the sequence is A BC BC, skip this step because the count of "BC BC" will be incremented by the previous code block
            if i < len(word)-1 and word[i+1] != new_pair:
                nex = word[i:i+2]
                reason = get_block_reason(nex, merge_check_strategy) if merge_check_strategy in LOCAL_STRATEGIES else None
                if reason is None:
                    stats[nex] += freq
                    indices[nex][j] += 1
                else:
                    buffer_blocked_merge(blocked_buffer, nex, freq, reason)
            i += 1


def get_pair_statistics(vocab, merge_check_strategy='none', blocked_buffer=None):
    """ Counts frequency of all symbol pairs, and create index.
    Args:
        vocab (list): A list of tuples, where each tuple contains a word (as a tuple of characters) and its frequency in each language.
        merge_check_strategy (str): Local strategy name; pairs failing its admissibility are excluded from the pool.
        blocked_buffer: Optional dict for accumulating step-0 blocked events (flushed by caller).
    Returns:
        tuple: A tuple containing two dictionaries:
            - stats (defaultdict): A dictionary mapping symbol pairs (tuples) to their frequencies (numpy arrays).
            - indices (defaultdict): A dictionary mapping symbol pairs (tuples) to their indices (defaultdicts).
    """

    # data structure of pair frequencies
    stats = defaultdict(lambda: numpy.zeros(len(vocab[0][1]),dtype=int))

    #index from pairs to words
    indices = defaultdict(lambda: defaultdict(int))

    for i, (word, freq) in enumerate(vocab):
        prev_char = word[0]
        for char in word[1:]:
            pair = (prev_char, char)
            reason = get_block_reason(pair, merge_check_strategy) if merge_check_strategy in LOCAL_STRATEGIES else None
            if reason is None:
                stats[pair] += freq
                indices[pair][i] += 1
            else:
                buffer_blocked_merge(blocked_buffer, pair, freq, reason)
            prev_char = char

    return stats, indices


def replace_pair(pair, vocab, indices):
    """ Replaces all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'
    Args:
        pair (tuple): A tuple of two characters (first, second) that form the pair to be replaced.
        vocab (defaultdict): A dictionary mapping words (tuples of characters) to their frequencies in each language.
        indices (defaultdict): A dictionary mapping pairs of characters to their indices in the vocabulary.
    Returns:
        list: A list of changes made, where each change is a tuple containing the index of the word, the new word, the old word, and its frequency.
    """
    split_char = ' '
    first, second = pair
    
    pair_str = ''.join(pair)
    pair_str = pair_str.replace('\\','\\\\')
    pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
    changes = []

    if sys.version_info < (3, 0):
        iterator = indices[pair].iteritems()
    else:
        iterator = indices[pair].items()
    for j, freq in iterator:
        if freq < 1:
            continue
        word, freq = vocab[j]
        new_word = split_char.join(word)
        new_word = pattern.sub(pair_str, new_word)
        new_word = tuple(new_word.split(split_char))

        vocab[j] = (new_word, freq)
        changes.append((j, new_word, word, freq))

    return changes

def prune_stats(stats, big_stats, threshold, full_sync=False):
    """ Prunes statistics dict for efficiency of max(). 
    The frequency of a symbol pair never increases, so pruning is generally safe
    (until the most frequent pair is less frequent than a pair we previously pruned)
    big_stats keeps full statistics for when we need to access pruned items
    """
    for item,freq in list(stats.items()):
        if full_sync or numpy.all(freq < threshold):
            del stats[item]
            if numpy.any(freq < 0):
                big_stats[item] += freq
            else:
                big_stats[item] = freq


def replace_pair_dict(pair, vocab):
    """ Replaces all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'.
    Args:
        pair (tuple): A tuple of two characters (first, second) that form the pair to be replaced.
        vocab (defaultdict): A dictionary mapping words (tuples of characters) to their frequencies in each language.
    Returns:
        numpy.ndarray: An array where each element corresponds to the change in text length (the sum of frequency*length for all vocab items) for each language.
    """
    length_change = None
    split_char = ' '
    first, second = pair

    pair_str = ''.join(pair)
    pair_str = pair_str.replace('\\','\\\\')
    pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')

    for word, freq in list(vocab.items()):
        if first in word and second in word and pair in zip(word, word[1:]):
            new_word = split_char.join(word)
            new_word = pattern.sub(pair_str, new_word)
            new_word = tuple(new_word.split(split_char))
            del vocab[word]
            vocab[new_word] = freq

            if length_change is None:
                length_change = numpy.zeros(len(freq), dtype=int)

            length_change += (len(word)-len(new_word))*freq

    if length_change is None:
        length_change = 0

    return length_change

@contextmanager
def open_file(filename, mode):
    if mode in ('r', 'w'):
        f = open(filename, mode, encoding="utf-8")
    elif mode in ('rb', 'wb'):
        f = open(filename, mode)
    try:
        yield f
    finally:
        f.close()


def preprocess_input_data(infiles, devfiles, is_dict=False, total_symbols=False, num_global=0, num_workers=1, bpe_file=None, merge_check_strategy='none', blocked_buffer=None):
    """ Reads input files and creates vocabulary data structure.
    Args:
        infiles (list[str]): A list of input file paths.
        devfiles (list[str]): A list of development file paths.
        is_dict (bool): Whether the input files are in dictionary format.
        total_symbols (bool): Whether to count total symbols.
        num_global (int): The number of global symbols.
        num_workers (int): The number of worker threads to use.
        bpe_file (fobj): file containing merge operations to pre-apply before learning
    Returns:
        tuple: A tuple containing:
            - dev_vocab (defaultdict): A dictionary mapping subwords to their frequencies in the development set.
            - sorted_vocab (list): A sorted list of tuples, where each tuple contains a subword and its frequency in each language.
            - stats (defaultdict): A dictionary mapping symbol pairs (tuples) to their frequencies (numpy arrays).
            - indices (defaultdict): A dictionary mapping symbol pairs (tuples) to their indices (defaultdicts).
            - big_stats (defaultdict): A dictionary containing full statistics for all symbol pairs.
            - threshold (numpy.ndarray): An array of thresholds for pruning statistics.
            - lengths (numpy.ndarray): An array where each element corresponds to the sum of frequency*length for all vocab items in the development set.
            - array_length (int): The length of the vocabulary array, which is the number of languages plus one for concatenation.
    """

    if not bpe_file is None:
        
        # ignore first line containing version information (if it exists)
        line = bpe_file.readline()
        offset = 1
        if not line.startswith('version'):
            bpe_file.seek(0)
            offset = 0
        
        bpe_codes = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(bpe_file.read().rstrip('\n').split('\n'))]

        for i, item in enumerate(bpe_codes):
            if len(item) != 2:
                sys.stderr.write('Error: invalid line {0} in BPE codes file: {1}\n'.format(i+offset, ' '.join(item)))
                sys.stderr.write('The line should exist of exactly two subword units, separated by whitespace\n')
                sys.exit(1)

        # some hacking to deal with duplicates (only consider first instance)
        bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(bpe_codes)))])
    else:
        bpe_codes = None

    vocabs = []
    joint_keys = set()
    for f in infiles:
        vocab = get_vocabulary(f, is_dict, num_workers)
        if not bpe_codes is None:
            vocab = pre_merge(vocab, bpe_codes)
        vocab = dict([(tuple(x,) ,y) for (x,y) in vocab.items()])
        vocabs.append(vocab)
        joint_keys = joint_keys.union(vocab.keys())

    dev_vocabs = []
    dev_keys = set()
    if devfiles:
        for f in devfiles:
            vocab = get_vocabulary(f, is_dict, num_workers)
            if not bpe_codes is None:
                vocab = pre_merge(vocab, bpe_codes)
            vocab = dict([(tuple(x,) ,y) for (x,y) in vocab.items()])
            
            dev_vocabs.append(vocab)
            dev_keys = dev_keys.union(vocab.keys())
    array_length = len(vocabs)
    if num_global:
        array_length += 1

    # merge vocabularies. Data structure maps from subword to list of frequency in each language, plus one for concatenation
    vocab = defaultdict(lambda: numpy.zeros(array_length,dtype=int))
    for i in range(len(vocabs)):
        for key in joint_keys:
            vocab[key][i] = vocabs[i].get(key, 0)

    if num_global:
        for key in joint_keys:
            vocab[key][-1] = sum(vocab[key])

    # merge dev vocabularies. Data structure maps from subword to list of frequency*word_length in each language
    dev_vocab = defaultdict(lambda: numpy.zeros(len(dev_vocabs),dtype=int))
    
    if dev_vocabs:
        for i in range(len(dev_vocabs)):
            for key in dev_keys:
                dev_vocab[key][i] = dev_vocabs[i].get(key, 0)

    sorted_vocab = sorted(vocab.items(), key=lambda x: sum(x[1]), reverse=True)
    stats, indices = get_pair_statistics(sorted_vocab, merge_check_strategy=merge_check_strategy, blocked_buffer=blocked_buffer)
    big_stats = copy.deepcopy(stats)
    if merge_check_strategy == 'rescan':
        invalidate_pairs_globally(stats, big_stats, indices, array_length)

    if total_symbols:
        uniq_char_internal = set()
        uniq_char_final = set()
        for word in vocab:
            for char in word[:-1]:
                uniq_char_internal.add(char)
            uniq_char_final.add(word[-1])
        sys.stderr.write('Number of word-internal characters: {0}\n'.format(len(uniq_char_internal)))
        sys.stderr.write('Number of word-final characters: {0}\n'.format(len(uniq_char_final)))
        sys.stderr.write('Reducing number of merge operations by {0}\n'.format(len(uniq_char_internal) + len(uniq_char_final)))
        num_symbols -= len(uniq_char_internal) + len(uniq_char_final)

    # threshold is inspired by Zipfian assumption, but should only affect how often we re-sync with full big_stats
    threshold = numpy.zeros(array_length,dtype=int)
    for l in range(array_length):
        threshold[l] = stats[max(stats, key=lambda x: (stats[x][l], x))][l] / 10

    if dev_vocab:
        lengths = functools.reduce(numpy.add, [len(key)*value for key, value in dev_vocab.items()])
    else:
        lengths = None
    return (dev_vocab, sorted_vocab, stats, indices, big_stats, threshold, lengths, array_length)

def learn_bpe(infiles, outfile, devfiles, num_symbols, min_frequency=2, verbose=False, is_dict=False, total_symbols=False, num_global=0, ratio=None, num_workers=1, bpe_file=None, merge_check_strategy='none', merge_log=None):    
    """
    Learn `num_symbols` merge operations using Parity-aware BPE from the provided training and development files
    and write the learned BPE operations to `outfile`.

    Args:
        infiles (list[str]): 
            List of paths to the input text files used for training the BPE model.
        outfile (file-like or str): 
            Path to the file where the learned BPE merge operations will be saved.
        devfiles (list[str]): 
            List of development text files, used for validation during BPE learning.
        num_symbols (int): 
            Number of BPE merge operations to learn.
        min_frequency (int, optional): 
            Minimum frequency threshold for a pair to be considered during merging. Defaults to 2.
        verbose (bool, optional): 
            If True, print detailed logs to stderr during training. Defaults to False.
        is_dict (bool, optional): 
            If True, interpret the input files as dictionaries with frequency counts. Defaults to False.
        total_symbols (bool, optional): 
            If True, subtract number of characters from the symbols to be generated (so that 'num_symbols' becomes an estimate for the total number of symbols needed to encode text). Defaults to False.
        num_global (int, optional): 
            Number of initial merges to perform globally across all corpora before handling them separately. Defaults to 0.
        ratio (list[float[):
            Desired ratio of compression (comparing to pre-tokenized length) per input language. Can be used for parity computation in lieu of development set.
        num_workers (int, optional): 
            Number of worker processes to use for parallel computations (if supported). Defaults to 1.
        bpe_file (file-like, optional):
            Path to file from which to pre-load BPE merges (to continue learning with different settings, e.g. for SuperBPE).

    Returns:
        None
            The function writes the learned BPE merge rules to the specified `outfile`.
    """
    logger.info("Learning parity-aware BPE with the following parameters:"
        "\n  num_symbols: {0}, min_frequency: {1}, verbose: {2}, is_dict: {3}, total_symbols: {4}, num_global: {5}, num_workers: {6}, utf8_merge_check_strategy: {7}".format(
            num_symbols, min_frequency, verbose, is_dict, total_symbols, num_global, num_workers, merge_check_strategy))
    
    # version numbering allows bckward compatibility
    outfile.write('#version: 0.2\n')

    merge_log_file, merge_log_writer = open_merge_log(merge_log)
    blocked_buffer = {} if merge_log_writer is not None else None

    dev_vocab, sorted_vocab, stats, indices, big_stats, threshold, lengths, array_length = \
        preprocess_input_data(infiles, devfiles, is_dict, total_symbols, num_global, num_workers, bpe_file,
                              merge_check_strategy=merge_check_strategy, blocked_buffer=blocked_buffer)

    # flush step-0 initial-build blocks before the merge loop starts
    flush_blocked_buffer(merge_log_writer, 0, blocked_buffer)

    if not ratio is None:
        initial_lengths = functools.reduce(numpy.add, [len(key)*value for key, value in sorted_vocab])
        lengths = numpy.copy(initial_lengths)

    for i in tqdm(range(num_symbols), desc="parity-aware BPE..."):
        if stats:
            if i < num_global:
                if verbose:
                    sys.stderr.write('lengths {0}: picking best subword based on concatenation\n'.format(lengths))
                max_index = -1

            else:
                if not ratio is None:
                    # we want to find the language with the least compression, adjusted by the user_defined desired ratio
                    compression_rates = initial_lengths/lengths
                    adjusted_compression_rates = compression_rates/ratio
                    max_index, max_value = min(enumerate(adjusted_compression_rates), key=operator.itemgetter(1))
                    if verbose:
                        sys.stderr.write('initial lengths  {0}\nlengths {1}\n'.format(initial_lengths, lengths))
                        sys.stderr.write('compression rates {0}\nadjusted compression rates {1}: picking best subword in corpus {2} \n'.format(compression_rates, adjusted_compression_rates, max_index))

                else:
                    max_index, max_value = max(enumerate(lengths), key=operator.itemgetter(1))
                    if verbose:
                        sys.stderr.write('lengths {0}: picking best subword in corpus {1} \n'.format(lengths, max_index))

            most_frequent = select_most_frequent_pair(stats, max_index, merge_check_strategy, big_stats, indices, array_length)

        # we probably missed the best pair because of pruning; go back to full statistics
        if not stats or (i and stats[most_frequent][max_index] < threshold[max_index]):
            prune_stats(stats, big_stats, threshold, full_sync=True)
            stats = copy.deepcopy(big_stats)
            most_frequent = select_most_frequent_pair(stats, max_index, merge_check_strategy, big_stats, indices, array_length)

            # threshold is inspired by Zipfian assumption, but should only affect how often we re-sync with full big_stats
            for l in range(array_length):
                threshold[l] = stats[max(stats, key=lambda x: (stats[x][l], x))][l] * i/(i+10000.0)

            prune_stats(stats, big_stats, threshold)

        if stats[most_frequent][max_index] < min_frequency:
            sys.stderr.write(f'no pair has frequency >= {min_frequency}. Stopping for language {max_index} with length: {lengths}\n')
            break

        if verbose:
            sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, most_frequent[0], most_frequent[1], stats[most_frequent]))

        outfile.write('{0} {1}\n'.format(*most_frequent))

        write_merge_log(merge_log_writer, i, max_index, most_frequent, stats[most_frequent])
        
        changes = replace_pair(most_frequent, sorted_vocab, indices)

        if not ratio is None:
            length_change = functools.reduce(numpy.add, [(len(c[2])-len(c[1]))*c[3] for c in changes])
            lengths -= length_change
        else:
            length_change = replace_pair_dict(most_frequent, dev_vocab)
            lengths -= length_change

        update_pair_statistics(most_frequent, changes, stats, indices,
                               merge_check_strategy=merge_check_strategy,
                               blocked_buffer=blocked_buffer)
        # flush step-i update-time blocks after the accepted row was written above
        flush_blocked_buffer(merge_log_writer, i, blocked_buffer)

        if merge_check_strategy == 'rescan':
            invalidate_pairs_globally(stats, big_stats, indices, array_length)

        if not i % 100:
            prune_stats(stats, big_stats, threshold)

        stats[most_frequent] = numpy.zeros(array_length, dtype=int)

    if merge_log_file is not None:
        merge_log_file.close()


def select_language_index(lengths, selected_indices, selection_threshold, window_size):
    """ Selects the index of the language with the maximum length from the remaining valid indices.
    The selection is based on a moving window approach, where indices that have been selected too often
    are excluded from further consideration.

    Args:
        lengths (numpy.ndarray): An array of lengths for each language.
        selected_indices (deque): A deque containing the indices of previously selected languages.
        selection_threshold (float): The threshold ratio for selecting an index.
        window_size (int): The size of the moving window.

    Returns:
        int: The index of the selected language.
    """
    final_index = -1
    # Boolean mask to keep track of valid indices
    mask = numpy.ones(len(lengths), dtype=bool)  # Start with all elements valid

    while True:
        # Find the maximum index in the remaining elements
        valid_indices = numpy.where(mask)[0]  # Indices of unmasked elements
        max_index = valid_indices[numpy.argmax(lengths[valid_indices])]  # Max in valid range
        count = selected_indices.count(max_index)
        ratio = count * 1.0 / window_size
        if ratio <= selection_threshold:
            final_index = max_index
            break
        else:
            # Exclude this index from further consideration
            mask[max_index] = False

    return final_index

def learn_bpe_moving_window(infiles, outfile, devfiles, num_symbols, window_size=100, alpha=2, min_frequency=2, verbose=False, is_dict=False, total_symbols=False, num_global=0, ratio=None, num_workers=1, bpe_file=None, merge_check_strategy='none', merge_log=None):    
    """
    Learn `num_symbols` merge operations using Parity-aware BPE (moving-window balancing variant) from the provided training and development files
    and write the learned BPE operations to `outfile`.

    Args:
        infiles (list[str]): 
            List of paths to the input text files used for training the BPE model.
        outfile (file-like or str): 
            Path to the file where the learned BPE merge operations will be saved.
        devfiles (list[str]): 
            List of development text files, used for validation during BPE learning.
        num_symbols (int): 
            Number of BPE merge operations to learn.
        window_size (int, optional): 
            Size of the context window for the moving-window balancing variant of parity-aware BPE. Defaults to 100.
        alpha (int, optional): 
            Ratio of the context window for the moving-window balancing variant of parity-aware BPE. Defaults to 2.
        min_frequency (int, optional): 
            Minimum frequency threshold for a pair to be considered during merging. Defaults to 2.
        verbose (bool, optional): 
            If True, print detailed logs to stderr during training. Defaults to False.
        is_dict (bool, optional): 
            If True, interpret the input files as dictionaries with frequency counts. Defaults to False.
        total_symbols (bool, optional): 
            If True, subtract number of characters from the symbols to be generated (so that 'num_symbols' becomes an estimate for the total number of symbols needed to encode text). Defaults to False.
        num_global (int, optional): 
            Number of initial merges to perform globally across all corpora before handling them separately. Defaults to 0.
        ratio (list[float[):
            Desired ratio of compression (comparing to pre-tokenized length) per input language. Can be used for parity computation in lieu of development set.
        num_workers (int, optional): 
            Number of worker processes to use for parallel computations (if supported). Defaults to 1.
        bpe_file (file-like, optional):
            Path to file from which to pre-load BPE merges (to continue learning with different settings, e.g. for SuperBPE).

    Returns:
        None
            The function writes the learned BPE merge rules to the specified `outfile`.
    """
    logger.info("Using Parity-aware BPE (moving-window variant) with window size {0} and alpha {1}".format(window_size, alpha))
    logger.info("Learning parity-aware BPE with the following parameters:"
          "\n  num_symbols: {0}, min_frequency: {1}, verbose: {2}, is_dict: {3}, total_symbols: {4}, num_global: {5}, num_workers: {6}, utf8_merge_check_strategy: {7}".format(
              num_symbols, min_frequency, verbose, is_dict, total_symbols, num_global, num_workers, merge_check_strategy))
    
    # if continuing learning on top of existing BPE file, contents of BPE file are included in output
    if bpe_file:
        outfile.write(bpe_file.read())
        bpe_file.seek(0)
    else:
        # version numbering allows bckward compatibility
        outfile.write('#version: 0.2\n')

    merge_log_file, merge_log_writer = open_merge_log(merge_log)
    blocked_buffer = {} if merge_log_writer is not None else None

    dev_vocab, sorted_vocab, stats, indices, big_stats, threshold, lengths, array_length = \
        preprocess_input_data(infiles, devfiles, is_dict, total_symbols, num_global, num_workers, bpe_file,
                              merge_check_strategy=merge_check_strategy, blocked_buffer=blocked_buffer)

    # flush step-0 initial-build blocks before the merge loop starts
    flush_blocked_buffer(merge_log_writer, 0, blocked_buffer)

    if not ratio is None:
        initial_lengths = functools.reduce(numpy.add, [len(key)*value for key, value in sorted_vocab])
        lengths = numpy.copy(initial_lengths)

    selection_threshold = alpha * 1.0 / len(threshold)
    selected_indices = deque(maxlen=window_size)

    for i in tqdm(range(num_symbols), desc="Parity-aware BPE (moving-window variant)... \n"):
        if stats:
            if i < num_global:
                if verbose:
                    sys.stderr.write('lengths {0}: picking best subword based on concatenation\n'.format(lengths))
                max_index = -1

            else:
                if not ratio is None:
                    # we want to find the language with the least compression, adjusted by the user_defined desired ratio
                    compression_rates = initial_lengths/lengths
                    adjusted_compression_rates = compression_rates/ratio
                    # if verbose:
                        # sys.stderr.write('initial lengths  {0}\nlengths {1}\n'.format(initial_lengths, lengths))
                        # sys.stderr.write('compression rates {0}\nadjusted compression rates {1}\n'.format(compression_rates, adjusted_compression_rates))
                    max_index = select_language_index(-adjusted_compression_rates, selected_indices, selection_threshold, window_size)
                    selected_indices.append(max_index)
                    if verbose:
                        sys.stderr.write('initial lengths  {0}\nlengths {1}\n'.format(initial_lengths, lengths))
                        sys.stderr.write('compression rates {0}\nadjusted compression rates {1}: picking best subword in corpus {2} \n'.format(compression_rates, adjusted_compression_rates, max_index))

                else:
                    max_index = select_language_index(lengths, selected_indices, selection_threshold, window_size)
                    selected_indices.append(max_index)
                    if verbose:
                        sys.stderr.write('lengths {0}: picking best subword in corpus {1} \n'.format(lengths, max_index))

            most_frequent = select_most_frequent_pair(stats, max_index, merge_check_strategy, big_stats, indices, array_length)

        # we probably missed the best pair because of pruning; go back to full statistics
        if not stats or (i and stats[most_frequent][max_index] < threshold[max_index]):
            prune_stats(stats, big_stats, threshold, full_sync=True)
            stats = copy.deepcopy(big_stats)
            most_frequent = select_most_frequent_pair(stats, max_index, merge_check_strategy, big_stats, indices, array_length)

            # threshold is inspired by Zipfian assumption, but should only affect how often we re-sync with full big_stats
            for l in range(array_length):
                threshold[l] = stats[max(stats, key=lambda x: (stats[x][l], x))][l] * i/(i+10000.0)

            prune_stats(stats, big_stats, threshold)

        if stats[most_frequent][max_index] < min_frequency:
            sys.stderr.write(f'no pair has frequency >= {min_frequency}. Stopping for language {max_index} with length: {lengths}\n')
            break

        if verbose:
            sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, most_frequent[0], most_frequent[1], stats[most_frequent]))

        outfile.write('{0} {1}\n'.format(*most_frequent))

        write_merge_log(merge_log_writer, i, max_index, most_frequent, stats[most_frequent])
        
        changes = replace_pair(most_frequent, sorted_vocab, indices)

        if not ratio is None:
            length_change = functools.reduce(numpy.add, [(len(c[2])-len(c[1]))*c[3] for c in changes])
            lengths -= length_change
        else:
            length_change = replace_pair_dict(most_frequent, dev_vocab)
            lengths -= length_change

        update_pair_statistics(most_frequent, changes, stats, indices,
                               merge_check_strategy=merge_check_strategy,
                               blocked_buffer=blocked_buffer)
        # flush step-i update-time blocks after the accepted row was written above
        flush_blocked_buffer(merge_log_writer, i, blocked_buffer)

        if merge_check_strategy == 'rescan':
            invalidate_pairs_globally(stats, big_stats, indices, array_length)

        if not i % 100:
            prune_stats(stats, big_stats, threshold)

        stats[most_frequent] = numpy.zeros(array_length, dtype=int)

    if merge_log_file is not None:
        merge_log_file.close()

if __name__ == '__main__':

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    newdir = os.path.join(currentdir, 'subword_nmt')
    if os.path.isdir(newdir):
        warnings.warn(
            "this script's location has moved to {0}. This symbolic link will be removed in a future version. Please point to the new location, or install the package and use the command 'subword-nmt'".format(newdir),
            DeprecationWarning
        )

    parser = create_parser()
    args = parser.parse_args()

    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

    if args.num_workers <= 0:
        args.num_workers = cpu_count()
    
    
    if sys.version_info < (3, 0):
        print("Python 2 is deprecated. Use Python 3")
        sys.exit(1)

    # if we do parity-oriented merging, number of development sets needs to match number of input corpora
    if args.dev:
        assert(len(args.input) == len(args.dev))

    if args.ratio:
        assert(args.dev is None)
        assert(len(args.input) == len(args.ratio))
        args.ratio = numpy.array(args.ratio)
        #normalize ratios by first value given
        args.ratio = args.ratio/args.ratio[0]

    if args.dev is None and args.ratio is None:
        print("script requires either dev sets or ratios")
        sys.exit(1)

    # read/write files as UTF-8
    for i,f in enumerate(args.input):
        if f.name != '<stdin>':
            args.input[i] = codecs.open(f.name, encoding='utf-8')
    if args.dev:
        for i,f in enumerate(args.dev):
            args.dev[i] = codecs.open(f.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
    if args.preload is None:
        bpe_file = None
    else:
        bpe_file = codecs.open(args.preload.name, encoding='utf-8')


    pretokenizer_list = []
    for pretokenizer in args.pretokenize:
        if pretokenizer == 'whitespace':
            pretokenizer_list.append(Whitespace())
        elif pretokenizer == 'bytelevel':
            pretokenizer_list.append(ByteLevel(use_regex=False))
        else:
            raise ValueError("pretokenizer {0} is not implemented".format(pretokenizer))

    pre_tokenizer = pre_tokenizers.Sequence(pretokenizer_list)
    BYTELEVEL_PRETOKENIZER_ENABLED = 'bytelevel' in args.pretokenize

    if args.variant == 'base':
        learn_bpe(
            args.input, args.output, args.dev, args.symbols,
            args.min_frequency, args.verbose,
            num_global=args.global_merges,
            is_dict=args.dict_input,
            total_symbols=args.total_symbols,
            ratio=args.ratio,
            num_workers=args.num_workers,
            bpe_file=bpe_file,
            merge_check_strategy=args.utf8_merge_check_strategy,
            merge_log=args.merge_log
        )
    elif args.variant == 'window':
        learn_bpe_moving_window(
            args.input, args.output, args.dev, args.symbols,
            args.window_size, args.alpha, args.min_frequency, args.verbose,
            num_global=args.global_merges,
            is_dict=args.dict_input,
            total_symbols=args.total_symbols,
            ratio=args.ratio,
            num_workers=args.num_workers,
            bpe_file=bpe_file,
            merge_check_strategy=args.utf8_merge_check_strategy,
            merge_log=args.merge_log
        )
    else:
        raise ValueError("Unknown BPE variant: {0}. Use 'base' or 'window'.".format(args.variant))

    # close files
    for f in args.input:
        if f.name != '<stdin>':
            f.close()
    if args.output.name != '<stdout>':
        args.output.close()
