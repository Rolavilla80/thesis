#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Rico Sennrich, Negar Foroutan
# UTF-8 case-blocking experiments: Rolando Villasenor (2026)

"""Parity-aware BPE with optional UTF-8 case-blocking filters.

This is a cleaned-up version of `parity_aware_learn_bpe.py` that retains only
the strategies actually used in the current experiments:

  - none                 : no admissibility filter (baseline)
  - block3, block4, block5,
    block34, block35,
    block45, block345    : true ablation -- block ONLY the listed cases by
                           (left_state, right_state) signature.

Cases (by left/right UTF-8 state):
  3: F+P (complete + prefix)   -- composite recoverable prefix
  4: S+F (suffix  + complete)  -- produces suffix-starting token
  5: S+P (suffix  + prefix)    -- produces suffix-starting token

Removed from this version (kept in `parity_aware_learn_bpe.py`):
  - rule1, max, rescan, char_first strategies and their helpers
  - implicit UTF-8 well-formedness fallback (Implementation A)

Optional `--merge-log <path>` writes a CSV with one row per accepted merge plus
aggregated rows for blocked candidate pairs (status = blocked_caseN).
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

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

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


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

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
        help="Parity-aware BPE variant, either 'base' or 'window'. Default: %(default)s")
    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=[sys.stdin], nargs='*',
        metavar='PATHS',
        help="Input texts (default: standard input).")
    parser.add_argument(
        '--dev', '-d', type=argparse.FileType('r'), nargs='*',
        metavar='PATHS',
        help="Development texts (used for parity computation).")
    parser.add_argument(
        '--ratio', '-r', type=float, nargs='*',
        help="Desired compression ratio per input language (alternative to --dev).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file for BPE codes (default: standard output)")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Number of new symbols (merges) to learn. Default: %(default)s")
    parser.add_argument(
        '--global-merges', '-g', type=int, default=0,
        help="For first INT merge operations, use global statistics. Default: %(default)s")
    parser.add_argument(
        '--min-frequency', type=int, default=2, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ. Default: %(default)s')
    parser.add_argument(
        '--preload', type=argparse.FileType('r'), default=None,
        metavar='PATH',
        help="Preload merges from a BPE file (default: None).")
    parser.add_argument(
        '--pretokenize', type=str, default=['whitespace', 'bytelevel'], nargs='*',
        choices=['whitespace', 'bytelevel'],
        help="Huggingface pre-tokenizer(s) to apply. Default: %(default)s")
    parser.add_argument(
        '--dict-input', action="store_true",
        help="If set, input file is interpreted as a dictionary file (word count pairs).")
    parser.add_argument(
        '--total-symbols', '-t', action="store_true",
        help="Subtract number of characters from --symbols.")
    parser.add_argument(
        '--window-size', '-w', type=int, default=100,
        help="Window size for the moving-window variant. Default: %(default)s")
    parser.add_argument(
        '--alpha', type=int, default=2,
        help="Alpha for the moving-window variant. Default: %(default)s")
    parser.add_argument(
        '--num-workers', type=int, default=1,
        help="Number of worker processes (Python 3 only). Default: %(default)s")
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="Verbose mode.")

    parser.add_argument(
        '--utf8-merge-check-strategy', type=str, default='none',
        choices=['none', 'block3', 'block4', 'block5',
                 'block34', 'block35', 'block45', 'block345'],
        help="Case-blocking strategy. Each strategy blocks ONLY the cases it lists "
             "(true independent ablation, no utf8 fallback). "
             "Cases: 3=F+P, 4=S+F, 5=S+P. Default: %(default)s")
    parser.add_argument(
        '--merge-log', type=str, default=None,
        help="Optional CSV path for logging accepted merges and aggregated blocked candidates."
    )

    return parser


# ---------------------------------------------------------------------------
# Vocabulary loading (unchanged from original)
# ---------------------------------------------------------------------------

def get_vocabulary(fobj, is_dict=False, num_workers=1):
    """Reads text and returns a Counter mapping words to frequencies."""
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
    """Apply list of BPE merge operations to each item in vocab."""

    new_vocab = Counter()

    for orig in vocab:

        if len(orig) == 1:
            new_vocab[orig] = vocab[orig]

        word = list(orig[:-1]) + [orig[-1]]

        while len(word) > 1:
            pairs = [(bpe_codes[pair], i, pair) for (i, pair) in enumerate(zip(word, word[1:])) if pair in bpe_codes]

            if not pairs:
                break

            bigram = min(pairs)[2]
            positions = [i for (rank, i, pair) in pairs if pair == bigram]

            i = 0
            new_word = []
            bigram = ''.join(bigram)
            for j in positions:
                if j < i:
                    continue
                new_word.extend(word[i:j])
                new_word.append(bigram)
                i = j + 2
            new_word.extend(word[i:])
            word = new_word

        word = tuple(word)
        new_vocab[word] = vocab[orig]

    return new_vocab


# ---------------------------------------------------------------------------
# UTF-8 byte-level helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_bytelevel_char_to_byte():
    """Inverse GPT-2 byte-level mapping used by Hugging Face ByteLevel."""
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
    """True iff raw is a proper prefix of exactly one UTF-8 character."""
    n = len(raw)
    if n == 0 or n > 3:
        return False

    b0 = raw[0]

    if 0x00 <= b0 <= 0x7F:
        return False

    if n == 1:
        return (0xC2 <= b0 <= 0xDF) or (0xE0 <= b0 <= 0xEF) or (0xF0 <= b0 <= 0xF4)

    b1 = raw[1]

    def cont(x):
        return 0x80 <= x <= 0xBF

    if n == 2:
        if b0 == 0xE0:
            return 0xA0 <= b1 <= 0xBF
        if 0xE1 <= b0 <= 0xEC or 0xEE <= b0 <= 0xEF:
            return cont(b1)
        if b0 == 0xED:
            return 0x80 <= b1 <= 0x9F

        if b0 == 0xF0:
            return 0x90 <= b1 <= 0xBF
        if 0xF1 <= b0 <= 0xF3:
            return cont(b1)
        if b0 == 0xF4:
            return 0x80 <= b1 <= 0x8F

        return False

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
    """Classify a byte-level symbol as 'complete', 'prefix', or 'suffix'."""
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
        return "prefix" if is_valid_utf8_char_prefix(raw[i:]) else "suffix"

    return "complete"


# ---------------------------------------------------------------------------
# Case-blocking admissibility (Implementation B: true independent ablation)
# ---------------------------------------------------------------------------

# Cases:  3 = F+P,  4 = S+F,  5 = S+P
STRATEGY_BLOCKED_CASES = {
    'block3':   frozenset({3}),
    'block4':   frozenset({4}),
    'block5':   frozenset({5}),
    'block34':  frozenset({3, 4}),
    'block35':  frozenset({3, 5}),
    'block45':  frozenset({4, 5}),
    'block345': frozenset({3, 4, 5}),
}

LOCAL_STRATEGIES = frozenset(STRATEGY_BLOCKED_CASES.keys())


def get_block_reason(pair, strategy):
    """Single source of truth for why a pair is rejected.

    Returns 'case3'/'case4'/'case5' if the strategy blocks the pair's case,
    else None. Each case-blocking strategy blocks ONLY the listed cases --
    there is no UTF-8 well-formedness fallback, so pairs whose merged symbol
    would be a suffix-starting token are admitted unless their case is in the
    blocked set. This is the true independent ablation.
    """
    if strategy not in STRATEGY_BLOCKED_CASES:
        return None

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


# ---------------------------------------------------------------------------
# Merge logging (CSV: accepted rows + aggregated blocked rows)
# ---------------------------------------------------------------------------

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
        "merged_valid": merged_state == "complete",
        "category": category,
        "left_hex": symbol_to_hex(pair[0]),
        "right_hex": symbol_to_hex(pair[1]),
        "merged_hex": symbol_to_hex(''.join(pair)),
    }


MERGE_LOG_HEADER = [
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
    'frequency_vector',
]


def open_merge_log(path):
    if not path:
        return None, None
    f = open(path, 'w', encoding='utf-8', newline='')
    writer = csv.writer(f)
    writer.writerow(MERGE_LOG_HEADER)
    return f, writer


def write_accepted_merge(writer, step, max_index, pair, pair_stats):
    """Write one CSV row for an accepted merge at `step`."""
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
        ';'.join(map(str, pair_stats.tolist())),
    ])


def buffer_blocked_merge(buffer, pair, freq, blocked_reason):
    """Accumulate a blocked-pair event into the per-step aggregation buffer.

    Buffer key is (left, right, reason); value is {'count': N, 'freq': vec}.
    The buffer is flushed at the end of each step, producing one row per
    unique (pair, reason) instead of one row per occurrence.
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
    """Write one aggregated row per unique (pair, reason) at `step`, then clear."""
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


# ---------------------------------------------------------------------------
# Pair-statistics maintenance
# ---------------------------------------------------------------------------

def update_pair_statistics(pair, changed, stats, indices,
                           merge_check_strategy='none', blocked_buffer=None):
    """Update pair statistics after a merge has been applied.

    `merge_check_strategy` controls which new pairs may enter the candidate
    pool. `blocked_buffer` (if provided) accumulates rejected events for the
    current step; the caller flushes it after this returns.
    """
    stats[pair] = 0
    indices[pair] = defaultdict(int)
    first, second = pair
    new_pair = first + second

    use_strategy = merge_check_strategy in LOCAL_STRATEGIES

    for j, word, old_word, freq in changed:

        # decrement the counts for pairs overlapping the now-merged occurrence
        i = 0
        while True:
            try:
                i = old_word.index(first, i)
            except ValueError:
                break
            if i < len(old_word) - 1 and old_word[i+1] == second:
                if i:
                    prev = old_word[i-1:i+1]
                    stats[prev] -= freq
                    indices[prev][j] -= 1
                if i < len(old_word) - 2:
                    if old_word[i+2] != first or i >= len(old_word) - 3 or old_word[i+3] != second:
                        nex = old_word[i+1:i+3]
                        stats[nex] -= freq
                        indices[nex][j] -= 1
                i += 2
            else:
                i += 1

        # increment the counts for pairs overlapping the new merged token
        i = 0
        while True:
            try:
                i = word.index(new_pair, i)
            except ValueError:
                break
            if i:
                prev = word[i-1:i+1]
                reason = get_block_reason(prev, merge_check_strategy) if use_strategy else None
                if reason is None:
                    stats[prev] += freq
                    indices[prev][j] += 1
                else:
                    buffer_blocked_merge(blocked_buffer, prev, freq, reason)
            if i < len(word) - 1 and word[i+1] != new_pair:
                nex = word[i:i+2]
                reason = get_block_reason(nex, merge_check_strategy) if use_strategy else None
                if reason is None:
                    stats[nex] += freq
                    indices[nex][j] += 1
                else:
                    buffer_blocked_merge(blocked_buffer, nex, freq, reason)
            i += 1


def get_pair_statistics(vocab, merge_check_strategy='none', blocked_buffer=None):
    """Count frequencies of all adjacent symbol pairs in `vocab` and create an index.

    `merge_check_strategy` filters pairs at insertion (true ablation: only the
    listed cases are blocked). `blocked_buffer` (if provided) accumulates the
    initial-build blocked events with step=0.
    """

    stats = defaultdict(lambda: numpy.zeros(len(vocab[0][1]), dtype=int))
    indices = defaultdict(lambda: defaultdict(int))

    use_strategy = merge_check_strategy in LOCAL_STRATEGIES

    for i, (word, freq) in enumerate(vocab):
        prev_char = word[0]
        for char in word[1:]:
            pair = (prev_char, char)
            reason = get_block_reason(pair, merge_check_strategy) if use_strategy else None
            if reason is None:
                stats[pair] += freq
                indices[pair][i] += 1
            else:
                buffer_blocked_merge(blocked_buffer, pair, freq, reason)
            prev_char = char

    return stats, indices


def replace_pair(pair, vocab, indices):
    """Replace all occurrences of `pair` in `vocab` with the merged symbol."""
    split_char = ' '
    first, second = pair

    pair_str = ''.join(pair)
    pair_str = pair_str.replace('\\', '\\\\')
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
    """Prune low-frequency pairs from stats; big_stats keeps the full counts."""
    for item, freq in list(stats.items()):
        if full_sync or numpy.all(freq < threshold):
            del stats[item]
            if numpy.any(freq < 0):
                big_stats[item] += freq
            else:
                big_stats[item] = freq


def replace_pair_dict(pair, vocab):
    """Replace `pair` in dev vocab and return the per-language length change."""
    length_change = None
    split_char = ' '
    first, second = pair

    pair_str = ''.join(pair)
    pair_str = pair_str.replace('\\', '\\\\')
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

            length_change += (len(word) - len(new_word)) * freq

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


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_input_data(infiles, devfiles, is_dict=False, total_symbols=False,
                          num_global=0, num_workers=1, bpe_file=None,
                          merge_check_strategy='none', blocked_buffer=None):
    """Read input files, build per-language vocabularies, and seed pair stats."""

    if not bpe_file is None:
        line = bpe_file.readline()
        offset = 1
        if not line.startswith('version'):
            bpe_file.seek(0)
            offset = 0

        bpe_codes = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(bpe_file.read().rstrip('\n').split('\n'))]

        for i, item in enumerate(bpe_codes):
            if len(item) != 2:
                sys.stderr.write('Error: invalid line {0} in BPE codes file: {1}\n'.format(i+offset, ' '.join(item)))
                sys.stderr.write('The line should consist of exactly two subword units, separated by whitespace\n')
                sys.exit(1)

        bpe_codes = dict([(code, i) for (i, code) in reversed(list(enumerate(bpe_codes)))])
    else:
        bpe_codes = None

    vocabs = []
    joint_keys = set()
    for f in infiles:
        vocab = get_vocabulary(f, is_dict, num_workers)
        if not bpe_codes is None:
            vocab = pre_merge(vocab, bpe_codes)
        vocab = dict([(tuple(x,), y) for (x, y) in vocab.items()])
        vocabs.append(vocab)
        joint_keys = joint_keys.union(vocab.keys())

    dev_vocabs = []
    dev_keys = set()
    if devfiles:
        for f in devfiles:
            vocab = get_vocabulary(f, is_dict, num_workers)
            if not bpe_codes is None:
                vocab = pre_merge(vocab, bpe_codes)
            vocab = dict([(tuple(x,), y) for (x, y) in vocab.items()])

            dev_vocabs.append(vocab)
            dev_keys = dev_keys.union(vocab.keys())
    array_length = len(vocabs)
    if num_global:
        array_length += 1

    vocab = defaultdict(lambda: numpy.zeros(array_length, dtype=int))
    for i in range(len(vocabs)):
        for key in joint_keys:
            vocab[key][i] = vocabs[i].get(key, 0)

    if num_global:
        for key in joint_keys:
            vocab[key][-1] = sum(vocab[key])

    dev_vocab = defaultdict(lambda: numpy.zeros(len(dev_vocabs), dtype=int))

    if dev_vocabs:
        for i in range(len(dev_vocabs)):
            for key in dev_keys:
                dev_vocab[key][i] = dev_vocabs[i].get(key, 0)

    sorted_vocab = sorted(vocab.items(), key=lambda x: sum(x[1]), reverse=True)
    stats, indices = get_pair_statistics(sorted_vocab,
                                         merge_check_strategy=merge_check_strategy,
                                         blocked_buffer=blocked_buffer)
    big_stats = copy.deepcopy(stats)

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

    threshold = numpy.zeros(array_length, dtype=int)
    for l in range(array_length):
        threshold[l] = stats[max(stats, key=lambda x: (stats[x][l], x))][l] / 10

    if dev_vocab:
        lengths = functools.reduce(numpy.add, [len(key) * value for key, value in dev_vocab.items()])
    else:
        lengths = None
    return (dev_vocab, sorted_vocab, stats, indices, big_stats, threshold, lengths, array_length)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def learn_bpe(infiles, outfile, devfiles, num_symbols, min_frequency=2, verbose=False,
              is_dict=False, total_symbols=False, num_global=0, ratio=None,
              num_workers=1, bpe_file=None,
              merge_check_strategy='none', merge_log=None):
    """Learn `num_symbols` merges with parity-aware BPE (base variant)."""

    logger.info("Learning parity-aware BPE with the following parameters:"
        "\n  num_symbols: {0}, min_frequency: {1}, verbose: {2}, is_dict: {3}, total_symbols: {4}, "
        "num_global: {5}, num_workers: {6}, utf8_merge_check_strategy: {7}".format(
            num_symbols, min_frequency, verbose, is_dict, total_symbols,
            num_global, num_workers, merge_check_strategy))

    outfile.write('#version: 0.2\n')

    merge_log_file, merge_log_writer = open_merge_log(merge_log)
    blocked_buffer = {} if merge_log_writer is not None else None

    dev_vocab, sorted_vocab, stats, indices, big_stats, threshold, lengths, array_length = \
        preprocess_input_data(infiles, devfiles, is_dict, total_symbols, num_global, num_workers,
                              bpe_file,
                              merge_check_strategy=merge_check_strategy,
                              blocked_buffer=blocked_buffer)

    # flush step-0 initial-build blocks before the merge loop
    flush_blocked_buffer(merge_log_writer, 0, blocked_buffer)

    if not ratio is None:
        initial_lengths = functools.reduce(numpy.add, [len(key) * value for key, value in sorted_vocab])
        lengths = numpy.copy(initial_lengths)

    for i in tqdm(range(num_symbols), desc="parity-aware BPE..."):
        if stats:
            if i < num_global:
                if verbose:
                    sys.stderr.write('lengths {0}: picking best subword based on concatenation\n'.format(lengths))
                max_index = -1
            else:
                if not ratio is None:
                    compression_rates = initial_lengths / lengths
                    adjusted_compression_rates = compression_rates / ratio
                    max_index, max_value = min(enumerate(adjusted_compression_rates), key=operator.itemgetter(1))
                    if verbose:
                        sys.stderr.write('initial lengths  {0}\nlengths {1}\n'.format(initial_lengths, lengths))
                        sys.stderr.write('compression rates {0}\nadjusted compression rates {1}: picking best subword in corpus {2} \n'.format(compression_rates, adjusted_compression_rates, max_index))
                else:
                    max_index, max_value = max(enumerate(lengths), key=operator.itemgetter(1))
                    if verbose:
                        sys.stderr.write('lengths {0}: picking best subword in corpus {1} \n'.format(lengths, max_index))

            most_frequent = max(stats, key=lambda x: (stats[x][max_index], x))

        if not stats or (i and stats[most_frequent][max_index] < threshold[max_index]):
            prune_stats(stats, big_stats, threshold, full_sync=True)
            stats = copy.deepcopy(big_stats)
            most_frequent = max(stats, key=lambda x: (stats[x][max_index], x))

            for l in range(array_length):
                threshold[l] = stats[max(stats, key=lambda x: (stats[x][l], x))][l] * i / (i + 10000.0)

            prune_stats(stats, big_stats, threshold)

        if stats[most_frequent][max_index] < min_frequency:
            sys.stderr.write(f'no pair has frequency >= {min_frequency}. Stopping for language {max_index} with length: {lengths}\n')
            break

        if verbose:
            sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, most_frequent[0], most_frequent[1], stats[most_frequent]))

        outfile.write('{0} {1}\n'.format(*most_frequent))

        write_accepted_merge(merge_log_writer, i, max_index, most_frequent, stats[most_frequent])

        changes = replace_pair(most_frequent, sorted_vocab, indices)

        if not ratio is None:
            length_change = functools.reduce(numpy.add, [(len(c[2]) - len(c[1])) * c[3] for c in changes])
            lengths -= length_change
        else:
            length_change = replace_pair_dict(most_frequent, dev_vocab)
            lengths -= length_change

        update_pair_statistics(most_frequent, changes, stats, indices,
                               merge_check_strategy=merge_check_strategy,
                               blocked_buffer=blocked_buffer)
        # flush step-i update-time blocks after the accepted row was written above
        flush_blocked_buffer(merge_log_writer, i, blocked_buffer)

        if not i % 100:
            prune_stats(stats, big_stats, threshold)

        stats[most_frequent] = numpy.zeros(array_length, dtype=int)

    if merge_log_file is not None:
        merge_log_file.close()


def select_language_index(lengths, selected_indices, selection_threshold, window_size):
    """Pick a language index using a moving-window fairness rule."""
    final_index = -1
    mask = numpy.ones(len(lengths), dtype=bool)

    while True:
        valid_indices = numpy.where(mask)[0]
        max_index = valid_indices[numpy.argmax(lengths[valid_indices])]
        count = selected_indices.count(max_index)
        ratio = count * 1.0 / window_size
        if ratio <= selection_threshold:
            final_index = max_index
            break
        else:
            mask[max_index] = False

    return final_index


def learn_bpe_moving_window(infiles, outfile, devfiles, num_symbols, window_size=100, alpha=2,
                            min_frequency=2, verbose=False, is_dict=False, total_symbols=False,
                            num_global=0, ratio=None, num_workers=1, bpe_file=None,
                            merge_check_strategy='none', merge_log=None):
    """Learn `num_symbols` merges with parity-aware BPE (moving-window variant)."""

    logger.info("Using Parity-aware BPE (moving-window variant) with window size {0} and alpha {1}".format(window_size, alpha))
    logger.info("Learning parity-aware BPE with the following parameters:"
        "\n  num_symbols: {0}, min_frequency: {1}, verbose: {2}, is_dict: {3}, total_symbols: {4}, "
        "num_global: {5}, num_workers: {6}, utf8_merge_check_strategy: {7}".format(
            num_symbols, min_frequency, verbose, is_dict, total_symbols,
            num_global, num_workers, merge_check_strategy))

    if bpe_file:
        outfile.write(bpe_file.read())
        bpe_file.seek(0)
    else:
        outfile.write('#version: 0.2\n')

    merge_log_file, merge_log_writer = open_merge_log(merge_log)
    blocked_buffer = {} if merge_log_writer is not None else None

    dev_vocab, sorted_vocab, stats, indices, big_stats, threshold, lengths, array_length = \
        preprocess_input_data(infiles, devfiles, is_dict, total_symbols, num_global, num_workers,
                              bpe_file,
                              merge_check_strategy=merge_check_strategy,
                              blocked_buffer=blocked_buffer)

    flush_blocked_buffer(merge_log_writer, 0, blocked_buffer)

    if not ratio is None:
        initial_lengths = functools.reduce(numpy.add, [len(key) * value for key, value in sorted_vocab])
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
                    compression_rates = initial_lengths / lengths
                    adjusted_compression_rates = compression_rates / ratio
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

            most_frequent = max(stats, key=lambda x: (stats[x][max_index], x))

        if not stats or (i and stats[most_frequent][max_index] < threshold[max_index]):
            prune_stats(stats, big_stats, threshold, full_sync=True)
            stats = copy.deepcopy(big_stats)
            most_frequent = max(stats, key=lambda x: (stats[x][max_index], x))

            for l in range(array_length):
                threshold[l] = stats[max(stats, key=lambda x: (stats[x][l], x))][l] * i / (i + 10000.0)

            prune_stats(stats, big_stats, threshold)

        if stats[most_frequent][max_index] < min_frequency:
            sys.stderr.write(f'no pair has frequency >= {min_frequency}. Stopping for language {max_index} with length: {lengths}\n')
            break

        if verbose:
            sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, most_frequent[0], most_frequent[1], stats[most_frequent]))

        outfile.write('{0} {1}\n'.format(*most_frequent))

        write_accepted_merge(merge_log_writer, i, max_index, most_frequent, stats[most_frequent])

        changes = replace_pair(most_frequent, sorted_vocab, indices)

        if not ratio is None:
            length_change = functools.reduce(numpy.add, [(len(c[2]) - len(c[1])) * c[3] for c in changes])
            lengths -= length_change
        else:
            length_change = replace_pair_dict(most_frequent, dev_vocab)
            lengths -= length_change

        update_pair_statistics(most_frequent, changes, stats, indices,
                               merge_check_strategy=merge_check_strategy,
                               blocked_buffer=blocked_buffer)
        flush_blocked_buffer(merge_log_writer, i, blocked_buffer)

        if not i % 100:
            prune_stats(stats, big_stats, threshold)

        stats[most_frequent] = numpy.zeros(array_length, dtype=int)

    if merge_log_file is not None:
        merge_log_file.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    newdir = os.path.join(currentdir, 'subword_nmt')
    if os.path.isdir(newdir):
        warnings.warn(
            "this script's location has moved to {0}. This symbolic link will be removed in a future version. "
            "Please point to the new location, or install the package and use the command 'subword-nmt'".format(newdir),
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

    if args.dev:
        assert(len(args.input) == len(args.dev))

    if args.ratio:
        assert(args.dev is None)
        assert(len(args.input) == len(args.ratio))
        args.ratio = numpy.array(args.ratio)
        args.ratio = args.ratio / args.ratio[0]

    if args.dev is None and args.ratio is None:
        print("script requires either dev sets or ratios")
        sys.exit(1)

    for i, f in enumerate(args.input):
        if f.name != '<stdin>':
            args.input[i] = codecs.open(f.name, encoding='utf-8')
    if args.dev:
        for i, f in enumerate(args.dev):
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
            merge_log=args.merge_log,
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
            merge_log=args.merge_log,
        )
    else:
        raise ValueError("Unknown BPE variant: {0}. Use 'base' or 'window'.".format(args.variant))

    for f in args.input:
        if f.name != '<stdin>':
            f.close()
    if args.output.name != '<stdout>':
        args.output.close()
