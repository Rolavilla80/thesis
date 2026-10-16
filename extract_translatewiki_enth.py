import argparse
import gzip
import os
import random
import re
import xml.etree.ElementTree as ET

_xtargets_re = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*;\s*([0-9]+(?:\.[0-9]+)?)\s*$")

def get_sent(token, pos, sid):
    """
    token: string like '403' or '16.1'
    pos: 1-based list (sentence-by-order)
    sid: dict (sentence-by-id)
    """
    # Prefer exact sentence-id match
    s = sid.get(token)
    if s:
        return s

    # Fallback: integer position
    if token.isdigit():
        i = int(token)
        if 0 < i < len(pos):
            return pos[i] or ""

    return ""

def parse_xtargets(xt):
    """
    xt is like:
      '20;20'
      '4.1 4.2 4.3;4.1 4.2 4.3'
      '1832.1 1832.2;1507.1 1507.2 1507.3'
    Returns (en_ids, th_ids) as lists of strings.
    """
    xt = xt.strip()
    if ";" not in xt:
        return None, None
    left, right = xt.split(";", 1)
    en_ids = [t for t in left.strip().split() if t]
    th_ids = [t for t in right.strip().split() if t]
    if not en_ids or not th_ids:
        return None, None
    return en_ids, th_ids


def resolve_doc_path(root, doc_path):
    doc_path = doc_path.replace("/", os.sep)
    p = os.path.join(root, doc_path)

    # 1) as-is
    if os.path.exists(p):
        return p

    # 2) if align says .gz, try without .gz
    if p.endswith(".gz"):
        p_nogz = p[:-3]
        if os.path.exists(p_nogz):
            return p_nogz

    # 3) if still missing and it ends with .xml, try without extension
    if p.endswith(".xml"):
        p_noext = p[:-4]
        if os.path.exists(p_noext):
            return p_noext

    return None

def read_xml_doc(path):
    """
    Returns:
      pos: 1-based list, pos[i] = i-th <s> in document order ("" allowed)
      sid: dict mapping <s id="..."> -> sentence string
    Supports BOTH:
      - tokenized: <s><w>tok</w>...</s>
      - plain:     <s>some text</s>
    """
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            data = f.read()
        root = ET.fromstring(data)
    else:
        tree = ET.parse(path)
        root = tree.getroot()

    pos = [""]   # 1-based
    sid = {}

    for s in root.findall(".//s"):
        s_id = s.get("id")

        # 1) tokenized case: collect <w> tokens
        words = []
        for w in s.findall(".//w"):
            if w.text:
                tok = w.text.strip()
                if tok:
                    words.append(tok)

        if words:
            sent = " ".join(words).strip()
        else:
            # 2) plain-text case: sentence is directly inside <s> ... </s>
            # itertext() is robust (also works if there are nested tags)
            raw = "".join(s.itertext())
            # normalize whitespace nicely
            sent = " ".join(raw.split()).strip()

        pos.append(sent)
        if s_id is not None:
            sid[s_id] = sent

    return pos, sid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align", required=True, help="Path to en_th.xml (XCES align file)")
    ap.add_argument("--root_en", required=True, help="Root folder that contains en/... files")
    ap.add_argument("--root_th", required=True, help="Root folder that contains th/... files")
    ap.add_argument("--out_dir", default="data_enth")
    ap.add_argument("--max_pairs", type=int, default=205_000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tree = ET.parse(args.align)
    align_root = tree.getroot()

    pairs = []
    missed_docs = 0
    parsed_docs = 0

    for lg in align_root.findall(".//linkGrp"):
        from_doc = lg.get("fromDoc")
        to_doc = lg.get("toDoc")
        if not from_doc or not to_doc:
            continue

        en_path = resolve_doc_path(args.root_en, from_doc)
        th_path = resolve_doc_path(args.root_th, to_doc)

        if en_path is None or th_path is None:
            missed_docs += 1
            if args.debug and missed_docs <= 5:
                print("Missing:", from_doc, "->", os.path.join(args.root_en, from_doc.replace("/", os.sep)))
                print("Missing:", to_doc, "->", os.path.join(args.root_th, to_doc.replace("/", os.sep)))
            continue

        try:
            en_pos, en_sid = read_xml_doc(en_path)
            th_pos, th_sid = read_xml_doc(th_path)

            if args.debug and parsed_docs == 1:
                print("EN sample id=1:", repr(en_sid.get("1", "")[:80]))
                print("TH sample id=1:", repr(th_sid.get("1", "")[:80]))
            parsed_docs += 1
        except Exception as e:
            if args.debug and parsed_docs <= 5:
                print("Parse error:", en_path, th_path, repr(e))
            continue

        for link in lg.findall(".//link"):
            xt = link.get("xtargets")
            if not xt:
                continue
            
            if args.debug and len(pairs) < 1 and random.random() < 0.0005:
                print("xtargets example:", xt)

            en_ids, th_ids = parse_xtargets(xt)
            if en_ids is None:
                continue

            # Fetch sentences by sentence-id map (NOT by position)
            en_parts = [get_sent(t, en_pos, en_sid) for t in en_ids]
            th_parts = [get_sent(t, th_pos, th_sid) for t in th_ids]

            # Join multiple aligned sentences
            en_sent = " ".join([s for s in en_parts if s]).strip()
            th_sent = " ".join([s for s in th_parts if s]).strip()

            if not en_sent or not th_sent:
                continue

            pairs.append((en_sent, th_sent))
            if len(pairs) >= args.max_pairs:
                break

        if len(pairs) >= args.max_pairs:
            break

    print(f"Parsed docs: {parsed_docs} | Missing docs: {missed_docs}")
    print(f"Collected {len(pairs)} aligned sentence pairs (after joining many-to-many).")

    random.seed(args.seed)
    random.shuffle(pairs)

    dev_n = min(2000, len(pairs) // 20)  # dev=2k target, else ~5%
    train_n = min(100000, max(0, len(pairs) - dev_n))  # train=100k target

    dev_pairs = pairs[:dev_n]
    train_pairs = pairs[dev_n:dev_n + train_n]

    train_en = os.path.join(args.out_dir, "train.en")
    train_th = os.path.join(args.out_dir, "train.th")
    dev_en = os.path.join(args.out_dir, "dev.en")
    dev_th = os.path.join(args.out_dir, "dev.th")

    with open(dev_en, "w", encoding="utf-8") as f_en, open(dev_th, "w", encoding="utf-8") as f_th:
        for en, th in dev_pairs:
            f_en.write(en + "\n")
            f_th.write(th + "\n")

    with open(train_en, "w", encoding="utf-8") as f_en, open(train_th, "w", encoding="utf-8") as f_th:
        for en, th in train_pairs:
            f_en.write(en + "\n")
            f_th.write(th + "\n")

    print(f"Wrote dev:   {len(dev_pairs)} pairs")
    print(f"Wrote train: {len(train_pairs)} pairs")
    print(f"Files in: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()