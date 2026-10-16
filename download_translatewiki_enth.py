from opustools import OpusGet


"""Download the latest English-Thai translation data from translatewiki.net using OpusGet. This was done because the page was down"""

OpusGet(
    source="en",
    target="th",
    directory="translatewiki",
    release="latest",
    preprocess="moses",
    download_dir="./opus_downloads",
    suppress_prompts=True,
).get_files()