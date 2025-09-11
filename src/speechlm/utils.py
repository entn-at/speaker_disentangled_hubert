import re

from transformers.models.whisper.english_normalizer import ADDITIONAL_DIACRITICS

vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'\",.?! ;:()[]—_" + "".join(ADDITIONAL_DIACRITICS)
pattern = f"[^{re.escape(vocab)}]"


def normalize_text(s: str) -> str:
    s = s.replace("‘", "'")
    s = s.replace("’", "'")
    tokens = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',.?")
    s_list = [x if x in tokens else ADDITIONAL_DIACRITICS.get(x, " ") for x in s]
    s = " ".join("".join(s_list).split()).strip()

    s = re.sub(r"\baround'em\b", "around them", s)
    s = re.sub(r"\bb'lieve\b", "believe", s)
    s = re.sub(r"\bbewilder'd\b", "bewildered", s)
    s = re.sub(r"\bcap'n\b", "captain", s)
    s = re.sub(r"\bCap'n\b", "Captain", s)
    s = re.sub(r"\bcharm'em\b", "charm them", s)
    s = re.sub(r"\bdiff'rence\b", "difference", s)
    s = re.sub(r"\be'en\b", "even", s)
    s = re.sub(r"\bfetchin'\s", "fetching ", s)
    s = re.sub(r"\bgive'em\b", "give them", s)
    s = re.sub(r"\binv'tation", "invitation", s)
    s = re.sub(r"\bmore'n\b", "more than", s)
    s = re.sub(r"\bof'em\b", "of them", s)
    s = re.sub(r"\bop'ning\b", "opening", s)
    s = re.sub(r"\bpass'd\b", "passed", s)
    s = re.sub(r"\bp'raps\b", "perhaps", s)
    s = re.sub(r"\bshorten'd\b", "shortened", s)
    s = re.sub(r"\bs'pose\b", "suppose", s)
    s = re.sub(r"\btellin'\s", "telling ", s)
    s = re.sub(r"\bvisitin'\s", "visiting ", s)
    s = re.sub(r"\bwith'em\b", "with them", s)

    s = re.sub(r"\s\?", "?", s)
    s = re.sub(r"\s,", ",", s)

    return s
