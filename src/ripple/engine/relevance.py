"""Noise filter for social text: drop template spam and posts with no financial content.

On the 2018 tweet data, ~95% of company-tagged tweets are spam or chatter ("I posted a new
video to Facebook", memes). Filtering them turns a buried signal into a ~25x volume spike on
the Facebook earnings-crash day.
"""

import re

SPAM_PATTERNS = re.compile(
    r"i posted a new (video|photo)|i added a video to|i liked a @?youtube video|posted a photo"
    r"|check out my|giveaway|follow (me|back)|#?win\b|click here|subscribe",
    re.IGNORECASE,
)

FINANCE_TERMS = re.compile(
    r"\$[A-Z]{1,5}\b|\bstocks?\b|\bshares?\b|earnings|revenue|profit|\bloss(es)?\b|market ?cap|\bmarkets?\b"
    r"|investors?|\binvest|\bCEO\b|billion|million|plunge|plummet|soar|surge|tumble|slump|\bdrop|\bfall|crash|rally"
    r"|tariff|trade war|lawsuit|\bsued?\b|\bfined?\b|\bSEC\b|\bFTC\b|regulat|antitrust|acqui|merger|\bdeal\b"
    r"|launch|unveil|recall|breach|hack|privacy|layoffs?|strike|downgrade|upgrade|analysts?|price target"
    r"|dividend|\bIPO\b|nasdaq|NYSE|\bdow\b|S&P|wall street|quarter|\bQ[1-4]\b|guidance|sales|valuation",
    re.IGNORECASE,
)

URL_OR_MENTION = re.compile(r"https?://\S+|@\w+")


def is_spam(text: str) -> bool:
    content = URL_OR_MENTION.sub("", text).strip()
    return len(content.split()) < 4 or bool(SPAM_PATTERNS.search(text))


def is_finance_relevant(text: str) -> bool:
    return bool(FINANCE_TERMS.search(text))


def keep_social(text: str) -> bool:
    return not is_spam(text) and is_finance_relevant(text)
