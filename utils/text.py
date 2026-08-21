from __future__ import annotations

import html
import re
import unicodedata


_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
_SPACE_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def light_normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", html.unescape(text))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _CONTROL_RE.sub("", value)
    value = _SPACE_RE.sub(" ", value).strip()
    return value


def approximate_token_len(text: str | None) -> int | None:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return 0
    if _CJK_RE.search(stripped):
        return len(stripped)
    return len(stripped.split())
