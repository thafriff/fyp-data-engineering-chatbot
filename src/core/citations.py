"""
Readable citations for the chat interface (display only).

The model writes citations as chunk IDs in square brackets, for example [S05-0138]. After an answer is complete, this module
renumbers the citations to [1], [2] ... in order of first appearance and checks each one against the chunks the model was
given. It never changes what the model receives: the prompt, retrieval and generation are untouched.

What counts as a citation follows the notebook 04 judge runner (evaluation/notebook04/run_judging.py, check_citations; plan
section 12, row 2), so the prototype and the evaluation agree:
- a bracket holding something like S06-0205 is a citation attempt; markdown links are skipped;
- a list such as [S06-0205, S06-0206] is checked ID by ID (separated by commas, semicolons or "and");
- a well-formed ID counts anywhere, including inside code, where a model may put it in a comment;
- malformed attempts (e.g. [S060322]) are looked for outside code only, because code is full of ordinary brackets.
The ID pattern itself is shared (settings.NB04_CITATION_ID_PATTERN). The regular expressions below are copies of the judge
runner's, because the app must not import from evaluation/ and the runner is frozen; a test checks that both give the same counts.
"""

import re
from dataclasses import dataclass

from config import settings

_CITATION_ID = re.compile(rf"^{settings.NB04_CITATION_ID_PATTERN}$")
_BRACKET = re.compile(r"\[([^\[\]\n]{1,100})\](?!\()")      # the (?!\() skips markdown links
_CITATION_LIKE = re.compile(r"\bS\s*-?\d", re.IGNORECASE)   # S06-0205, S060205, s06 0205 and so on, but not arr[i] or [1, 2]
_TOKEN_SPLIT = re.compile(r"[,;]|\band\b")
_TOKEN_SPLIT_KEEP = re.compile(r"([,;]|\band\b)")            # the same separators, kept so a list can be rebuilt
_FENCE = re.compile(r"```.*?(```|\Z)", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")


@dataclass
class CitationCheck:
    """One answer's citations. cited_ids are the retrieved chunks it cites, in order of first appearance, so chunk
    cited_ids[i] is shown as [i + 1]. outside_set and malformed list each problem once, in order of appearance.
    counts are occurrences, the same three numbers as the judge runner's check_citations."""
    text: str
    cited_ids: list[str]
    outside_set: list[str]
    malformed: list[str]
    counts: dict

    @property
    def cites_course_material(self) -> bool:
        """False when no citation points at a chunk the model was given: the judge runner's "no valid citation"."""
        return bool(self.cited_ids)

    def to_dict(self) -> dict:
        """What the chat history stores, so the citations survive a rerun without parsing again."""
        return {"cited_ids": self.cited_ids, "outside_set": self.outside_set, "malformed": self.malformed, "counts": self.counts}


def _add_once(items: list[str], item: str) -> None:
    """Append an item unless it is already listed (keeps the order of first appearance)."""
    if item not in items:
        items.append(item)


def check(answer: str, retrieved_ids: list[str] | set[str]) -> CitationCheck:
    """Renumber and check the citations in a finished answer. retrieved_ids are the chunks the model was given."""
    retrieved = set(retrieved_ids)
    cited, outside, malformed = [], [], []
    counts = {"valid": 0, "outside_set": 0, "malformed": 0}

    def renumber(match: re.Match) -> str:
        """Replace each cited retrieved chunk ID with its number; anything else in the bracket is left as written."""
        pieces = _TOKEN_SPLIT_KEEP.split(match.group(1))
        changed = False
        for i, piece in enumerate(pieces):
            token = piece.strip()
            if not _CITATION_ID.match(token):
                continue
            if token in retrieved:
                _add_once(cited, token)
                counts["valid"] += 1
                pieces[i] = piece.replace(token, str(cited.index(token) + 1))
                changed = True
            else:
                _add_once(outside, token)
                counts["outside_set"] += 1
        return f"[{''.join(pieces)}]" if changed else match.group(0)

    text = _BRACKET.sub(renumber, answer)

    for content in _BRACKET.findall(_INLINE_CODE.sub("", _FENCE.sub("", answer))):
        if _CITATION_LIKE.search(content):
            for token in (t.strip() for t in _TOKEN_SPLIT.split(content)):
                if token and not _CITATION_ID.match(token):
                    counts["malformed"] += 1
                    _add_once(malformed, token)
    counts["citations"] = counts["valid"] + counts["outside_set"]
    return CitationCheck(text=text, cited_ids=cited, outside_set=outside, malformed=malformed, counts=counts)


def session_counts(checks: list[dict]) -> dict:
    """How often each problem happened across the answers checked in this chat: answers affected, not occurrences."""
    return {"answers_checked": len(checks),
            "outside_set": sum(bool(c["outside_set"]) for c in checks),
            "malformed": sum(bool(c["malformed"]) for c in checks),
            "no_citation": sum(not c["cited_ids"] for c in checks)}
