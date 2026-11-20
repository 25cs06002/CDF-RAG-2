"""Conservative query-only target resolution for G1 candidate retrieval.

No benchmark targets, labels, or reviewer decisions are inputs. Exact mention
spans avoid a broad node shadowing an explicitly named subtype. An explicit
directional predicate may disambiguate an endpoint; otherwise multiple
eligible concepts require abstention, not a guessed medical target.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

COMPARATIVE = re.compile(
    r"\b(?:compar(?:e|es|ed|ing)|versus|vs\.?|different|difference|"
    r"contrast|distinguish)\b", re.I
)
DIRECTIONAL = re.compile(
    r"\b(?:leads?\s+to|leading\s+to|results?\s+in|resulting\s+in|"
    r"causes?\s+|causing\s+|contributes?\s+to|contributing\s+to|"
    r"progress(?:es|ing)?\s+to|development\s+of|risk\s+of|"
    r"risk\s+factors?\s+for|pathways?\s+to|linked\s+to)\b",
    re.I,
)
DIABETES_SUBTYPE = re.compile(r"\btype\s*[12]\s*diabetes\b", re.I)


def _occurrences(question: str, phrase: str):
    if not phrase.strip():
        return
    pattern = re.compile(
        r"(?<![a-z0-9])" + re.escape(phrase.casefold()) + r"(?![a-z0-9])"
    )
    yield from ((m.start(), m.end()) for m in pattern.finditer(question.casefold()))


def mentioned_concepts(
    query: str,
    nodes: Iterable[str],
    aliases: Mapping[str, str] | None = None,
    eligible: Iterable[str] | None = None,
) -> list[tuple[int, int, str]]:
    """Return non-overlapping (start, end, canonical concept) mentions.

    Exact mention, alias approval and evidence eligibility are required; when
    a long mention contains a short one, the longer one wins at that span.
    A separate free-standing short mention is still retained.
    """
    canonical = list(dict.fromkeys(nodes))
    allowed = set(canonical if eligible is None else eligible)
    phrases = [(name, name) for name in canonical if name in allowed]
    phrases += [
        (alias, target) for alias, target in (aliases or {}).items()
        if target in allowed
    ]
    hits = []
    for phrase, target in phrases:
        if target.casefold() == "diabetes" and DIABETES_SUBTYPE.search(query):
            # The generic concept may not stand in for an unapproved subtype.
            continue
        hits.extend((start, end, target) for start, end in _occurrences(query, phrase))
    # Longest mention wins over embedded broad terms, e.g. 'Hypertension'
    # inside 'Severe hypertension'. Preserve independent short mentions.
    hits.sort(key=lambda x: (-(x[1] - x[0]), x[0], x[2]))
    chosen = []
    for hit in hits:
        if all(hit[1] <= old[0] or hit[0] >= old[1] for old in chosen):
            chosen.append(hit)
    return sorted(chosen, key=lambda x: (x[0], x[1], x[2]))


def resolve_target(
    query: str,
    nodes: Iterable[str],
    aliases: Mapping[str, str] | None = None,
    eligible: Iterable[str] | None = None,
) -> tuple[str | None, str]:
    """Choose one explicit evidence-eligible effect or abstain.

    A directional clause disambiguates only when it contains exactly one
    eligible target after its predicate and before a sentence boundary.
    This does not infer clinical causality or silently merge subtypes.
    """
    mentions = mentioned_concepts(query, nodes, aliases, eligible)
    unique = {name for _, _, name in mentions}
    if not unique:
        return None, "no_explicit_reviewed_effect"
    if len(unique) == 1:
        return next(iter(unique)), "unique_approved_effect"
    if COMPARATIVE.search(query):
        return None, "comparative_multiple_effects"
    for cue in reversed(list(DIRECTIONAL.finditer(query))):
        clause_end = min(
            [p for p in (query.find(".", cue.end()), query.find("?", cue.end()),
                         query.find(";", cue.end())) if p >= 0]
            or [len(query)]
        )
        following = {
            name for start, end, name in mentions
            if cue.end() <= start < clause_end
        }
        if len(following) == 1:
            return next(iter(following)), "explicit_directional_endpoint"
        if len(following) > 1:
            return None, "ambiguous_directional_endpoints"
    return None, "ambiguous_multiple_effects"
