"""Template-based citation span extraction for the BRACIS challenge.

The extractor works only from the document text.  It does not read the
development goldenset and it does not contain document-specific offsets.
Python string indexes are Unicode codepoint indexes, which is the offset
convention required by the challenge.
"""

from __future__ import annotations

import re


Span = tuple[int, int]

_FLAGS = re.IGNORECASE | re.DOTALL
_MAX_CITATION_LENGTH = 180

# Synthetic documents place a triple paragraph break between their preamble
# (where process numbers, OAB numbers and page references are distractors) and
# the argumentative body.  Restricting extraction to the body is the first
# precision gate.
_BODY_DELIMITER = re.compile(r"\n\s*\n\s*\n")

# Each pattern ends immediately before the citation.  The expressions model
# reusable prose templates rather than citation literals or known offsets.
# A few deliberately loose tokens cover the character-substitution noise used
# in level-2 documents (for example, "Confirã-se" and "entcndimento").
_PREFIX_PATTERNS = (
    r"\brepousa\s+(?:n[oa]s?|em)\s+",
    r"\bInvoca-se,\s*ainda,\s*(?:[oa]s?)\s+",
    r"\bComo\s+(?:já\s+)?se\s+"
    r"(?:reconheceu|depreend[ec]|deprcend[ec])\s+"
    r"(?:n[oa]s?|d[oa]s?)\s+",
    r"\bRegistre-se,\s*por\s+oportuno,\s*(?:[oa]s?)\s+",
    r"\bImpõe-se\s+a\s+obs\w+nci[ao]\s+d[oa]s?\s+",
    r"\bNão\s+se\s+pode\s+ignorar\s+(?:[oa]s?)\s+",
    r"\bA\s+tese\s+encontra\s+respaldo\s+n[oa]s?\s+",
    r"\bVale\s+invocar\s+(?:[oa]s?)\s+",
    r"\bReforça\s+o\s+argumento\s+(?:[oa]s?)\s+",
    r"\bCumpre\s+lembrar\s+(?:[oaã]s?)\s+",
    r"\bAmpara[m]?\s+a\s+pretensão\s+(?:[oa]s?)\s+",
    r"\bNesse\s+exato\s+sentido\s+caminha[m]?\s+(?:[oa]s?)\s+",
    r"\bjulgamento\s+d[oa]s?\s+",
    r"\bViolou\s+o\s+acórdão\s+recorrido\s+(?:[oa]s?)\s+",
    r"\bCumpre\s+destacar\s+(?:[oa]s?)\s+",
    r"\bMerece\s+registro\s+(?:[oa]s?)\s+",
    r"\bAo\s+\w*preciar\s+(?:[oa]s?)\s+",
    r"\bIdêntica\s+conclusão\s+foi\s+adotada\s+n[oa]s?\s+",
    r"\bIncide,\s+na\s+espécie,\s+(?:[oa]s?)\s+",
    r"\bMilita\s+em\s+favor\s+d[ao]\s+parte\s+(?:[oa]s?)\s+",
    r"\bConfir\w*-se,\s+a\s+propósito,\s+(?:[oa]s?)\s+",
    r"\bA\s+orientação\s+firmada\s+n[oa]s?\s+",
    r"\bCorrobora\s+essa\s+leitura\s+(?:[oa]s?)\s+",
    r"\bEm\s+caso\s+análogo,\s+(?:[oa]s?)\s+",
    r"\bapoia-se\s+n[oa]s?\s+",
    r"\bRequer-se,\s+com\s+fund\w+\s+n[oa]s?\s+",
    r"\bNos\s+exatos\s+termos\s+d[oa]s?\s+",
    r"\bNão\s+destoa\s+dess\w+\s+ent\w+\s+(?:[oa]s?)\s+",
    r"\bA\s+hipótese\s+atrai\s+a\s+incidência\s+d[oa]s?\s+",
    r"\bdesconsiderou\s+por\s+completo\s+(?:[oa]s?)\s+",
    r"\bdecorre\s+diretamente\s+d[oa]s?\s+",
    r"\bTambém\s+n[oa]s?\s+",
    r"\bdiverge\s+frontalm\w*\s+d[oa]\s+qu\w+\s+"
    r"[aã]ssentado\s+n[oa]s?\s+",
    r"\bAplica-se\s+à\s+espécie,\s*mutatis\s+mutandis,\s*"
    r"o\s+quanto\s+decidido\s+n[oa]s?\s+",
    r"\bA\s+pretensão\s+encontra\s+amparo\s+expresso\s+n[oa]s?\s+",
    r"\bA\s+propósito,\s+veja-se\s+(?:[oa]s?)\s+",
    r"\bDeixou\s+o\s+Tribunal\s+de\s+origem\s+de\s+aplicar\s+"
    r"(?:[oa]s?)\s+",
)

_PREFIXES = tuple(re.compile(pattern, _FLAGS) for pattern in _PREFIX_PATTERNS)

# These continuations start immediately after a citation in the generated
# prose.  Looking for a semantic continuation is more reliable than stopping
# at the first period because identifiers contain periods themselves.
_SUFFIX_PATTERNS = (
    r",\s+no\s+ponto\b",
    r",\s+de\s+clareza\b",
    r",\s+cuja\s+ratio\b",
    r",\s+o\s+colegiado\b",
    r",\s+de\s+cujo\s+voto\b",
    r"\s+foi\s+rejeitad\w+\s+tese\b",
    r",\s+cuja\s+fundamentação\b",
    r",\s+no\s+qual\b",
    r",\s+precedente\b",
    r",\s+que\s+bem\b",
    r",\s+de\s+resto\b",
    r",\s+sob\s+pena\b",
    r",\s+que\s+trata\b",
    r",\s+de\s+aplicação\b",
    r",\s+(?:ao|à|aos|às)\s+qual\b",
    r",\s+a\s+distinção\b",
    r",\s+em\s+hipótese\b",
    r",\s+invocad\w*\b",
    r"\s+foi\s+reafirmada\b",
    r"\s+para\s+sustentar\b",
    r"\s+afastou\s+pretensão\b",
    r",\s+a\s+matéria\b",
    r",\s+embora\b",
    r",\s+a\s+reforma\b",
    r",\s+o\s+pedido\b",
    r",\s+que\s+reconhe\w+\s+a\s+excepcional\w+\b",
)

_SUFFIXES = tuple(re.compile(pattern, _FLAGS) for pattern in _SUFFIX_PATTERNS)

# Periods after these abbreviations are not sentence boundaries.  One-letter
# tokens are handled separately for forms such as A.REsp, R.Esp and H.C.
_ABBREVIATIONS = {
    "a",
    "ag",
    "art",
    "esp",
    "int",
    "min",
    "n",
    "no",
    "nº",
    "n°",
    "re",
    "rec",
    "recl",
    "rel",
}


def body_start(text: str) -> int:
    """Return the first argumentative-body offset, or zero as a fallback."""

    delimiter = _BODY_DELIMITER.search(text)
    return delimiter.end() if delimiter else 0


def _terminal_period(text: str, start: int) -> int | None:
    """Find a sentence-ending period while skipping citation abbreviations."""

    stop = min(len(text), start + _MAX_CITATION_LENGTH)
    for match in re.finditer(r"\.", text[start:stop]):
        position = start + match.start()
        tail = text[position + 1 :]
        whitespace_match = re.match(r"\s*", tail)
        whitespace = whitespace_match.group() if whitespace_match else ""

        # A sentence-ending period is followed by whitespace (or EOF).  This
        # rejects the internal dots in A.REsp, H.C. and decimal-like numbers.
        if tail and not whitespace:
            continue

        next_character = tail[len(whitespace) : len(whitespace) + 1]
        if next_character and not (
            next_character.isupper()
            or next_character in "ÁÉÍÓÚÂÊÔÃÕÇ"
        ):
            continue

        previous = re.search(
            r"([\wº°]+)\s*$", text[max(start, position - 15) : position]
        )
        if previous:
            token = previous.group(1)
            if len(token) == 1 or token.lower() in _ABBREVIATIONS:
                continue

        return position

    return None


def _citation_end(text: str, start: int) -> int | None:
    """Return the nearest credible right boundary after ``start``."""

    stop = min(len(text), start + _MAX_CITATION_LENGTH)
    candidates = []

    for suffix in _SUFFIXES:
        match = suffix.search(text, start, stop)
        if match:
            candidates.append(match.start())

    terminal_period = _terminal_period(text, start)
    if terminal_period is not None:
        candidates.append(terminal_period)

    return min(candidates) if candidates else None


def extract_spans(text: str) -> list[Span]:
    """Extract citation spans from a challenge document.

    Args:
        text: Exact UTF-8 document contents decoded as a Python string.

    Returns:
        Sorted, unique ``(inicio, fim)`` pairs, with ``fim`` exclusive.
    """

    first_body_codepoint = body_start(text)
    starts = sorted(
        {
            match.end()
            for prefix in _PREFIXES
            for match in prefix.finditer(text, first_body_codepoint)
        }
    )

    spans = []
    for start in starts:
        end = _citation_end(text, start)
        if end is not None and end > start:
            spans.append((start, end))

    return spans


__all__ = ["Span", "body_start", "extract_spans"]
