from __future__ import annotations

import ast
import io
import keyword
import re
import tokenize
import unicodedata
from dataclasses import dataclass

_CODE_FENCE_RE = re.compile(r"^\s*```(?:python|py)?\s*|\s*```\s*$", re.IGNORECASE)
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


@dataclass(frozen=True)
class CodePreprocessingOptions:
    remove_docstrings: bool = True
    remove_comments: bool = True
    preserve_newlines: bool = True
    split_identifiers: bool = True
    max_tokens: int = 180


def normalize_code_text(code: str) -> str:
    """Normalize Unicode/newlines without deleting code semantics."""
    text = unicodedata.normalize("NFKC", str(code or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CODE_FENCE_RE.sub("", text).strip()
    return text


def _docstring_line_ranges(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not body or not isinstance(body, list):
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            start = getattr(first, "lineno", 0)
            end = getattr(first, "end_lineno", start)
            lines.update(range(start, end + 1))
    return lines


def strip_python_docstrings(code: str) -> str:
    """Remove module/class/function docstrings while preserving the remaining lines."""
    text = normalize_code_text(code)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    remove = _docstring_line_ranges(tree)
    return "\n".join(line for i, line in enumerate(text.splitlines(), start=1) if i not in remove)


def strip_python_comments(code: str) -> str:
    """Remove COMMENT tokens without treating # characters inside strings as comments."""
    text = normalize_code_text(code)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        kept = [tok for tok in tokens if tok.type != tokenize.COMMENT]
        return tokenize.untokenize(kept)
    except (tokenize.TokenError, IndentationError):
        return text


def split_identifier(identifier: str) -> list[str]:
    pieces: list[str] = []
    for snake_piece in re.split(r"_+", identifier):
        pieces.extend(_CAMEL_RE.sub(" ", snake_piece).split())
    return [piece.lower() for piece in pieces if piece]


def lexical_code_tokens(code: str, *, split_identifiers: bool = True) -> list[str]:
    """Tokenize Python while preserving keywords, operators, names, strings, and numbers."""
    text = normalize_code_text(code)
    result: list[str] = []
    try:
        stream = tokenize.generate_tokens(io.StringIO(text).readline)
        for tok in stream:
            if tok.type in {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NL}:
                continue
            if tok.type == tokenize.NEWLINE:
                result.append("<NL>")
            elif tok.type == tokenize.INDENT:
                result.append("<INDENT>")
            elif tok.type == tokenize.DEDENT:
                result.append("<DEDENT>")
            elif tok.type == tokenize.STRING:
                result.append("<STR>")
            elif tok.type == tokenize.NUMBER:
                result.append("<NUM>")
            elif tok.type == tokenize.NAME:
                original = tok.string.lower()
                result.append(original)
                if split_identifiers and not keyword.iskeyword(original):
                    subtokens = split_identifier(tok.string)
                    if len(subtokens) > 1:
                        result.extend(f"<SUB>{part}" for part in subtokens)
            elif tok.type == tokenize.COMMENT:
                continue
            else:
                token_text = tok.string.strip()
                if token_text:
                    result.append(token_text)
    except (tokenize.TokenError, IndentationError):
        result = re.findall(r"[A-Za-z_]\w*|==|!=|<=|>=|//|\*\*|[-+*/%<>=()\[\]{},.:]", text)
    return result


def preprocess_code(
    code: str,
    options: CodePreprocessingOptions | None = None,
    *,
    mode: str = "semantic",
) -> str:
    """
    Prepare source code for tokenization.

    `legacy` reproduces the supplied notebook's whitespace-only cleanup and must be
    used with the supplied tokenizer/checkpoint. `semantic` removes target leakage
    and preserves code operators through lexical tokens.
    """
    text = normalize_code_text(code)
    if mode == "legacy":
        return re.sub(r"\s+", " ", text).strip()

    opts = options or CodePreprocessingOptions()
    if opts.remove_docstrings:
        text = strip_python_docstrings(text)
    if opts.remove_comments:
        text = strip_python_comments(text)
    tokens = lexical_code_tokens(text, split_identifiers=opts.split_identifiers)
    return " ".join(tokens[: opts.max_tokens]).strip()


def validate_python_code(code: str) -> tuple[bool, str | None]:
    try:
        ast.parse(normalize_code_text(code))
        return True, None
    except SyntaxError as exc:
        return False, f"Line {exc.lineno}: {exc.msg}"
