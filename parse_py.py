parse_pyimport ast
import re
from typing import Any, Dict, List, Literal, Tuple


Engine = Literal["auto", "cpython3", "ironpython", "ironpython2", "ironpython3"]


class ScriptFormatError(ValueError):
    """Raised when a script cannot be normalized into valid Python."""


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _strip_and_split(text: str, indent_size: int) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.expandtabs(indent_size).rstrip() for line in text.split("\n")]

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    return lines


def _dedent_common_whitespace(lines: List[str]) -> List[str]:
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return lines

    min_indent = min(_leading_spaces(line) for line in non_empty)
    if min_indent == 0:
        return lines

    return [line[min_indent:] if line.strip() else "" for line in lines]


def _normalize_indentation(lines: List[str], indent_size: int) -> List[str]:
    normalized = []
    for line in lines:
        if not line.strip():
            normalized.append("")
            continue

        stripped = line.lstrip(" ")
        spaces = _leading_spaces(line)
        spaces = round(spaces / indent_size) * indent_size
        normalized.append((" " * max(0, spaces)) + stripped)
    return normalized


def detect_ironpython_features(text: str) -> Dict[str, Any]:
    features = {
        "py2_print": bool(re.search(r"(?m)^\s*print\s+[^(\n].*$", text)),
        "py2_except": bool(re.search(r"(?m)^\s*except\s+[^:\n,]+,\s*[A-Za-z_]\w*", text)),
        "xrange": bool(re.search(r"\bxrange\b", text)),
        "raw_input": bool(re.search(r"\braw_input\b", text)),
        "iteritems": ".iteritems(" in text,
        "iterkeys": ".iterkeys(" in text,
        "itervalues": ".itervalues(" in text,
        "unicode_type": bool(re.search(r"\bunicode\b", text)),
        "long_type": bool(re.search(r"\blong\b", text)),
        "basestring_type": bool(re.search(r"\bbasestring\b", text)),
        "old_not_equal": "<>" in text,
        "clr_import": bool(re.search(r"(?m)^\s*import\s+clr\b", text)),
        "clr_add_reference": "clr.AddReference" in text,
        "system_import": bool(re.search(r"(?m)^\s*(from|import)\s+System\b", text)),
    }

    legacy_count = sum(
        int(features[k]) for k in [
            "py2_print", "py2_except", "xrange", "raw_input",
            "iteritems", "iterkeys", "itervalues",
            "unicode_type", "long_type", "basestring_type",
            "old_not_equal",
        ]
    )

    features["likely_ironpython_or_py2"] = (
        legacy_count >= 2
        or features["clr_import"]
        or features["clr_add_reference"]
        or features["system_import"]
    )
    return features


def _resolve_engine(engine: Engine, detection: Dict[str, Any]) -> str:
    if engine != "auto":
        return engine
    return "ironpython2" if detection.get("likely_ironpython_or_py2") else "cpython3"


def _replace_elseif(lines: List[str], fixes: List[str]) -> List[str]:
    updated = []
    changed = False

    for line in lines:
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        if re.match(r"^elseif\b", stripped):
            stripped = re.sub(r"^elseif\b", "elif", stripped, count=1)
            changed = True
        updated.append(indent + stripped)

    if changed:
        fixes.append("Replaced 'elseif' with 'elif'.")
    return updated


def _fix_py2_except(lines: List[str], fixes: List[str]) -> List[str]:
    updated = []
    changed = False
    pattern = re.compile(r"^(\s*)except\s+([^:,\n]+)\s*,\s*([A-Za-z_]\w*)\s*(.*)$")

    for line in lines:
        match = pattern.match(line)
        if match and " as " not in line:
            indent, exc, var, trailing = match.groups()
            line = f"{indent}except {exc} as {var}{trailing}"
            changed = True
        updated.append(line)

    if changed:
        fixes.append("Converted Python 2 except syntax to 'except ... as ...'.")
    return updated


def _convert_py2_print(line: str) -> Tuple[str, bool]:
    stripped = line.lstrip()
    indent = line[:len(line) - len(stripped)]

    if not stripped.startswith("print"):
        return line, False
    if re.match(r"^print\s*\(", stripped):
        return line, False
    if re.match(r"^print\s*$", stripped):
        return indent + "print()", True

    match = re.match(r"^print\s+(.+)$", stripped)
    if not match:
        return line, False

    expr = match.group(1).strip()
    return indent + f"print({expr})", True


def _fix_py2_print(lines: List[str], fixes: List[str]) -> List[str]:
    updated = []
    changed = False

    for line in lines:
        new_line, line_changed = _convert_py2_print(line)
        updated.append(new_line)
        changed = changed or line_changed

    if changed:
        fixes.append("Converted Python 2 print statements to print(...).")
    return updated


def _fix_old_not_equal(lines: List[str], fixes: List[str]) -> List[str]:
    updated = []
    changed = False

    for line in lines:
        if "<>" in line:
            line = line.replace("<>", "!=")
            changed = True
        updated.append(line)

    if changed:
        fixes.append("Replaced old '<>' operator with '!='.")
    return updated


def _replace_word_boundary(
    lines: List[str],
    pattern: str,
    replacement: str,
    message: str,
    fixes: List[str],
) -> List[str]:
    updated = []
    changed = False
    regex = re.compile(pattern)

    for line in lines:
        new_line, count = regex.subn(replacement, line)
        if count:
            changed = True
        updated.append(new_line)

    if changed:
        fixes.append(message)
    return updated


def _fix_iter_methods(lines: List[str], fixes: List[str]) -> List[str]:
    changed = False
    updated = []
    replacements = {
        ".iteritems(": ".items(",
        ".iterkeys(": ".keys(",
        ".itervalues(": ".values(",
    }

    for line in lines:
        new_line = line
        for old, new in replacements.items():
            if old in new_line:
                new_line = new_line.replace(old, new)
                changed = True
        updated.append(new_line)

    if changed:
        fixes.append("Replaced deprecated dict iterator methods with Python 3 equivalents.")
    return updated


def _fix_py2_builtins(lines: List[str], fixes: List[str]) -> List[str]:
    lines = _replace_word_boundary(lines, r"\bxrange\b", "range", "Replaced 'xrange' with 'range'.", fixes)
    lines = _replace_word_boundary(lines, r"\braw_input\b", "input", "Replaced 'raw_input' with 'input'.", fixes)
    lines = _replace_word_boundary(lines, r"\bbasestring\b", "str", "Replaced 'basestring' with 'str'.", fixes)
    lines = _replace_word_boundary(lines, r"\bunicode\b", "str", "Replaced 'unicode' with 'str'.", fixes)
    lines = _replace_word_boundary(lines, r"\blong\b", "int", "Replaced 'long' with 'int'.", fixes)
    return lines


def _needs_colon(stripped: str) -> bool:
    if not stripped or stripped.startswith("#") or stripped.endswith(":"):
        return False

    patterns = [
        r"^def\s+\w+\s*\(.*\)\s*(?:->\s*.+)?$",
        r"^class\s+\w+(\s*\(.*\))?$",
        r"^if\s+.+$",
        r"^elif\s+.+$",
        r"^else$",
        r"^for\s+.+\s+in\s+.+$",
        r"^while\s+.+$",
        r"^try$",
        r"^except(?:\s+.+)?$",
        r"^finally$",
        r"^with\s+.+$",
        r"^match\s+.+$",
        r"^case\s+.+$",
    ]
    return any(re.match(pattern, stripped) for pattern in patterns)


def _fix_missing_colons(lines: List[str], fixes: List[str]) -> List[str]:
    updated = []
    changed = False

    for line in lines:
        if _needs_colon(line.strip()):
            updated.append(line + ":")
            changed = True
        else:
            updated.append(line)

    if changed:
        fixes.append("Added missing colons to block headers.")
    return updated


def _looks_like_block_header(stripped: str) -> bool:
    keywords = {
        "def", "class", "if", "elif", "else", "for", "while",
        "try", "except", "finally", "with", "match", "case"
    }
    if not stripped:
        return False
    head = stripped.split()[0].rstrip(":")
    return head in keywords and stripped.endswith(":")


def _fix_block_indentation(lines: List[str], indent_size: int, fixes: List[str]) -> List[str]:
    updated = lines[:]
    changed = False

    i = 0
    while i < len(updated) - 1:
        current = updated[i]
        stripped = current.strip()

        if stripped and _looks_like_block_header(stripped):
            current_indent = _leading_spaces(current)
            j = i + 1
            while j < len(updated) and not updated[j].strip():
                j += 1

            if j < len(updated):
                next_indent = _leading_spaces(updated[j])
                if next_indent <= current_indent:
                    updated[j] = (" " * (current_indent + indent_size)) + updated[j].lstrip()
                    changed = True
        i += 1

    if changed:
        fixes.append("Indented lines under block headers.")
    return updated


def _balance_delimiters(line: str) -> Tuple[str, bool]:
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = {")", "]", "}"}
    stack = []

    in_single = False
    in_double = False
    escaped = False

    for ch in line:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in closers and stack and ch == stack[-1]:
            stack.pop()

    if stack:
        return line + "".join(reversed(stack)), True
    return line, False


def _fix_unclosed_quotes(line: str) -> Tuple[str, bool]:
    def odd_unescaped_count(s: str, quote: str) -> bool:
        count = 0
        escaped = False
        for ch in s:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == quote:
                count += 1
        return count % 2 == 1

    single_odd = odd_unescaped_count(line, "'")
    double_odd = odd_unescaped_count(line, '"')

    if single_odd and not double_odd:
        return line + "'", True
    if double_odd and not single_odd:
        return line + '"', True
    return line, False


def _fix_line_closures(lines: List[str], fixes: List[str]) -> List[str]:
    updated = []
    quote_changed = False
    delim_changed = False

    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            updated.append(line)
            continue

        line2, changed_quote = _fix_unclosed_quotes(line)
        line3, changed_delim = _balance_delimiters(line2)
        updated.append(line3)

        quote_changed = quote_changed or changed_quote
        delim_changed = delim_changed or changed_delim

    if quote_changed:
        fixes.append("Closed simple unbalanced quotes on individual lines.")
    if delim_changed:
        fixes.append("Closed simple unbalanced parentheses/brackets/braces.")
    return updated


def _validate_script(script: str, resolved_engine: str) -> None:
    try:
        ast.parse(script)
    except SyntaxError as e:
        raise ScriptFormatError(
            f"Script is still not valid Python for engine '{resolved_engine}': "
            f"{e.msg} (line {e.lineno}, offset {e.offset})\n\n"
            f"Resulting script:\n{script}"
        ) from e


def parse_multiline_script_with_report(
    text: str,
    indent_size: int = 4,
    validate_python: bool = True,
    auto_fix: bool = True,
    engine: Engine = "auto",
    max_passes: int = 5,
) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Normalize and auto-fix a multiline script.

    Engines:
        - auto
        - cpython3
        - ironpython
        - ironpython2
        - ironpython3

    Returns:
        cleaned_script, fixes, report
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(indent_size, int) or indent_size <= 0:
        raise ValueError("indent_size must be a positive integer")
    if not isinstance(max_passes, int) or max_passes < 1:
        raise ValueError("max_passes must be at least 1")

    detection = detect_ironpython_features(text)
    resolved_engine = _resolve_engine(engine, detection)
    fixes: List[str] = []

    lines = _strip_and_split(text, indent_size)
    if not lines:
        return "", fixes, {
            "requested_engine": engine,
            "resolved_engine": resolved_engine,
            "detection": detection,
        }

    dedented = _dedent_common_whitespace(lines)
    if dedented != lines:
        fixes.append("Removed common leading indentation.")
    lines = dedented

    normalized = _normalize_indentation(lines, indent_size)
    if normalized != lines:
        fixes.append("Normalized indentation.")
    lines = normalized

    if auto_fix:
        for _ in range(max_passes):
            before = lines[:]

            lines = _replace_elseif(lines, fixes)

            if resolved_engine == "cpython3":
                lines = _fix_py2_print(lines, fixes)
                lines = _fix_py2_except(lines, fixes)
                lines = _fix_old_not_equal(lines, fixes)
                lines = _fix_py2_builtins(lines, fixes)
                lines = _fix_iter_methods(lines, fixes)

            elif resolved_engine == "ironpython3":
                lines = _fix_py2_print(lines, fixes)
                lines = _fix_py2_except(lines, fixes)
                lines = _fix_old_not_equal(lines, fixes)
                lines = _fix_py2_builtins(lines, fixes)
                lines = _fix_iter_methods(lines, fixes)

            elif resolved_engine == "ironpython2":
                # Preserve as much Python 2 style as possible.
                lines = _fix_old_not_equal(lines, fixes)  # still worth fixing
                # Avoid forcing print()/input()/range/items()/str/int rewrites.

            elif resolved_engine == "ironpython":
                # Conservative preservation mode.
                lines = _fix_old_not_equal(lines, fixes)

            lines = _fix_missing_colons(lines, fixes)
            lines = _fix_block_indentation(lines, indent_size, fixes)
            lines = _fix_line_closures(lines, fixes)
            lines = _normalize_indentation(lines, indent_size)

            if lines == before:
                break

    script = "\n".join(lines)

    if validate_python:
        # For ironpython2, Python 2 print syntax will fail in ast.parse under Python 3.
        # So validation is best-effort unless the code has already been modernized.
        if resolved_engine == "ironpython2":
            modernized_preview = script
            modernized_preview_lines = modernized_preview.splitlines()
            temp_fixes: List[str] = []
            modernized_preview_lines = _fix_py2_print(modernized_preview_lines, temp_fixes)
            modernized_preview_lines = _fix_py2_except(modernized_preview_lines, temp_fixes)
            modernized_preview_lines = _fix_py2_builtins(modernized_preview_lines, temp_fixes)
            modernized_preview_lines = _fix_iter_methods(modernized_preview_lines, temp_fixes)
            modernized_preview = "\n".join(modernized_preview_lines)
            _validate_script(modernized_preview, resolved_engine)
        else:
            _validate_script(script, resolved_engine)

    report = {
        "requested_engine": engine,
        "resolved_engine": resolved_engine,
        "detection": detection,
    }
    return script, fixes, report


def parse_multiline_script(
    text: str,
    indent_size: int = 4,
    validate_python: bool = True,
    auto_fix: bool = True,
    engine: Engine = "auto",
    max_passes: int = 5,
) -> str:
    script, _, _ = parse_multiline_script_with_report(
        text=text,
        indent_size=indent_size,
        validate_python=validate_python,
        auto_fix=auto_fix,
        engine=engine,
        max_passes=max_passes,
    )
    return script
