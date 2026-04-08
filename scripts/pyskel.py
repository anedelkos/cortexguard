#!/usr/bin/env python3
"""
pyskel — print Python code skeletons (signatures + docstrings, no bodies).

Usage:
    python scripts/pyskel.py src/cortexguard/edge/orchestrator.py
    python scripts/pyskel.py src/cortexguard/edge/
    python scripts/pyskel.py src/
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def _unparse_annotation(node: ast.expr | None) -> str:
    if node is None:
        return ""
    return ast.unparse(node)


def _format_args(args: ast.arguments) -> str:
    parts: list[str] = []

    # positional-only args (before /)
    for _i, arg in enumerate(args.posonlyargs):
        ann = f": {_unparse_annotation(arg.annotation)}" if arg.annotation else ""
        parts.append(f"{arg.arg}{ann}")
    if args.posonlyargs:
        parts.append("/")

    # regular args
    n_defaults = len(args.defaults)
    n_args = len(args.args)
    for i, arg in enumerate(args.args):
        ann = f": {_unparse_annotation(arg.annotation)}" if arg.annotation else ""
        default_offset = i - (n_args - n_defaults)
        if default_offset >= 0:
            default = f" = {ast.unparse(args.defaults[default_offset])}"
        else:
            default = ""
        parts.append(f"{arg.arg}{ann}{default}")

    # *args
    if args.vararg:
        ann = f": {_unparse_annotation(args.vararg.annotation)}" if args.vararg.annotation else ""
        parts.append(f"*{args.vararg.arg}{ann}")
    elif args.kwonlyargs:
        parts.append("*")

    # keyword-only args
    for i, arg in enumerate(args.kwonlyargs):
        ann = f": {_unparse_annotation(arg.annotation)}" if arg.annotation else ""
        default = ""
        if args.kw_defaults[i] is not None:
            default = f" = {ast.unparse(args.kw_defaults[i])}"  # type: ignore[arg-type]
        parts.append(f"{arg.arg}{ann}{default}")

    # **kwargs
    if args.kwarg:
        ann = f": {_unparse_annotation(args.kwarg.annotation)}" if args.kwarg.annotation else ""
        parts.append(f"**{args.kwarg.arg}{ann}")

    return ", ".join(parts)


def _docstring(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        first_line = node.body[0].value.value.strip().splitlines()[0]
        return first_line
    return None


def _format_func(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    indent: str = "    ",
) -> list[str]:
    lines: list[str] = []

    for dec in node.decorator_list:
        lines.append(f"{indent}@{ast.unparse(dec)}")

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = _format_args(node.args)
    ret = f" -> {_unparse_annotation(node.returns)}" if node.returns else ""
    lines.append(f"{indent}{prefix} {node.name}({args}){ret}: ...")

    doc = _docstring(node)
    if doc:
        lines.append(f'{indent}    """{doc}"""')

    return lines


def skeleton(source: str, path: str) -> str:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:
        return f"# {path}: SyntaxError: {e}\n"

    out: list[str] = [f"# {path}", ""]

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            header = f"class {node.name}" + (f"({bases})" if bases else "") + ":"
            out.append(header)
            doc = _docstring(node)
            if doc:
                out.append(f'    """{doc}"""')
                out.append("")

            has_members = False
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.extend(_format_func(child, indent="    "))
                    has_members = True

            if not has_members:
                out.append("    ...")
            out.append("")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.extend(_format_func(node, indent=""))
            doc = _docstring(node)
            if doc:
                out.append(f'    """{doc}"""')
            out.append("")

    return "\n".join(out)


def process_path(p: Path) -> None:
    if p.is_file() and p.suffix == ".py":
        files = [p]
    elif p.is_dir():
        files = sorted(
            f for f in p.rglob("*.py") if "__pycache__" not in f.parts and ".venv" not in f.parts
        )
    else:
        print(f"# skipped: {p}", file=sys.stderr)
        return

    for f in files:
        try:
            source = f.read_text(encoding="utf-8")
        except OSError as e:
            print(f"# {f}: {e}", file=sys.stderr)
            continue
        print(skeleton(source, str(f)))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        process_path(Path(arg))


if __name__ == "__main__":
    main()
