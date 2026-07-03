from __future__ import annotations

from typing import Iterable


def print_section(title: str) -> None:
    print("")
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def print_table(headers: list[str], rows: Iterable[Iterable[object]]) -> None:
    rows = [[_format_cell(v) for v in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    print(line(headers))
    print(line(["-" * w for w in widths]))
    for row in rows:
        print(line(row))


def ask_text(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if not value and default is not None:
        return default
    return value


def ask_float(prompt: str, default: float) -> float:
    while True:
        value = ask_text(prompt, str(default))
        try:
            return float(value)
        except ValueError:
            print("Please enter a number.")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{default_text}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def choose_one(prompt: str, options: list[tuple[str, str]], default: str) -> str:
    print(prompt)
    for idx, (key, label) in enumerate(options, start=1):
        marker = " default" if key == default else ""
        print(f"  {idx}. {label} ({key}){marker}")
    while True:
        value = ask_text("Choose one", default)
        if value in {key for key, _ in options}:
            return value
        if value.isdigit():
            idx = int(value)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        print("Unknown option.")


def choose_many(
    prompt: str,
    options: list[tuple[str, str]],
    default: list[str],
) -> list[str]:
    print(prompt)
    for idx, (key, label) in enumerate(options, start=1):
        marker = " default" if key in default else ""
        print(f"  {idx}. {label} ({key}){marker}")
    default_text = ",".join(default)
    while True:
        value = ask_text("Choose comma-separated ids or numbers", default_text)
        selected: list[str] = []
        ok = True
        keys = [key for key, _ in options]
        for token in [t.strip() for t in value.split(",") if t.strip()]:
            if token in keys:
                selected.append(token)
                continue
            if token.isdigit() and 1 <= int(token) <= len(options):
                selected.append(keys[int(token) - 1])
                continue
            ok = False
            print(f"Unknown option: {token}")
            break
        if ok and selected:
            return list(dict.fromkeys(selected))


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)

