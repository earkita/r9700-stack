"""Explicit DeepSeek image pins; interface and numerical checks remain required."""

SUPPORTED_COMMITS = ("18f8f960", "5555a2d8")


def supported_commit(commit):
    return bool(commit) and commit.startswith(SUPPORTED_COMMITS)
