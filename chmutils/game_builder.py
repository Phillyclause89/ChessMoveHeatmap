"""GBuilder class that overrides `GameBuilder.handle_error` to raise exception."""
from chess.pgn import GameBuilder

__all__ = [
    'GBuilder'
]


class GBuilder(GameBuilder):
    r"""Overrides `GameBuilder.handle_error` to raise exception."""

    def handle_error(self, error: Exception) -> None:
        r"""Override of GameBuilder.handle_error method to raise errors."""
        raise error
