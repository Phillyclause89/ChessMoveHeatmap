"""PPExecutor class that implements processes property for ProcessPoolExecutor"""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process
from typing import Optional, Tuple

__all__ = ['PPExecutor']


class PPExecutor(ProcessPoolExecutor):
    """Implements processes property for ProcessPoolExecutor"""

    @property
    def processes(self) -> Tuple[Optional[Process], ...]:
        r"""Expose the private `_processes` from ProcessPoolExecutor.

        Returns
        -------
        Tuple[Optional[Process], ...]
            A tuple of Process objects currently managed by the executor.
        """
        return tuple(self._processes.values())
