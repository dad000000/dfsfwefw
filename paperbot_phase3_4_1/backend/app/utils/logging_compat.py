from __future__ import annotations

import logging
import sys


class KVLogger(logging.Logger):
    """Allow log.info('msg', key=value) style without crashing std logging."""

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1, **kwargs):
        if kwargs:
            extra = dict(extra or {})
            extra.update(kwargs)
        super()._log(level, msg, args, exc_info=exc_info, extra=extra, stack_info=stack_info, stacklevel=stacklevel)


def setup_logging(level: str = "INFO") -> None:
    logging.setLoggerClass(KVLogger)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

