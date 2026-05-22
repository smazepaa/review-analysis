import logging
import logging.handlers
import os
import sys
from pathlib import Path

_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"

_LEVEL_COLOURS: dict[int, str] = {
    logging.DEBUG: "\x1b[36m",     # cyan
    logging.INFO: "\x1b[32m",      # green
    logging.WARNING: "\x1b[33m",   # yellow
    logging.ERROR: "\x1b[31m",     # red
    logging.CRITICAL: "\x1b[35m",  # magenta
}


class _ColourFormatter(logging.Formatter):
    """Formatter that adds ANSI colour to the level-name column."""

    _FMT = "{colour}{bold}{level:<8}{reset} {dim}{time}{reset}  {name}  {message}"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = _LEVEL_COLOURS.get(record.levelno, "")
        dim = "\x1b[2m"
        time = self.formatTime(record, "%H:%M:%S")
        level = record.levelname

        parts = record.name.split(".")
        short_name = ".".join(parts[-2:]) if len(parts) >= 2 else record.name
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return self._FMT.format(
            colour=colour,
            bold=_BOLD,
            level=level,
            reset=_RESET,
            dim=dim,
            time=time,
            name=short_name,
            message=msg,
        )


class _PlainFormatter(logging.Formatter):
    _FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    _DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATEFMT)


_configured = False


def setup_logging(log_dir: Path | None = None) -> None:
    global _configured
    if _configured:
        return
    _configured = True

    raw_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, raw_level, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    if sys.stdout.isatty():
        console.setFormatter(_ColourFormatter())
    else:
        console.setFormatter(_PlainFormatter())
    root.addHandler(console)

    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(_PlainFormatter())
    root.addHandler(file_handler)

    logging.getLogger("uvicorn.access").propagate = False
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logging.getLogger(__name__).debug(
        "Logging initialised (level=%s, file=%s)", raw_level, log_file
    )
