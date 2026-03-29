import logging
from pathlib import Path

LOGS = Path("data/logs/")
LOGS.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%D-%Hh-%Mm-%Ss",
        filename=LOGS / f"{name}.log",
        filemode="a",
        level=logging.INFO,
        force=True,
    )
