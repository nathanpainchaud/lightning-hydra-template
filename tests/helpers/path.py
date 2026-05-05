import re
from pathlib import Path

import rootutils


def project_src_dir() -> Path:
    """Get the project source directory.

    Returns:
        The path to the project source directory.
    """
    # We use `rootutils.find_root` rather than the env var 'PROJECT_ROOT' as this function is meant to help parametrize
    # tests at collection time, while env vars are not set yet (and hacking pipeline to have env vars defined then is
    # not a trivial workaround).
    project_root = rootutils.find_root(__file__, indicator="pyproject.toml")
    project_dirname = _snake_case(project_root.stem.lower())
    return project_root / "src" / project_dirname


def _snake_case(string: str) -> str:
    """Converts a string to snake case."""
    string = re.sub(r"[\-\.\s]", "_", string)
    return string[0].lower() + re.sub(r"[A-Z]", lambda matched: "_" + matched.group(0).lower(), string[1:])
