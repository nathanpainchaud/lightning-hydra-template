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


def collect_config_group_options(group: str, glob_expr: str = "[!_]*.yaml") -> list[str]:
    """Dynamically collect options for a config group.

    Args:
        group: The config group to collect options for, separated by slashes if nested (e.g., "data/dataset").
        glob_expr: The glob expression to use to select config options. The default expression matches options under
            config group (non-recursively) that do not start with an underscore (i.e., convention for abstract configs).

    Returns:
        A list of options collected for the config group.
    """
    return [
        option_path.stem
        # Sort matches, to ensure consistent and predictable ordering across config groups
        for option_path in sorted((project_src_dir() / "configs" / group).glob(glob_expr))
    ]


def _snake_case(string: str) -> str:
    """Converts a string to snake case."""
    string = re.sub(r"[\-\.\s]", "_", string)
    return string[0].lower() + re.sub(r"[A-Z]", lambda matched: "_" + matched.group(0).lower(), string[1:])
