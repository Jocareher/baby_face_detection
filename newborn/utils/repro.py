from __future__ import annotations
import subprocess, sys, json, datetime, getpass, socket
from pathlib import Path
from typing import Optional
import torch


def get_git_repo_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """
    Returns the root directory of the current Git repository by searching for a '.git' folder,
    starting from `start_path` (or the current working directory if not provided) and moving up
    the directory tree.

    Parameters
    ----------
    start_path : Optional[Path]
        The directory to start searching from. If None, uses the current working directory.

    Returns
    -------
    Optional[Path]
        The path to the Git repository root, or None if not found.
    """
    start_path = Path.cwd() if start_path is None else Path(start_path).resolve()
    for parent in [start_path, *start_path.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def run_shell_command(command: list[str], cwd: Optional[Path] = None) -> str:
    """
    Executes a shell command and returns its standard output as a string.
    Returns an empty string if the command fails.

    Parameters
    ----------
    command : list[str]
        The command and its arguments to execute.
    cwd : Optional[Path]
        The working directory to execute the command in.

    Returns
    -------
    str
        The standard output of the command, or an empty string if execution fails.
    """
    try:
        return (
            subprocess.check_output(command, stderr=subprocess.DEVNULL, cwd=cwd)
            .decode()
            .strip()
        )
    except Exception:
        return ""


def save_reproducibility_metadata(
    output_dir: Path,
    parsed_args: dict,
    include_git_diff: bool = True,
    include_pip_freeze: bool = True,
) -> Path:
    """
    Creates a 'run_info.txt' file in the specified directory containing metadata
    for reproducibility, including environment details, Git status, and optionally
    the output of 'git diff' and 'pip freeze'.

    Parameters
    ----------
    output_dir : Path
        Directory where the metadata file will be saved.
    parsed_args : dict
        Dictionary of parsed command-line arguments (e.g., from vars(parse_args())).
    include_git_diff : bool, default=False
        If True, appends the output of 'git diff' to the file.
    include_pip_freeze : bool, default=True
        If True, appends the output of 'pip freeze' to the file.

    Returns
    -------
    Path
        The path to the generated metadata file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_dir / "metadata.txt"

    # --- Gather Git repository information ---
    repo_root = get_git_repo_root()
    in_git_repo = repo_root is not None
    git_commit = git_branch = "N/A"
    git_dirty = False
    git_diff_content = ""

    if in_git_repo:
        git_commit = (
            run_shell_command(["git", "rev-parse", "HEAD"], cwd=repo_root)[:10] or "N/A"
        )
        git_branch = (
            run_shell_command(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root
            )
            or "N/A"
        )
        git_dirty = bool(
            run_shell_command(["git", "status", "--porcelain"], cwd=repo_root)
        )
        if include_git_diff and git_commit != "N/A":
            git_diff_content = run_shell_command(["git", "diff"], cwd=repo_root)

    # --- Collect environment and run metadata ---
    metadata = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
        "working_dir": str(Path.cwd()),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "python_ver": sys.version.replace("\n", " "),
        "torch_ver": torch.__version__,
        "cuda_ver": torch.version.cuda or "cpu",
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
    }

    # --- Write metadata to file ---
    with metadata_file.open("w", encoding="utf-8") as f:
        f.write(
            "# --- RUN REPRODUCIBILITY METADATA -----------------------------------\n"
        )
        for key, value in metadata.items():
            f.write(f"{key:15}: {value}\n")

        f.write("\n# Parsed arguments\n")
        json.dump(parsed_args, f, indent=2)
        f.write("\n")

        if git_diff_content:
            f.write(
                "\n# --- GIT DIFF -----------------------------------------------------\n"
            )
            f.write(git_diff_content + "\n")

        if include_pip_freeze:
            pip_output = run_shell_command([sys.executable, "-m", "pip", "freeze"])
            f.write(
                "\n# --- PIP FREEZE ---------------------------------------------------\n"
            )
            f.write(pip_output + "\n")

    print(f"[INFO] Saved reproducibility metadata to {metadata_file}")
    return metadata_file
