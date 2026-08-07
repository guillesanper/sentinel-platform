"""
Verify that every ``python -m <module>`` in the Kubernetes manifests exists.

This guard exists because of a specific failure. The training CronJob shipped
for a long time invoking ``python -m sentinel.training.cli train --config
/config/training-config.yaml``: a module that has never existed in this
repository, pointed at a config file no code path reads. Nothing caught it,
because nothing ever ran the command a manifest declares — linting checks
Python that is imported, tests check Python that is called, and a container
command is neither until a pod starts.

The check is deliberately narrow. It resolves the module with
:func:`importlib.util.find_spec`, which is the same lookup ``python -m``
performs. Note that resolving a submodule imports its parent packages —
``ml.training.train`` costs the ``ml.training`` import, and therefore mlflow —
but never the target module itself, so nothing in the manifest's command line
actually runs.

Usage:
    kubectl kustomize k8s/base/ | python scripts/check_manifest_commands.py
    python scripts/check_manifest_commands.py rendered.yaml

Exits non-zero and names the offenders if any module cannot be resolved.
"""

import importlib.util
import sys
from pathlib import Path
from typing import NamedTuple

import yaml

# The manifests declare commands that run with the image's WORKDIR (/app) on
# sys.path, which is where `ml/` and `api/` live. Reproduce that here so the
# check gives the same answer whatever directory it is invoked from — without
# it, `python scripts/check_manifest_commands.py` puts `scripts/` on the path
# instead of the repository root and reports every module as missing.
#
# This runs after the imports, not before: nothing imported at module level is
# first-party, so the path only has to be right by the time `check_module`
# resolves anything.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Container specs live under different keys depending on the workload kind, and
# a CronJob buries them three levels deep. Rather than encode every shape, the
# walker below recurses over the whole document looking for the container
# shape itself.
_COMMAND_KEY = "command"
_PYTHON_MODULE_FLAG = "-m"


class Invocation(NamedTuple):
    """A ``python -m <module>`` found in a manifest."""

    kind: str
    name: str
    container: str
    module: str


def _module_invocation(command: list[object]) -> str | None:
    """Return the module a ``command`` list runs with ``-m``, or ``None``."""
    parts = [str(item) for item in command]
    # `python`, `python3` and `/usr/bin/python` all count; anything else is not
    # a Python entry point and is out of scope for this check.
    if not parts or Path(parts[0]).name.split(".")[0] not in ("python", "python3"):
        return None
    try:
        flag_index = parts.index(_PYTHON_MODULE_FLAG)
    except ValueError:
        return None
    if flag_index + 1 >= len(parts):
        return None
    return parts[flag_index + 1]


def _walk(node: object, kind: str, name: str, found: list[Invocation]) -> None:
    """Collect every container command under ``node``."""
    if isinstance(node, dict):
        command = node.get(_COMMAND_KEY)
        if isinstance(command, list) and "name" in node:
            module = _module_invocation(command)
            if module is not None:
                found.append(Invocation(kind, name, str(node["name"]), module))
        for value in node.values():
            _walk(value, kind, name, found)
    elif isinstance(node, list):
        for item in node:
            _walk(item, kind, name, found)


def collect_invocations(documents: list[object]) -> list[Invocation]:
    """Find every ``python -m`` container command across a manifest stream."""
    found: list[Invocation] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        kind = str(doc.get("kind", "?"))
        name = str(doc.get("metadata", {}).get("name", "?"))
        _walk(doc, kind, name, found)
    return found


MISSING = "not found"


def check_module(module: str) -> str | None:
    """
    Return ``None`` if ``python -m module`` would resolve, else the reason.

    The distinction this function draws is the whole point of it. Because
    :func:`~importlib.util.find_spec` imports parent packages, a missing
    *third-party dependency* raises exactly the same exception type as a
    *misspelled manifest*. Collapsing the two would make the check useless in
    both directions: run it in an environment without mlflow installed and
    every command reports as broken.

    So a ``ModuleNotFoundError`` naming the target module or one of its parents
    is the defect being hunted, and anything else is a problem with the
    environment doing the checking, reported as such.
    """
    try:
        return None if importlib.util.find_spec(module) is not None else MISSING
    except ModuleNotFoundError as exc:
        absent = exc.name or ""
        if absent == module or module.startswith(f"{absent}."):
            return MISSING
        return f"needs {absent!r}, which is not installed here"
    except (ImportError, ValueError) as exc:
        # An empty or malformed name raises ValueError; a parent that exists
        # but fails at import time raises ImportError.
        return f"could not be resolved: {exc}"


def main(argv: list[str]) -> int:
    """Read manifests from a file argument or stdin and check them."""
    raw = Path(argv[1]).read_text(encoding="utf-8") if len(argv) > 1 else sys.stdin.read()
    invocations = collect_invocations(list(yaml.safe_load_all(raw)))

    if not invocations:
        print("No `python -m` container commands found in the manifests.")
        # Not an error in itself, but worth surfacing: it usually means the
        # input was empty or the render silently produced nothing.
        return 0

    reasons = {inv: check_module(inv.module) for inv in invocations}
    for inv, reason in reasons.items():
        status = "ok" if reason is None else "FAIL"
        detail = "" if reason is None else f"  <- {reason}"
        print(
            f"[{status:>4}] {inv.kind}/{inv.name} ({inv.container}): python -m {inv.module}{detail}"
        )

    absent = [inv for inv, reason in reasons.items() if reason == MISSING]
    unresolvable = [inv for inv, reason in reasons.items() if reason not in (None, MISSING)]

    if absent:
        print(
            f"\n{len(absent)} manifest command(s) reference a module that does not exist. "
            "A container declaring an unimportable entry point fails at runtime, "
            "on a schedule, where nobody is watching.",
            file=sys.stderr,
        )
        return 1

    if unresolvable:
        print(
            f"\n{len(unresolvable)} command(s) could not be checked because this environment "
            "is missing their dependencies. Install requirements.txt and re-run — this is a "
            "problem with the check, not with the manifests.",
            file=sys.stderr,
        )
        return 2

    print(f"\nAll {len(invocations)} manifest command(s) resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
