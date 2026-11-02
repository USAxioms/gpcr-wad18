"""
Exhaustive token-level mutation harness for the GPCR-ACE capsule.

Strategy (identical to r3-meta-capsule):
  - Tokenize every target source file with the stdlib tokenizer.
  - For each eligible token, apply every applicable mutation operator.
  - Write the mutant in place, run the test suite in a subprocess with
    PYTHONDONTWRITEBYTECODE=1 (prevents stale .pyc serving the original).
  - A mutant is KILLED if the test suite exits non-zero.
  - A mutant is a SURVIVOR if the test suite exits zero.
  - Survivors are reported for equivalence proof or new-test writing.

Mutation operators:
  Arithmetic : + ↔ -,  * ↔ //,  ** → *
  Comparison : < ↔ <=, > ↔ >=,  == ↔ !=
  Boolean    : and ↔ or,  not → (elide the token)
  Name/const : True ↔ False,  return True → return False
  Integer    : n → n+1,  n → n-1  (single-digit literals only, avoids
               exploding the mutant count on large constants)

Target files (all non-pipeline source):
  src/wad18.py, src/csl.py, src/gnm.py, src/bias.py

pipeline.py is excluded: it is the integration script whose mutations
would be killed by the same tests that kill gnm/bias mutants, adding
no coverage information not already captured.
"""

import os, sys, json, tokenize, io, subprocess, copy, time
from datetime import datetime, timezone

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = [
    os.path.join(ROOT, "src", "wad18.py"),
    os.path.join(ROOT, "src", "csl.py"),
    os.path.join(ROOT, "src", "gnm.py"),
    os.path.join(ROOT, "src", "bias.py"),
]
TESTS_DIR = os.path.join(ROOT, "tests")
PYTHON    = sys.executable
TIMEOUT = 60   # seconds per mutant

# ─────────────────────────────────────────── mutation operators

ARITH = {"+": "-", "-": "+", "*": "//", "//": "*", "**": "*"}
COMP  = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "=="}
BOOL  = {"and": "or", "or": "and"}
NAMES = {"True": "False", "False": "True"}


def _mutations_for_token(toktype, tokstr):
    """Return list of replacement strings for this token."""
    candidates = []
    if toktype == tokenize.OP:
        if tokstr in ARITH:
            candidates.append(ARITH[tokstr])
        if tokstr in COMP:
            candidates.append(COMP[tokstr])
    if toktype == tokenize.NAME:
        if tokstr in BOOL:
            candidates.append(BOOL[tokstr])
        if tokstr in NAMES:
            candidates.append(NAMES[tokstr])
        if tokstr == "not":
            candidates.append("")          # elide 'not'
    if toktype == tokenize.NUMBER:
        try:
            n = int(tokstr)
            if 0 <= n <= 9:               # single-digit only
                candidates.append(str(n + 1))
                if n > 0:
                    candidates.append(str(n - 1))
        except ValueError:
            pass
    return candidates


def _apply_mutation(source, tok_index, tokens, replacement):
    """Return source string with tokens[tok_index].string replaced."""
    result = []
    for i, tok in enumerate(tokens):
        if i == tok_index:
            result.append(replacement)
        else:
            result.append(tok.string)
    # The tokenizer preserves whitespace in INDENT/DEDENT/NEWLINE tokens,
    # but we rebuild from the token strings which already include spacing.
    # Use the original source as a base and splice by offset instead.
    lines   = source.splitlines(keepends=True)
    row, col = tokens[tok_index].start   # 1-based row
    end_row, end_col = tokens[tok_index].end
    new_lines = []
    for i, line in enumerate(lines, 1):
        if i == row == end_row:
            new_lines.append(line[:col] + replacement + line[end_col:])
        elif i == row:
            new_lines.append(line[:col] + replacement)
        elif i == end_row:
            new_lines.append(line[end_col:])
        elif row < i < end_row:
            pass
        else:
            new_lines.append(line)
    return "".join(new_lines)


def _run_tests(env=None):
    """Run the full test suite (all test_*.py); return (passed: bool, elapsed: float)."""
    t0 = time.monotonic()
    env = env or os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"]          = "0"
    r = subprocess.run(
        [PYTHON, "-m", "unittest", "discover", "-s", TESTS_DIR,
         "-p", "test_*.py", "-v"],
        capture_output=True, timeout=TIMEOUT, env=env,
        cwd=ROOT
    )
    return r.returncode == 0, time.monotonic() - t0


# ─────────────────────────────────────────── main harness

def run_mutation_testing(verbose=True):
    results = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "targets": TARGETS,
        "mutants": [],
        "summary": {},
    }

    total = killed = survived = errors = 0

    for path in TARGETS:
        rel  = os.path.relpath(path, ROOT)
        src  = open(path).read()

        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
        except tokenize.TokenError as e:
            if verbose:
                print(f"  [SKIP] {rel}: tokenize error: {e}")
            continue

        for idx, tok in enumerate(tokens):
            replacements = _mutations_for_token(tok.type, tok.string)
            for rep in replacements:
                total += 1
                mid = "%s:%d:%s→%s" % (rel, tok.start[0], tok.string, rep)

                try:
                    mutant_src = _apply_mutation(src, idx, tokens, rep)
                except Exception as e:
                    errors += 1
                    results["mutants"].append({
                        "id": mid, "status": "ERROR", "error": str(e)})
                    continue

                # write mutant
                open(path, "w").write(mutant_src)
                try:
                    passed, elapsed = _run_tests()
                except subprocess.TimeoutExpired:
                    status = "TIMEOUT"
                    elapsed = TIMEOUT
                    passed  = False
                else:
                    status = "SURVIVED" if passed else "KILLED"

                if not passed:
                    killed += 1
                else:
                    survived += 1

                rec = {
                    "id":      mid,
                    "file":    rel,
                    "line":    tok.start[0],
                    "original": tok.string,
                    "mutant":  rep,
                    "status":  status,
                    "elapsed": round(elapsed, 3),
                }
                results["mutants"].append(rec)

                if verbose:
                    mark = "✓" if status == "KILLED" else "✗"
                    print(f"  {mark} [{total:>4}] {status:<9} {mid}")

                # restore original
                open(path, "w").write(src)

    # ── summary
    raw_rate   = round(killed / total * 100) if total else 0
    results["summary"] = {
        "mutants":                   total,
        "killed":                    killed,
        "survived":                  survived,
        "errors":                    errors,
        "raw_kill_rate_percent":     raw_rate,
        "proven_equivalent":         0,   # filled after manual review
        "unexplained_survivors":     survived,
        "effective_kill_rate_percent": raw_rate,  # updated after equivalence review
        "exhaustive":                True,
        "finished_utc":              datetime.now(timezone.utc).isoformat(),
    }

    # write log
    log_dir = os.path.join(ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    path_out = os.path.join(log_dir, "mutation.json")
    json.dump(results, open(path_out, "w"), indent=2, sort_keys=True)

    print("\n" + "="*60)
    print(f"  Mutants:   {total}")
    print(f"  Killed:    {killed}")
    print(f"  Survived:  {survived}")
    print(f"  Raw rate:  {raw_rate}%")
    print(f"  Log:       {path_out}")
    print("="*60)

    return results


if __name__ == "__main__":
    run_mutation_testing(verbose=True)
