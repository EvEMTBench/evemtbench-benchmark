# lib_leak_scan.sh — the ONE forbidden-string scan core, sourced by make_repro_pack.sh and
# make_public_release.sh (drive it via `<script> --scan-only <dir>` in tests, so the tests always
# exercise the shipped patterns, never a copy).
#
# Design rules:
# - FAIL CLOSED: a grep hard error (unreadable file, missing dir — exit 2) counts as a hit, so an
#   errored scan can never be reported clean.
# - SELF-SAFE PATTERNS: every pattern is spelled so its own definition does not match it (e.g. a
#   one-character class around the last letter of an account name). That lets the defining scripts
#   and this library ship in the public export with NO scan exclusions.
# - Occurrence-level allowlisting: allowlists are applied to the matched substring (grep -o), never
#   by dropping whole lines — a forbidden string co-located with an allowlisted one still refuses.

# Shared forbidden patterns: absolute home/cluster paths, key/token material (incl. JWT-shaped
# blobs such as Zenodo preview tokens), and the HPC account names.
LEAK_COMMON_PATTERNS=(
  '/home/[a-z]'
  # absolute only: a relative path segment such as tests/cluster/ is repo layout, not a leak
  '(^|[^A-Za-z0-9_.-])/cluster[/]'
  'BEGIN [A-Z ]*PRIVATE KEY'
  'ssh-(rsa|ed25519) '
  'ghp_[A-Za-z0-9]{20,}'
  'glpat-[A-Za-z0-9_-]{10,}'
  'token=eyJ[A-Za-z0-9_-]{10,}'
  'eyJ[A-Za-z0-9_-]{10,}\.eyJ'
  'b304[d]c'
  'iwi5217[h]'
)

# leak_scan_core <dir> <pattern>... — print hits; return 0 on hits OR error (a problem), 1 if clean.
# -I skips binary files (compressed bytes false-positive the token patterns; binary fixtures are
# verified clean separately via `strings`).
leak_scan_core() {
  local dir="$1"; shift
  local args=() p out status
  for p in "$@"; do args+=(-e "$p"); done
  out="$(grep -rInE -I "${args[@]}" -- "$dir" 2>&1)"; status=$?
  case $status in
    1) return 1 ;;                        # clean
    0) printf '%s\n' "$out"; return 0 ;;  # hits
    *)
      printf 'leak_scan: grep failed (exit %s) — FAIL CLOSED, treating as a hit.\n%s\n' \
        "$status" "$out"
      return 0
      ;;
  esac
}

# leak_scan_occurrences <dir> <pattern> [allow-regex] — like leak_scan_core, but matches are
# extracted per OCCURRENCE (grep -o) and the optional allowlist regex is applied to each matched
# substring, so an allowlisted URL can never mask another hit on the same line.
leak_scan_occurrences() {
  local dir="$1" pattern="$2" allow="${3:-}"
  local out status
  out="$(grep -rInoE -I -e "$pattern" -- "$dir" 2>&1)"; status=$?
  case $status in
    1) return 1 ;;
    0)
      if [ -n "$allow" ]; then
        out="$(printf '%s\n' "$out" | grep -vE "$allow")" || return 1
      fi
      printf '%s\n' "$out"
      return 0
      ;;
    *)
      printf 'leak_scan: grep failed (exit %s) — FAIL CLOSED, treating as a hit.\n%s\n' \
        "$status" "$out"
      return 0
      ;;
  esac
}
