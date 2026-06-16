import os
import sys
import time
from collections import defaultdict

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATASET_DIR  = r"./data/dataset"          # <-- change this
EXCLUDE_SUFFIX = "chocomufin.xml"  # <-- files ending with this are skipped
# ─────────────────────────────────────────────────────────────────────────────

CHARS = {
    '\u0328', '&', '\u0335', '\u1d9c', '\ua759',
    '\uf038', '\u0307', '\u0302', ']', 'X', '3', '0',
    '\u0308', '\u1ddd', '\u036b', '\u00f7', '\u1d56', '\ua757', '2', '[',
    ',', '1', '*', '\u0167', '\u205c', '8'
}

COLS   = 80
BAR_W  = 30
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CLR    = "\033[2K\r"


def bar(filled, total, width=BAR_W):
    pct  = filled / total if total else 0.0
    done = int(pct * width)
    return (GREEN + "█" * done + DIM + "░" * (width - done) + RESET,
            f"{pct*100:5.1f}%")


def collect_files(root):
    all_files, skipped = [], []
    for dirpath, _, files in os.walk(root):
        for f in files:
            fpath = os.path.join(dirpath, f)
            if f.endswith(EXCLUDE_SUFFIX):
                skipped.append(fpath)
            else:
                all_files.append(fpath)
    return all_files, skipped


def count_chars_in_file(filepath):
    counts     = defaultdict(int)
    total_chars = 0
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
            content = fh.read()
        total_chars = len(content)
        for ch in content:
            if ch in CHARS:
                counts[ch] += 1
    except Exception as e:
        return counts, total_chars, str(e)
    return counts, total_chars, None


def render_live_table(total_target, total_all, max_count):
    sorted_chars = sorted(CHARS)
    lines = []
    for ch in sorted_chars:
        cnt_t  = total_target.get(ch, 0)
        cnt_a  = total_all.get(ch, 0)
        u      = f"U+{ord(ch):04X}"
        label  = repr(ch).ljust(10)
        b, pct = bar(cnt_t, max_count if max_count else 1)
        hit    = GREEN + "●" + RESET if cnt_t > 0 else DIM + "○" + RESET
        lines.append(
            f"  {hit} {CYAN}{label}{RESET} {DIM}{u}{RESET}  "
            f"{b} {YELLOW}{pct}{RESET}  "
            f"{BOLD}{cnt_t:>8}{RESET}  "
            f"{DIM}all:{cnt_a:>8}{RESET}"
        )
    return lines


def print_final_table(total_target, total_all, xml_total_chars, xml_files_count,
                      all_total_chars, elapsed, errors, skipped_count):

    grand_target = sum(total_target.values())
    grand_all    = sum(total_all.values())

    print(f"\n{BOLD}{GREEN}Done{RESET}  {elapsed:.2f}s  |  "
          f"XML files: {xml_files_count}  |  "
          f"Skipped ({EXCLUDE_SUFFIX}): {skipped_count}")

    if errors:
        print(f"\n{RED}Read errors ({len(errors)}):{RESET}")
        for fp, e in errors[:10]:
            print(f"  {DIM}{fp}{RESET}: {e}")

    # ── XML-only table ────────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}XML files  (excl. {EXCLUDE_SUFFIX}){RESET}")
    print(f"  Total chars in XMLs : {xml_total_chars:,}")
    print(f"  Target chars found  : {grand_target:,}")
    if xml_total_chars:
        print(f"  % of all XML chars  : {grand_target/xml_total_chars*100:.4f}%")
    print(DIM + "─" * 72 + RESET)
    print(f"  {BOLD}{'Char':<12}{'Unicode':<12}{'Count':>10}  {'% of XML chars':>16}  {'% of XML hits':>14}{RESET}")
    print(DIM + "─" * 72 + RESET)
    for ch, cnt in sorted(total_target.items(), key=lambda x: -x[1]):
        u    = f"U+{ord(ch):04X}"
        pxml = cnt / xml_total_chars * 100  if xml_total_chars  else 0.0
        phit = cnt / grand_target    * 100  if grand_target     else 0.0
        print(f"  {CYAN}{repr(ch):<12}{RESET}{DIM}{u:<12}{RESET}"
              f"{BOLD}{cnt:>10}{RESET}  "
              f"{YELLOW}{pxml:>15.4f}%{RESET}  "
              f"{GREEN}{phit:>13.2f}%{RESET}")
    zeros = [ch for ch in CHARS if total_target.get(ch, 0) == 0]
    if zeros:
        print(f"\n  {DIM}Not found in XMLs: {', '.join(repr(c) for c in sorted(zeros))}{RESET}")

    # ── ALL files table ───────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}ALL files (full dataset){RESET}")
    print(f"  Total chars across all files : {all_total_chars:,}")
    print(f"  Target chars found           : {grand_all:,}")
    if all_total_chars:
        print(f"  % of all chars               : {grand_all/all_total_chars*100:.4f}%")
    print(DIM + "─" * 72 + RESET)
    print(f"  {BOLD}{'Char':<12}{'Unicode':<12}{'Count':>10}  {'% of all chars':>16}  {'% of all hits':>14}{RESET}")
    print(DIM + "─" * 72 + RESET)
    for ch, cnt in sorted(total_all.items(), key=lambda x: -x[1]):
        u    = f"U+{ord(ch):04X}"
        pall = cnt / all_total_chars * 100  if all_total_chars else 0.0
        phit = cnt / grand_all       * 100  if grand_all       else 0.0
        print(f"  {CYAN}{repr(ch):<12}{RESET}{DIM}{u:<12}{RESET}"
              f"{BOLD}{cnt:>10}{RESET}  "
              f"{YELLOW}{pall:>15.4f}%{RESET}  "
              f"{GREEN}{phit:>13.2f}%{RESET}")
    zeros_all = [ch for ch in CHARS if total_all.get(ch, 0) == 0]
    if zeros_all:
        print(f"\n  {DIM}Not found anywhere: {', '.join(repr(c) for c in sorted(zeros_all))}{RESET}")


def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"{RED}Directory not found: {DATASET_DIR}{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}{CYAN}Scanning:{RESET} {DATASET_DIR}")
    print(f"{DIM}Excluding files ending with: {EXCLUDE_SUFFIX}{RESET}")
    print(DIM + "─" * COLS + RESET)

    files, skipped = collect_files(DATASET_DIR)
    xml_files      = [f for f in files if f.lower().endswith('.xml')]
    total_files    = len(files)

    print(f"  {BOLD}{total_files}{RESET} files to scan  |  "
          f"{BOLD}{len(xml_files)}{RESET} XML  |  "
          f"{DIM}{len(skipped)} skipped{RESET}\n")

    # accumulators split: xml-only vs all
    total_xml  = defaultdict(int)   # target chars, XML files only
    total_all  = defaultdict(int)   # target chars, every file
    xml_chars  = 0                  # total character count in XMLs
    all_chars  = 0                  # total character count everywhere
    errors     = []
    n_chars    = len(CHARS)

    # ── initial live table ────────────────────────────────────────────────────
    table_lines = render_live_table(total_xml, total_all, 1)
    print("\n".join(table_lines))
    print()   # progress line
    print()   # filename line
    LIVE_BLOCK = n_chars + 2

    t0 = time.time()

    for i, fpath in enumerate(files, 1):
        counts, fchars, err = count_chars_in_file(fpath)
        if err:
            errors.append((fpath, err))

        all_chars += fchars
        for ch, n in counts.items():
            total_all[ch] += n

        if fpath.lower().endswith('.xml'):
            xml_chars += fchars
            for ch, n in counts.items():
                total_xml[ch] += n

        max_count = max(total_xml.values()) if total_xml else 1

        # ── redraw live block ─────────────────────────────────────────────────
        sys.stdout.write(f"\033[{LIVE_BLOCK}A")
        for line in render_live_table(total_xml, total_all, max_count):
            sys.stdout.write(CLR + line + "\n")

        b, pct      = bar(i, total_files, width=40)
        elapsed     = time.time() - t0
        eta         = (elapsed / i * (total_files - i)) if i else 0
        fname_short = os.path.basename(fpath)[:45].ljust(45)
        tag         = f"{CYAN}[xml]{RESET}" if fpath.lower().endswith('.xml') else f"{DIM}[   ]{RESET}"

        sys.stdout.write(CLR + f"  {b} {pct}  {BOLD}{i}/{total_files}{RESET}  ETA {eta:5.1f}s\n")
        sys.stdout.write(CLR + f"  {tag} {DIM}{fname_short}{RESET}\n")
        sys.stdout.flush()

    elapsed = time.time() - t0
    print_final_table(total_xml, total_all, xml_chars, len(xml_files),
                      all_chars, elapsed, errors, len(skipped))


if __name__ == "__main__":
    main()
