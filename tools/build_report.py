"""REBUILD_REPORT_v4.md: generated from run logs and result files; the final section lists the pytest outcome of the last run (logs/pytest_final.txt)."""
import os, sys, re, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pandas as pd, numpy as np
from mppt.config import RESULTS, ROOT
R = lambda *p: os.path.join(RESULTS, *p)


def main():
    def parse(path):
        out = []
        for ln in open(path).read().splitlines() if os.path.exists(path) else []:
            m = re.match(r'^(\w+)\s+(ok|FAILED.*?)\s+(\d+) s$', ln.strip())
            if m: out.append((m.group(1), m.group(2), int(m.group(3))))
        return out
    steps = parse(os.path.join(ROOT, 'logs', 'run_all_full.log')) + parse(os.path.join(ROOT, 'logs', 'run_all_full2.log'))
    cq = parse(os.path.join(ROOT, 'logs', 'clean_clone_quick', 'run_quick_clean.log')) + parse(os.path.join(ROOT, 'logs', 'clean_clone_quick', 'run_quick_clean2.log'))
    summ = open(R('RESULTS.md')).read().split('## What the data say / do not say')[1].split('## Tables')[0].strip()
    abl = pd.read_csv(R('ablation', 'ablation_runs.csv')); hw = open(R('hpc', 'hardware.json')).read()
    L = ['# REBUILD REPORT v4 (Round 4, reviewer-driven revision)\n',
         'Everything below is generated from the files in this repository by `tools/build_report.py`. Numbers come from `results/*.csv` of the run described here; nothing was tuned on the 2019–2020 test years.\n',
         '## 1. What the data say / do not say\n', summ + '\n', '## 2. What was run and how long it took\n',
         'Single entry point: `python run_all.py` (`--quick` for the small smoke test that writes to `results_quick/`). Steps run sequentially; the timing experiment ran alone on the machine.\n', '| step | outcome | wall-clock (s) |\n|---|---|---|']
    L += [f'| {n} | {o} | {t} |' for n, o, t in steps]
    L += [f'| ablation (run separately, before the sequence above) | ok | training time of all {len(abl)} ablation runs: {abl.fit_seconds.sum():.0f} s (sum of fit seconds) |', '', f'Machine: `{hw.strip()}`\n',
          '### Clean-clone QUICK run (`python run_all.py --quick`, fresh `git clone` of the pushed branch, empty caches)\n', 'QUICK mode is NOT a 5-minute run here: the steps below add up to ' + f'{sum(t for _, o, t in cq if o == "ok") / 60:.0f} minutes (the first pass stopped at the HPC step because of a bug that was then fixed; the steps were continued in a second invocation).\n',
          '| step | outcome | wall-clock (s) |\n|---|---|---|', *[f'| {n} | {o} | {t} |' for n, o, t in cq], '\nThe full run was executed in the working repository as separate invocations of the same steps (logs/run_all_full.log, logs/run_all_full2.log, logs/ablation.log), not as one `python run_all.py` from a fresh clone (about five hours); pytest after the clean-clone QUICK run: see logs/clean_clone_quick/tests_quick.log (14 passed, 1 skipped because the evidence-path test refers to the full run).\n',
          '## 3. What was cut or not done\n',
          '- Hardware-in-the-loop and any bench experiment: not done (not possible from this repository); replaced by a switching-level converter simulation, measured latency, noise and quantisation studies; traceability status "partly".',
          '- Literature comparison (`Table_Literature`) and the critical discussion of the five papers named by Reviewer A: not done, because no paper was read in this study; the table is left unfilled.',
          '- The int8 / float16 latencies are numpy emulation, not microcontroller measurements; the cycle estimate in `quantisation.csv` is labelled as an estimate.',
          '- The full v3 regression (retraining everything) was lost to container restarts; a reduced gate with cached models was used (see CHANGELOG_ROUND4.md).',
          '- Ablation: five seeds for the current architecture, three seeds for the other variants, one run for ridge (deterministic).',
          '- Radar chart of v3 dropped (it stacked several metrics in one figure).',
          '- Manuscript text, language editing, reference lists, and the point-by-point letter are not in this repository.\n', '## 4. Known limitations\n',
          '- Simulation only; one PV-module model family; the scan is assumed to take one control cycle and not to be slew-limited (the converter study shows that 14 settling windows fit in 0.84 s but the low-voltage probes are not fully settled).',
          '- The quasi-static plant is the main result; the converter plant uses an averaged CCM model valid for inductor currents above half the ripple.',
          '- Conformal intervals are valid in distribution only. Seeds: 10 scenario seeds, 5 DNN seeds; Wilcoxon p-values are paired over scenario seeds.\n',
          '## 5. Acceptance checks\n', '- Every manuscript number is in `results/PAPER_NUMBERS.json` with its source CSV and column (test: `test_paper_numbers_equal_csv_values`).',
          '- No figure carries a title; panel letters are enforced when saving (`mppt_style.save_figure`); `docs/FIGURE_MAP.md` lists every figure.', '- Changelog of changed numbers: `CHANGELOG_ROUND4.md` (generated part).\n', '## 6. Final test run (pass/fail list)\n']
    p = os.path.join(ROOT, 'logs', 'tests.log')
    L.append('```\n' + (open(p).read().strip() if os.path.exists(p) else '(pytest has not been run yet)') + '\n```\n')
    open(os.path.join(ROOT, 'REBUILD_REPORT_v4.md'), 'w').write('\n'.join(L))


if __name__ == '__main__':
    main()
