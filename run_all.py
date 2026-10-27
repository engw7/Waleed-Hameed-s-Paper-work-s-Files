"""Single entry point:  python run_all.py            full run (several hours on 4 CPUs)
                       python run_all.py --quick     QUICK mode (small data, few epochs, about 5 minutes; writes to results_quick/)
Steps run strictly one after the other (timing experiments need an otherwise idle machine); each step writes its own log to logs/."""
import os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = [('tables', 'experiments/exp_tables.py', []), ('main', 'experiments/exp_main.py', []), ('hpc', 'experiments/exp_hpc.py', []), ('ablation', 'experiments/exp_ablation.py', []),
         ('stats', 'experiments/exp_stats.py', ['stats', 'noise', 'quant']), ('converter', 'experiments/exp_converter.py', []), ('figures', 'experiments/exp_figures.py', []), ('docs', 'tools/build_docs.py', []),
         ('traceability', 'tools/build_traceability.py', []), ('report', 'tools/build_report.py', []), ('tests', '-m', ['pytest', '-q', '-rA', 'tests']), ('report_final', 'tools/build_report.py', [])]


def main():
    quick = '--quick' in sys.argv; env = dict(os.environ, MPPT_QUICK='1' if quick else '0', PYTHONHASHSEED='0', PYTHONPATH=ROOT)
    os.makedirs(os.path.join(ROOT, 'logs'), exist_ok=True); only = [a for a in sys.argv[1:] if not a.startswith('--')]; summary = []
    for name, script, args in STEPS:
        if only and name not in only: continue
        t0 = time.time(); cmd = [sys.executable, script] + args if script != '-m' else [sys.executable, '-m'] + args
        with open(os.path.join(ROOT, 'logs', f"{name}{'_quick' if quick else ''}.log"), 'w') as lg: rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=lg, stderr=subprocess.STDOUT)
        summary.append((name, rc, time.time() - t0)); print(f'{name:<14} {"ok" if rc == 0 else "FAILED (rc=%d)" % rc:<14} {time.time() - t0:8.0f} s', flush=True)
        if rc != 0 and name not in ('tests', 'report'): print('stopping: see logs/', name); break
    return 0 if all(r == 0 for _, r, _ in summary) else 1


if __name__ == '__main__':
    sys.exit(main())
