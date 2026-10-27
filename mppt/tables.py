"""Table export: every table is written as results/tables/<name>.csv, .md and .tex from one DataFrame (no number is typed by hand)."""
import os
import numpy as np, pandas as pd
from .config import RESULTS


def _fmt(v, digits):
    if isinstance(v, (bool, np.bool_)): return 'yes' if v else 'no'
    if isinstance(v, (int, np.integer)): return f'{v:d}'
    if isinstance(v, (float, np.floating)):
        if np.isnan(v): return '-'
        return f'{v:.{digits}g}' if abs(v) >= 1e5 or (v != 0 and abs(v) < 10 ** -digits) else f'{v:.{digits}f}'.rstrip('0').rstrip('.') if digits > 0 else f'{v:.0f}'
    return '' if v is None else str(v)


def _tex(s): return str(s).replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#').replace('~', r'\~{}')


def save_table(name, df, caption='', digits=3):
    d = os.path.join(RESULTS, 'tables'); os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, name + '.csv'), index=False)
    cells = [[_fmt(v, digits) for v in row] for row in df.itertuples(index=False)]
    with open(os.path.join(d, name + '.md'), 'w') as f:
        if caption: f.write(f'**{name}.** {caption}\n\n')
        f.write('| ' + ' | '.join(map(str, df.columns)) + ' |\n|' + '---|' * len(df.columns) + '\n')
        for r in cells: f.write('| ' + ' | '.join(c.replace('|', '\\|') for c in r) + ' |\n')
    with open(os.path.join(d, name + '.tex'), 'w') as f:
        f.write('\\begin{table}[t]\n\\centering\n' + (f'\\caption{{{_tex(caption)}}}\n' if caption else '') + f'\\label{{tab:{name}}}\n\\small\n\\begin{{tabular}}{{{"l" * len(df.columns)}}}\n\\hline\n')
        f.write(' & '.join(_tex(c) for c in df.columns) + ' \\\\\n\\hline\n')
        for r in cells: f.write(' & '.join(_tex(c) for c in r) + ' \\\\\n')
        f.write('\\hline\n\\end{tabular}\n\\end{table}\n')
