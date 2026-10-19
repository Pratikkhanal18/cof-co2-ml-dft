import os as _os; _ROOT = _os.path.abspath(_os.environ.get('COF_ROOT', '.'))  # repository root
import re, glob, os, pandas as pd, numpy as np
R = 8.314462618e-3  # kJ/mol/K
def parse(f):
    s = open(f).read()
    out = {'id': os.path.basename(f).replace('.data', '')}
    for gas in ('CO2', 'N2'):
        m = re.search(r'\[%s\] Average Henry coefficient:\s+([-\d.eE+]+) \+/- ([-\d.eE+]+)' % gas, s)
        u = re.search(r'\[%s\] Average  <U_gh>_1-<U_h>_0:.*?\(\s+([-\d.eE+]+) \+/-\s+([-\d.eE+]+) kJ/mol\)' % gas, s)
        if not m or not u: return None
        out[f'KH_{gas}'] = float(m.group(1)); out[f'KH_{gas}_err'] = float(m.group(2))
        out[f'dU_{gas}'] = float(u.group(1)); out[f'dU_{gas}_err'] = float(u.group(2))
    T = float(re.search(r'External temperature:\s+([\d.]+)', s).group(1)) if re.search(r'External temperature:\s+([\d.]+)', s) else 298.0
    out['T'] = T
    for gas in ('CO2', 'N2'):
        out[f'Qst_{gas}'] = -(out[f'dU_{gas}']) + R * T  # isosteric heat at infinite dilution, kJ/mol (positive)
    out['S_henry'] = out['KH_CO2'] / out['KH_N2']
    out['S_henry_err'] = out['S_henry'] * np.sqrt((out['KH_CO2_err'] / out['KH_CO2']) ** 2 + (out['KH_N2_err'] / out['KH_N2']) ** 2)
    out['log10_S'] = np.log10(out['S_henry'])
    return out
if __name__ == '__main__':
    rows = [r for f in sorted(glob.glob(f'{_ROOT}/raspa/widom/done/*.data')) if (r := parse(f))]
    df = pd.DataFrame(rows); df.to_csv(f'{_ROOT}/widom_results.csv', index=False)
    print(len(df)); print(df.describe().T[['mean', 'min', '50%', 'max']])
