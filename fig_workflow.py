import os as _os; _ROOT = _os.path.abspath(_os.environ.get("COF_ROOT", "."))  # repository root
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
fig, ax = plt.subplots(figsize=(7.8, 5.2)); ax.set_xlim(0, 100); ax.set_ylim(-2, 72); ax.axis('off')
C = {'data': '#dbe9f6', 'sim': '#fde3c8', 'dft': '#e6dcf0', 'ml': '#d6efd8', 'out': '#f0f0f0'}
def box(x, y, w, h, title, body, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.5', fc=C[kind], ec='#444444', lw=0.8))
    ax.text(x + w / 2, y + h - 2.2, title, ha='center', va='top', weight='bold', fontsize=7.4)
    ax.text(x + w / 2, y + h - 6.0, body, ha='center', va='top', fontsize=6.3, linespacing=1.4)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=9, lw=0.9, color='#333333'))
# row 1
box(2, 54, 28, 16, '1  Database', 'CURATED-COFs: 874 experimentally\nreported frameworks\n(871 with ≤3000 atoms per cell)', 'data')
box(36, 54, 28, 16, '2  Charges and geometry', 'EQeq partial charges\nZeo++: LCD, PLD, ASA (1.65 Å),\naccessible volume (1.3 Å)', 'data')
box(70, 54, 28, 16, '3  Widom insertion (RASPA)', 'CO$_2$ and N$_2$, 298 K, DREIDING/UFF\n+ TraPPE; K$_H$, S, Q$_{st}^0$\nrefinement: 6000 and 30000 cycles', 'sim')
arrow(30.8, 62, 35.2, 62); arrow(64.8, 62, 69.2, 62)
# row 2
box(70, 30, 28, 18, '4  Quality control', '866 completed → remove\nunphysical charges (10),\nLi$^+$/Na$^+$ frameworks (5),\nno accessible surface (12)\n→ 839 frameworks', 'data')
box(36, 30, 28, 18, '5  Descriptors and diversity', '37 geometric and chemical\ndescriptors; standardised PCA;\nk-means (k = 100) → diverse\ntraining set (87 after QC)', 'data')
box(2, 30, 28, 18, '6  Machine learning', 'RF, MLP (descriptors) and\nSchNet (atomic graphs);\n5-fold CV + screening scenario;\nuncertainty, Shapley values', 'ml')
arrow(84, 53.2, 84, 48.8); arrow(69.2, 39, 64.8, 39); arrow(35.2, 39, 30.8, 39)
# row 3
box(2, 5, 28, 17, '7  Ranking and triage', 'ranked list of 839 COFs;\nsimulation saved vs recall\nof the top 5%; ranking\nstability (bootstrap, sensitivity)', 'out')
box(36, 5, 28, 17, '8  DFT on linkage motifs', '5 linkage clusters + benzene;\nPBE-D3(BJ)/def2-TZVP,\ncounterpoise-corrected\nCO$_2$ and N$_2$ binding energies', 'dft')
box(70, 5, 28, 17, '9  Validation vs experiment', 'pure-CO$_2$ GCMC for 9 COFs\n(298 K/55 bar, 273 K/1 bar);\nhelium void fractions', 'sim')
arrow(16, 29.2, 16, 22.8)
arrow(84, 29.2, 84, 22.8)
def darrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=9, lw=0.9, color='#666666', linestyle=(0, (3, 2))))
darrow(35.2, 13, 30.8, 13)
ax.text(33, 15.0, 'interpretation', ha='center', fontsize=5.6, style='italic', color='#555555')
ax.add_patch(FancyArrowPatch((84, 4.3), (16, 4.3), connectionstyle='arc3,rad=0.0', arrowstyle='-', lw=0.9, color='#666666', linestyle=(0, (3, 2))))
darrow(16, 4.3, 16, 4.6)
ax.text(50, 1.6, 'reliability of simulated values (force-field limits)', ha='center', fontsize=5.6, style='italic', color='#555555')
fig.savefig('figs/fig_workflow.png', dpi=600, bbox_inches='tight', facecolor='white')
print('ok')
