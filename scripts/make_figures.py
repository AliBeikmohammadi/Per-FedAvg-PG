"""Generate all experiment figures from results/*.npz into figures/ (PDF + PNG).

Run: python make_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)

# Colorblind-safe palette (Wong 2011)
BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = (
    '#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#7f7f7f')
COL = {'exact': BLUE, 'fo': ORANGE, 'fedavg': GREEN}


COLW  = 241.147 / 72.27      # 3.337 in : one column of the two-column body
TEXTW = 506.295 / 72.27      # 7.006 in : single-column appendix

W_MAIN = 0.49 * COLW         # 1.635 in : two panels side by side in one column
W_APPX = 0.62 * TEXTW        # 4.344 in : single panel, single-column appendix
W_WIDE = 0.92 * TEXTW        # 6.445 in : three-panel rollout figure


SERIF = ['Linux Libertine O', 'Libertinus Serif', 'Linux Libertine',
         'STIX Two Text', 'STIXGeneral', 'DejaVu Serif']

BASE = {
    'font.family': 'serif', 'font.serif': SERIF, 'mathtext.fontset': 'stix',
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'figure.dpi': 150, 'savefig.dpi': 600,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.constrained_layout.use': True,
    'figure.constrained_layout.h_pad': 0.008,
    'figure.constrained_layout.w_pad': 0.008,
    'savefig.bbox': None,        
}

def style(pt, compact=False):
    plt.rcParams.update(BASE)
    plt.rcParams.update({
        'font.size': pt, 'axes.labelsize': pt, 'axes.titlesize': pt,
        'legend.fontsize': pt - 0.5,
        'xtick.labelsize': pt - 0.5, 'ytick.labelsize': pt - 0.5,
        'axes.linewidth':   0.5 if compact else 0.8,
        'lines.linewidth':  1.1 if compact else 1.8,
        'lines.markersize': 2.6 if compact else 5.0,
        'xtick.major.width': 0.5 if compact else 0.8,
        'ytick.major.width': 0.5 if compact else 0.8,
        'xtick.major.size':  2.0 if compact else 3.0,
        'ytick.major.size':  2.0 if compact else 3.0,
        'xtick.major.pad':   1.5 if compact else 3.0,
        'ytick.major.pad':   1.5 if compact else 3.0,
        'axes.labelpad':     1.5 if compact else 4.0,
        'legend.handlelength':   1.3 if compact else 2.0,
        'legend.handletextpad':  0.4 if compact else 0.8,
        'legend.labelspacing':   0.22 if compact else 0.5,
        'legend.borderaxespad':  0.15 if compact else 0.5,
    })

def main_style():      style(7.5, compact=True)    
def appendix_style():  style(9.0, compact=False)   

def _save(fig, name):
    for ext in ('.pdf', '.png'):
        fig.savefig(os.path.join(FIG, name + ext))
    plt.close(fig)

_kfmt = FuncFormatter(lambda v, _: '0' if v == 0 else
                      (f'{v/1000:g}k' if v >= 1000 else f'{v:g}'))



def _band(ax, x, Y, color, label, ls='-'):
    m, s = Y.mean(0), Y.std(0)
    ax.plot(x, m, color=color, ls=ls, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0)


def fig_e1():
    main_style()
    d = np.load(os.path.join(RES, 'e1e2.npz'), allow_pickle=True)
    fig, ax = plt.subplots(figsize=(W_MAIN, W_MAIN / 1.28))
    x = d['exact/per_agent_traj']
    _band(ax, x, d['exact/Fpost'], COL['exact'], 'exact')
    ax.plot(x, d['exact/fpre'].mean(0), color=COL['exact'], ls='--', lw=1.0)#label='exact, before adaptation'
    _band(ax, d['fo/per_agent_traj'], d['fo/Fpost'], COL['fo'], 'first-order')
    _band(ax, d['fedavg/per_agent_traj'], d['fedavg/Fpost'], COL['fedavg'], 'FedAvg-PG')
    if 'random_floor' in d.files:
        ax.axhline(float(d['random_floor'][0]), ls=':', color=GRAY, lw=0.9) #label='random policy'
    ax.set_xticks([0, 4000, 8000, 12000])
    ax.set_xlabel('per-agent trajectories')
    ax.set_ylabel('expected return')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0)
    ax.xaxis.set_major_formatter(_kfmt)   # e1
    _save(fig, 'fig_e1_personalization')


def fig_e3():
    main_style()
    d = np.load(os.path.join(RES, 'e3.npz'), allow_pickle=True)
    a = d['alphas']
    fig, ax = plt.subplots(figsize=(W_MAIN, W_MAIN / 1.28))
    for v, lab in [('exact', 'exact'), ('fo', 'first-order'), ('fedavg', 'FedAvg-PG')]:
        ax.errorbar(a, d[f'{v}/F_mean'], yerr=d[f'{v}/F_std'], fmt='o-',
                    color=COL[v], capsize=1.6, elinewidth=0.7, label=lab)
    ax.set_xticks([0, 1, 2, 3]) 
    ax.set_xlabel(r'adaptation step size $\alpha$')
    ax.set_ylabel(r'final return $F$')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0)
    _save(fig, 'fig_e3_alpha')


def fig_e4():
    main_style()
    d = np.load(os.path.join(RES, 'e4.npz'), allow_pickle=True)
    mh = d['m_h']
    fig, ax = plt.subplots(figsize=(W_MAIN, W_MAIN / 1.28))
    rng = np.random.default_rng(0)
    med = []
    for x in mh:
        v = d[f'mh{int(x)}/vals']
        jit = x * (1 + 0.06 * rng.standard_normal(len(v)))     # multiplicative jitter on log-x
        ax.scatter(jit, v, s=4.5, color=BLUE, alpha=0.35, lw=0, zorder=2)
        med.append(np.median(v))
    ax.plot(mh, med, 'D-', color=BLUE, zorder=3, label='exact (median)')
    ax.axhline(d['fedavg_mean'][0], ls='--', color=COL['fedavg'], lw=1.0, label='FedAvg-PG')
    ax.set_xscale('log')
    ax.set_xlabel(r'curvature batch $m_{\mathrm{h}}$')
    ax.set_ylabel('post-adapt. return')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0)
    _save(fig, 'fig_e4_curvature')


def fig_e5():
    main_style()
    d = np.load(os.path.join(RES, 'e5.npz'), allow_pickle=True)
    mt, mc = d['meta_traj'], d['meta_curves']
    st, sc = d['scratch_traj'], d['scratch_curves']
    fig, ax = plt.subplots(figsize=(W_MAIN, W_MAIN / 1.28))
    pos = mt > 0
    mm, ms = mc.mean(0), mc.std(0)
    ax.plot(mt[pos], mm[pos], 'D-', color=BLUE, label='meta-init + 1 step')
    ax.fill_between(mt[pos], (mm - ms)[pos], (mm + ms)[pos], color=BLUE, alpha=0.15, lw=0)
    ax.axhline(mc[:, 0].mean(), ls='--', color=BLUE, lw=1.0, label='meta-init, zero-shot')  
    m, sd = sc.mean(0), sc.std(0)
    ax.plot(st, m, 'o-', color=RED, label='from scratch')
    ax.fill_between(st, m - sd, m + sd, color=RED, alpha=0.15, lw=0)
    ax.axhline(float(d['random_ref'][0]), ls=':', color='#bbbbbb') #label='random init'
    ax.set_xscale('log')
    ax.set_xlabel('adaptation trajectories')
    ax.set_ylabel('post-adapt. return')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0)
    ax.set_ylim(top=2.45)
    _save(fig, 'fig_e5_fewshot')
    j = int(np.argmax(m >= mc[:, 0].mean())) if np.any(m >= mc[:, 0].mean()) else len(m) - 1
    print(f"E5: zero-shot return={mc[:,0].mean():.2f}; from-scratch matches it near {int(st[j])} traj.")


def fig_e5_success():
    appendix_style()
    d = np.load(os.path.join(RES, 'e5.npz'), allow_pickle=True)
    mt, ms = d['meta_traj'], d['meta_success']
    st, ss = d['scratch_traj'], d['scratch_success']
    fig, ax = plt.subplots(figsize=(W_APPX, W_APPX / 1.45))
    pos = mt > 0
    mm, msd = ms.mean(0), ms.std(0)
    ax.plot(mt[pos], mm[pos], 'D-', color=BLUE, label='meta-init + 1 step')
    ax.fill_between(mt[pos], (mm - msd)[pos], (mm + msd)[pos], color=BLUE, alpha=0.15, lw=0)
    ax.axhline(ms[:, 0].mean(), ls='--', color=BLUE, lw=1.4, label='meta-init, zero-shot')
    m, sd = ss.mean(0), ss.std(0)
    ax.plot(st, m, 'o-', color=RED, label='from scratch (independent)')
    ax.fill_between(st, m - sd, m + sd, color=RED, alpha=0.15, lw=0)
    ax.axhline(float(d['random_success'][0]), ls=':', color='#bbbbbb', label='random init')
    ax.set_xscale('log'); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('adaptation trajectories (held-out agent)')
    ax.set_ylabel('success rate (reaches goal)')
    ax.legend(loc='lower right')
    _save(fig, 'fig_e5_success')


def fig_e6_learning():
    appendix_style()
    d = np.load(os.path.join(RES, 'e1e2.npz'), allow_pickle=True)
    fig, ax = plt.subplots(figsize=(W_APPX, W_APPX / 1.45))
    for v, lab in [('exact', 'exact'), ('fo', 'first-order'), ('fedavg', 'FedAvg-PG')]:
        _band(ax, d[f'{v}/rounds'], d[f'{v}/Fpost'], COL[v], lab)
    if 'random_floor' in d.files:
        ax.axhline(float(d['random_floor'][0]), ls=':', color=GRAY, label='random policy')
    ax.set_xlabel('communication round $k$')
    ax.set_ylabel('post-adaptation return $F$')
    ax.legend(loc='lower right')
    _save(fig, 'fig_e6_learning')


def fig_e7_trajectories():
    appendix_style()
    d = np.load(os.path.join(RES, 'e7_traj.npz'), allow_pickle=True)
    goal = d['goal']; gr = float(d['goal_r'])
    panels = [('before', 'meta-init (before adaptation)', ORANGE),
              ('after', 'meta-init + 1 step (200 traj)', BLUE),
              ('scratch', 'from scratch (200 traj)', RED)]
    fig, axes = plt.subplots(1, 3, figsize=(W_WIDE, W_WIDE / 3.0))
    for ax, (key, title, color) in zip(axes, panels):
        P = d[key]
        reached = 0
        for tr in P:
            dist = np.linalg.norm(tr - goal[None, :], axis=1)
            hit = np.where(dist <= gr)[0]
            end = hit[0] + 1 if len(hit) else len(tr)   # draw until first reaching the goal
            if len(hit):
                reached += 1
                ax.plot(tr[:end, 0], tr[:end, 1], color=color, alpha=0.6, lw=1.1)
                ax.scatter([tr[end - 1, 0]], [tr[end - 1, 1]], color=color, s=12, zorder=4)  # reach point
            else:
                ax.plot(tr[:, 0], tr[:, 1], color=color, alpha=0.35, lw=0.9, ls=':')  # never reaches
        ax.scatter(P[:, 0, 0], P[:, 0, 1], c='k', s=10, zorder=3)
        ax.add_patch(plt.Circle((goal[0], goal[1]), gr, color=GREEN, alpha=0.25, lw=0))
        ax.scatter([goal[0]], [goal[1]], marker='*', c=GREEN, s=160, zorder=5,
                   edgecolors='k', linewidths=0.5)
        ax.set_title(f'{title}\n({reached}/{len(P)} rollouts reach the goal)', fontsize=8.5)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
    _save(fig, 'fig_e7_trajectories')

if __name__ == '__main__':
    figs = [('e1', fig_e1), ('e3', fig_e3), ('e4', fig_e4), ('e5_return', fig_e5),
            ('e5_success', fig_e5_success), ('e6_learning', fig_e6_learning),
            ('e7_trajectories', fig_e7_trajectories)]
    for name, fn in figs:
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  skip {name}: missing data ({os.path.basename(str(e).split()[-1]) if False else e})")      
    from matplotlib.font_manager import findfont, FontProperties
    print("text font:", os.path.basename(findfont(FontProperties(family='serif'))))
    print("figures written to", FIG)
