"""Metaanálisis de ln(RMSE espacial / RMSE AA) con efectos aleatorios (REML + Hartung-Knapp).

Uso:  python metaanalisis.py   (lee datos_extraccion.csv y escribe resultados/ y ../figuras/)
"""
import csv, json, math, os
import numpy as np
from scipy import stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, '..', 'figuras')
RES = os.path.join(AQUI, 'resultados')
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

RHO = 0.5                       # correlación supuesta entre los errores de ambos métodos
AZUL, NARANJA = '#2a78d6', '#eb6834'
TINTA, TINTA2, REJILLA = '#0b0b0b', '#52514e', '#d9d8d4'


def leer():
    filas = []
    with open(os.path.join(AQUI, 'datos_extraccion.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['ln_ratio_directo']:
                y = float(r['ln_ratio_directo'])
            else:
                y = math.log(float(r['rmse_esp']) / float(r['rmse_aa']))
            r['y'] = y
            r['n_val'] = int(r['n_val'])
            r['hibrido'] = int(r['hibrido']); r['cov_equiv'] = int(r['cov_equiv'])
            filas.append(r)
    return filas


def varianza(n, rho=RHO):
    # var[ln RMSE] ≈ 1/(2n) por método; diferencia de dos estimadores correlacionados
    return (1 - rho) / n


def reml(y, v):
    y, v = np.asarray(y), np.asarray(v)
    k = len(y)
    def nll(t2):
        w = 1 / (v + t2)
        mu = np.sum(w * y) / np.sum(w)
        return 0.5 * (np.sum(np.log(v + t2)) + np.log(np.sum(w)) + np.sum(w * (y - mu) ** 2))
    t2 = optimize.minimize_scalar(nll, bounds=(0, 10), method='bounded').x if k > 1 else 0.0
    w = 1 / (v + t2)
    mu = np.sum(w * y) / np.sum(w)
    # Hartung-Knapp-Sidik-Jonkman
    q_hk = np.sum(w * (y - mu) ** 2) / (k - 1) if k > 1 else 1
    se_hk = math.sqrt(max(q_hk, 1e-12) / np.sum(w)) if k > 1 else math.sqrt(1 / np.sum(w))
    tcrit = stats.t.ppf(0.975, k - 1) if k > 1 else 1.96
    # heterogeneidad
    wf = 1 / v
    muf = np.sum(wf * y) / np.sum(wf)
    Q = float(np.sum(wf * (y - muf) ** 2))
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    pQ = 1 - stats.chi2.cdf(Q, k - 1) if k > 1 else float('nan')
    p = 2 * (1 - stats.t.cdf(abs(mu / se_hk), k - 1)) if k > 1 else float('nan')
    pi = (mu - stats.t.ppf(0.975, k - 2) * math.sqrt(t2 + se_hk ** 2),
          mu + stats.t.ppf(0.975, k - 2) * math.sqrt(t2 + se_hk ** 2)) if k > 2 else (float('nan'),) * 2
    return dict(k=k, mu=float(mu), lo=float(mu - tcrit * se_hk), hi=float(mu + tcrit * se_hk), p=float(p),
                tau2=float(t2), Q=Q, pQ=float(pQ), I2=float(I2), pi_lo=float(pi[0]), pi_hi=float(pi[1]),
                w=(w / w.sum() * 100).tolist())


def egger(y, v):
    se = np.sqrt(v)
    res = stats.linregress(1 / se, y / se)
    t = res.intercept / res.intercept_stderr
    p = 2 * (1 - stats.t.cdf(abs(t), len(y) - 2))
    return dict(intercepto=float(res.intercept), p=float(p))


def pct(x):
    """Diferencia porcentual de RMSE del método espacial respecto al AA."""
    return (math.exp(x) - 1) * 100


def main():
    d_todos = leer()
    # Análisis principal: solo comparaciones con un brazo de AA no híbrido (los híbridos van a sensibilidad)
    d = [r for r in d_todos if r['hibrido'] == 0]
    y = [r['y'] for r in d]; v = [varianza(r['n_val']) for r in d]
    out = {'global': reml(y, v)}
    for r, w in zip(d, out['global']['w']):
        r['peso'] = w

    def sub(nombre, cond):
        s = [r for r in d if cond(r)]
        out[nombre] = reml([r['y'] for r in s], [varianza(r['n_val']) for r in s])
        out[nombre]['ids'] = [r['id'] for r in s]

    sub('precipitacion', lambda r: r['variable'] == 'P')
    sub('temperatura', lambda r: r['variable'] == 'T')
    sub('familia_kriging', lambda r: not r['metodo_esp'].upper().startswith(('GWR', 'MGWR')))
    sub('familia_gwr', lambda r: r['metodo_esp'].upper().startswith(('GWR', 'MGWR')))
    sub('cov_equivalentes', lambda r: r['cov_equiv'] == 1)
    sub('cov_no_equivalentes', lambda r: r['cov_equiv'] == 0)
    sub('validacion_independiente', lambda r: 'independ' in r['validacion'].lower() or 'aleatoria 100' in r['validacion'].lower() or 'partición' in r['validacion'].lower())
    sub('validacion_cruzada', lambda r: not ('independ' in r['validacion'].lower() or 'partición' in r['validacion'].lower()))
    # sensibilidad
    sub('sin_riesgo_alto', lambda r: r['riesgo'] != 'Alto')
    sub('solo_riesgo_bajo', lambda r: r['riesgo'] == 'Bajo')
    sub('un_efecto_por_estudio', lambda r: r['id'] != 'arganaraz2025_T')
    # sensibilidad específica de las revisiones del equipo
    def alt(cambios):
        yy = [cambios.get(r['id'], r['y']) for r in d]
        return reml(yy, v)
    out['husrevoglu_sin_EMP'] = alt({'husrevoglu2025': 0.22})
    out['zhu_con_SRF'] = alt({'zhu2023': 0.188171})
    out['chen_con_SRF'] = alt({'chen2021': math.log(35.98 / 32.20)})
    out['con_hibridos'] = reml([r['y'] for r in d_todos], [varianza(r['n_val']) for r in d_todos])
    out['lu_n959'] = reml(y, [varianza(959) if r['id'] == 'lu2023' else vi for r, vi in zip(d, v)])
    for rho in (0.0, 0.8):
        out[f'rho_{rho}'] = reml(y, [varianza(r['n_val'], rho) for r in d])
    loo = []
    for i, r in enumerate(d):
        yy = y[:i] + y[i + 1:]; vv = v[:i] + v[i + 1:]
        m = reml(yy, vv); loo.append((r['id'], m['mu'], m['lo'], m['hi']))
    out['leave_one_out'] = loo
    out['egger'] = egger(np.array(y), np.array(v))
    # prueba de diferencia entre subgrupos (Q entre)
    def qbetween(a, b):
        A, B = out[a], out[b]
        se_a = (A['hi'] - A['mu']) / stats.t.ppf(0.975, A['k'] - 1)
        se_b = (B['hi'] - B['mu']) / stats.t.ppf(0.975, B['k'] - 1)
        z = (A['mu'] - B['mu']) / math.sqrt(se_a ** 2 + se_b ** 2)
        return float(2 * (1 - stats.norm.cdf(abs(z))))
    out['p_sub_variable'] = qbetween('precipitacion', 'temperatura')
    out['p_sub_familia'] = qbetween('familia_kriging', 'familia_gwr')
    out['p_sub_cov'] = qbetween('cov_equivalentes', 'cov_no_equivalentes')
    out['p_sub_validacion'] = qbetween('validacion_independiente', 'validacion_cruzada')
    out['estudios'] = [dict(id=r['id'], y=r['y'], v=varianza(r['n_val']), peso=r['peso']) for r in d]
    json.dump(out, open(os.path.join(RES, 'resultados.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

    # ---------- gráfico de bosque ----------
    plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5, 'axes.edgecolor': TINTA2,
                         'axes.linewidth': 0.6, 'xtick.color': TINTA2, 'ytick.color': TINTA2})
    orden = sorted(d, key=lambda r: (r['variable'], r['y']))
    k = len(orden)
    fig, ax = plt.subplots(figsize=(3.45, 4.6))
    for i, r in enumerate(orden):
        yi = k - i + 1
        se = math.sqrt(varianza(r['n_val']))
        c = AZUL if r['variable'] == 'P' else NARANJA
        mk = 's' if r['variable'] == 'P' else 'o'
        ax.plot([r['y'] - 1.96 * se, r['y'] + 1.96 * se], [yi, yi], color=c, lw=1.2, solid_capstyle='round')
        ax.plot(r['y'], yi, mk, color=c, ms=3 + r['peso'] / 3, mec='white', mew=0.6)
    g = out['global']
    ax.fill([g['lo'], g['mu'], g['hi'], g['mu']], [0.4, 0.75, 0.4, 0.05], color=TINTA, lw=0)
    ax.plot([g['pi_lo'], g['pi_hi']], [0.4, 0.4], color=TINTA2, lw=0.8, ls=(0, (2, 1.5)))
    ax.axvline(0, color=TINTA2, lw=0.6)
    ax.set_yticks([k - i + 1 for i in range(k)] + [0.4])
    etiquetas = [r['id'].replace('_P', ' (P)').replace('_T', ' (T)') for r in orden] + ['Efecto global (IC 95 %)']
    ax.set_yticklabels(etiquetas)
    ax.get_yticklabels()[-1].set_fontweight('bold')
    ax.set_xlabel('ln(RMSE espacial / RMSE AA)\n← favorece espacial          favorece AA →',
                  color=TINTA, fontsize=7)
    ax.grid(axis='x', color=REJILLA, lw=0.4); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.set_ylim(-0.3, k + 1.6)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='s', color=AZUL, lw=1.2, ms=4, label='Precipitación'),
                       Line2D([], [], marker='o', color=NARANJA, lw=1.2, ms=4, label='Temperatura'),
                       Line2D([], [], color=TINTA2, lw=0.8, ls=(0, (2, 1.5)), label='Int. de predicción')],
              loc='lower center', fontsize=6, frameon=False, ncol=3, bbox_to_anchor=(0.38, 1.0),
              handlelength=1.6, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'bosque.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(FIG, 'bosque.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ---------- gráfico de embudo ----------
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    se = np.sqrt(v); ymax = se.max() * 1.1
    s = np.linspace(0, ymax, 50)
    ax.plot(g['mu'] - 1.96 * s, s, color=TINTA2, lw=0.6, ls=(0, (2, 1.5)))
    ax.plot(g['mu'] + 1.96 * s, s, color=TINTA2, lw=0.6, ls=(0, (2, 1.5)))
    ax.axvline(g['mu'], color=TINTA, lw=0.8)
    ax.axvline(0, color=TINTA2, lw=0.5)
    for r, si in zip(d, se):
        c = AZUL if r['variable'] == 'P' else NARANJA
        ax.plot(r['y'], si, 's' if r['variable'] == 'P' else 'o', color=c, ms=4, mec='white', mew=0.6)
    ax.invert_yaxis()
    ax.set_xlabel('ln(RMSE espacial / RMSE AA)'); ax.set_ylabel('Error estándar')
    ax.grid(color=REJILLA, lw=0.4); ax.set_axisbelow(True)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker='s', color=AZUL, lw=0, ms=4, label='Precipitación'),
                       Line2D([], [], marker='o', color=NARANJA, lw=0, ms=4, label='Temperatura')],
              loc='lower left', fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'embudo.pdf')); fig.savefig(os.path.join(FIG, 'embudo.png'), dpi=200)
    plt.close(fig)

    # resumen legible
    def fmt(m):
        return (f"k={m['k']}, ln={m['mu']:.3f} [{m['lo']:.3f}; {m['hi']:.3f}] "
                f"({pct(m['mu']):+.1f} %), p={m['p']:.3f}, tau2={m['tau2']:.4f}, I2={m['I2']:.1f} %")
    for key in ('global', 'precipitacion', 'temperatura', 'familia_kriging', 'familia_gwr', 'cov_equivalentes',
                'cov_no_equivalentes', 'validacion_independiente', 'validacion_cruzada', 'sin_riesgo_alto',
                'con_hibridos', 'lu_n959', 'solo_riesgo_bajo', 'un_efecto_por_estudio', 'rho_0.0', 'rho_0.8', 'husrevoglu_sin_EMP', 'zhu_con_SRF', 'chen_con_SRF'):
        print(f'{key:26s}', fmt(out[key]))
    print('PI global', round(g['pi_lo'], 3), round(g['pi_hi'], 3), '| Q', round(g['Q'], 1), 'pQ', g['pQ'])
    print('Egger', out['egger'])
    print('p subgrupos: variable', round(out['p_sub_variable'], 3), 'familia', round(out['p_sub_familia'], 3),
          'cov', round(out['p_sub_cov'], 3), 'validacion', round(out['p_sub_validacion'], 3))
    print('LOO rango', round(min(x[1] for x in loo), 3), round(max(x[1] for x in loo), 3))
    for r in d:
        print(f"{r['id']:24s} y={r['y']:+.3f} se={math.sqrt(varianza(r['n_val'])):.3f} peso={r['peso']:.1f}")
    print('ids validacion_independiente', out['validacion_independiente']['ids'])


if __name__ == '__main__':
    main()
