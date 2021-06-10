import numpy as np
import scipy.stats as stats
import sklearn.manifold as manifold
from pybosaris.libperformance import min_cllr, cllr
from pybosaris.libmath import pavx, optimal_llr_from_Popt, optimal_llr
from matplotlib import cm
import pandas


__author__ = ['Tomi Kinnunen', 'Andreas Nautsch', 'Md Sahidullah']
__credits__ = ['Nicholas Evans', 'Xin Wang', 'Massimiliano Todisco', 'Héctor Delgado', 'Junichi Yamagishi', 'Kong Aik Lee']


score_file = pandas.read_csv('scores.txt.zip', sep=' ')
systems = score_file.columns[3:]
key = (score_file.trial_type == 'target').values

scores  = score_file[systems].values.T
nsys, _ = np.shape(scores)

database = np.vstack(systems.str.split('_').values)[:, 0]  # 2 databases
nfeat = np.vstack(systems.str.split('_').values)[:, 1]     # 4 feature types
ndist = np.vstack(systems.str.split('_').values)[:, 2]     # 10 distribution counts

unique_feats = np.asarray(['12', '16', '20', '24'])  # np.unique(nfeat)
unique_dist = np.asarray(['2', '4', '8', '16', '32', '64', '128', '256', '512', '1024'])  # hard coded for better control


# set True to run the pool-adjacent violators experiment; then, scores are pre-calibrated
oracle_calibration = False


# performance estimation
mcllr = np.zeros(nsys)
for i in range(0, nsys):
    tar = scores[i, key]
    non = scores[i, ~key]
    # mcllr[i] = min_cllr(tar_llrs, nontar_llrs)  # short form for the below
    cal_scores = np.concatenate([tar, non])
    Pideal = np.concatenate([np.ones(len(tar)), np.zeros(len(non))])
    perturb = np.argsort(cal_scores, kind='mergesort')
    Pideal = Pideal[perturb]
    Popt, width, foo = pavx(Pideal)
    tar_llrs, nontar_llrs = optimal_llr_from_Popt(Popt=Popt, perturb=perturb, Ntar=len(tar), Nnon=len(non), monotonicity_epsilon=1e-6)
    print("Team {} - Popt: {}, width: {}, foo: {}".format(i+1, Popt.shape, width.shape, foo.shape))  # effective reduction of score groups
    # min Cllr performance computation is now equivalent to taking Cllr from scores after oracle calibration
    mcllr[i] = cllr(tar_llrs, nontar_llrs)

    # using PAV scores instead
    if oracle_calibration:
        # scores are calibrated using Laplace's rule of succession: infinite LLR values are avoided
        tar_llrs, nontar_llrs = optimal_llr(tar, non, monotonicity_epsilon=0, laplace=True)
        scores[i, key] = tar_llrs
        scores[i, ~key] = nontar_llrs


# Compute pairwise system similarities 
sys_sim = np.empty([nsys, nsys])
for i in range(0, nsys):
    print("Computing Kendall's tau for team {}/{}".format(i+1, nsys))
    tar = scores[i, key]
    non = scores[i, ~key]
    for j in range(0, nsys):
        sys_sim[i, j], _ = stats.kendalltau(scores[i, :], scores[j, :])

# Distance matrix from Kendall's tau matrix
dist_matrix = (-sys_sim+1)/2

# MDS - multidimensional scaling
seed = np.random.RandomState(seed=3)
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(dist_matrix).embedding_

# setting up the LaTeX plots
topn = 5
top_sys_idx = mcllr.argsort()[:topn]
linewidths = np.ones(nsys)
linewidths[top_sys_idx] = 3

markers = np.asarray(nsys * ['o'])
markers[(database == 'Vox') & (nfeat == '12')] = 'd'
markers[(database == 'Vox') & (nfeat == '16')] = 's'
markers[(database == 'Vox') & (nfeat == '20')] = 'P'
markers[(database == 'Vox') & (nfeat == '24')] = 'X'
markers[(database == 'Lib') & (nfeat == '12')] = 'o'
markers[(database == 'Lib') & (nfeat == '16')] = 'p'
markers[(database == 'Lib') & (nfeat == '20')] = 'h'
markers[(database == 'Lib') & (nfeat == '24')] = '8'

cmap = cm.get_cmap('BrBG')(np.linspace(0, 1, len(np.unique(ndist))))
colors = np.zeros((nsys, cmap.shape[1]))
colorbrew = np.empty(nsys, dtype='|O')
for i, d in enumerate(unique_dist):
    colors[ndist == d, :] = cmap[i, :]
    colorbrew[ndist == d] = 'PuOr-' + str(len(np.unique(ndist))) + '-' + str(i+1)

# creating tikz file string by string
tikz_code = """
\\begin{{tikzpicture}}[font=\\scriptsize]
    \\begin{{axis}}[
    width=\\mdswidth,
    height=\\mdsheight,
    legend columns=2,
    legend cell align={{left}},
    legend style={{anchor=north,draw=white!80!black,at={{(0.5,-.15)}}}},
    ticks=none,
    xmin={xmin}, xmax={xmax},
    ymin={ymin}, ymax={ymax}
    ]
""".format(xmin=pos[:, 0].min()-0.1, xmax=pos[:, 0].max()+0.1, ymin=pos[:, 1].min()-0.1, ymax=pos[:, 1].max()+0.1)

normal_scale = 0.75

marks = np.asarray(['*', 'square*', 'triangle*', 'diamond*', 'pentagon*'])
for i, db in enumerate(np.unique(database)):
    if db == 'Vox':
        marks = np.asarray(['*', 'square*', 'triangle*', 'diamond*', 'pentagon*'])
        fill = 'fill'
    else:
        marks = np.asarray(['o', 'square', 'triangle', 'diamond', 'pentagon'])
        fill = 'draw'

    for j, nf in enumerate(unique_feats):
        draws = "\\addplot"
        if db == 'Vox':
            draws += "[densely dotted]"
        else:
            draws += "[solid]"
        draws += " coordinates {"
        for k, nd in enumerate(unique_dist):
            idx = i * len(np.unique(nfeat)) * len(np.unique(ndist)) + j * len(unique_dist) + k
            scale = normal_scale
            lw = 'thin'
            draws += "({},{})".format(pos[idx, 0], pos[idx, 1])
            if db == 'Vox':
                clr = colorbrew[idx]
                clr2 = 'black'
            else:
                clr = 'white'
                clr2 = colorbrew[idx]

            if idx in top_sys_idx:
                lw = 'thick'
                scale += 0.5

                tikz_code += """
        \\addplot [only marks, mark={mark}, mark options={{scale={scale}, solid, {lw}, draw={edge}, fill={color}}}]
        table{{%
        x                      y
        {pos0} {pos1}
        }};""".format(pos0=pos[idx, 0], pos1=pos[idx, 1], scale=scale, mark=marks[j], color=clr, edge=clr2, lw=lw)

        draws += "};"
        tikz_code += draws

for i, db in enumerate(np.unique(database)):
    for j, nf in enumerate(unique_feats):
        for k, nd in enumerate(unique_dist):
            idx = i * len(np.unique(nfeat)) * len(np.unique(ndist)) + j * len(unique_dist) + k
            scale = normal_scale
            lw = 'thin'
            if db == 'Vox':
                clr = colorbrew[idx]
                clr2 = 'black'
            else:
                clr = 'white'
                clr2 = colorbrew[idx]

            if ((k == 0) or (k == len(unique_dist)-1)):
                tikz_code += """
        \\node[above,font=\\tiny] at ({pos0}, {pos1}) {{{performance}}};
                """.format(pos0=pos[idx, 0], pos1=pos[idx, 1], performance=mcllr[idx].round(3))

            if idx in top_sys_idx:
                tikz_code += """
        \\node[above,font=\\tiny] at ({pos0}, {pos1}) {{{performance}}};
                """.format(pos0=pos[idx, 0], pos1=pos[idx, 1], performance=mcllr[idx].round(3))
                continue

            tikz_code += """
        \\addplot [only marks, mark={mark}, mark options={{scale={scale}, solid, {lw}, draw={edge}, fill={color}}}]
        table{{%
        x                      y
        {pos0} {pos1}
        }};""".format(pos0=pos[idx, 0], pos1=pos[idx, 1], scale=scale, mark=marks[j], color=clr, edge=clr2, lw=lw)

tikz_code += """
    \\end{axis}
\\end{tikzpicture}
"""

str_oracle = ''
if oracle_calibration:
    str_oracle = '_oracle'

with open("asv" + str_oracle + ".tikz", "w") as text_file:
    text_file.write(tikz_code)

legend = """
\\begin{tikzpicture}[font=\\scriptsize]
    \\begin{customlegend}[
        legend cell align=left,
        legend columns=3,
        legend entries={
            VoxCeleb, LibriSpeech, top-5, 12 filters, 16 filters, 20 filters, 24 filters, """ + \
            ' mixtures, '.join(unique_dist) + ' mixtures' + """
        },
        legend style={draw=white!80!black,anchor=north}]
            \\addlegendimage{mark=*, densely dotted,mark options={scale="""+str(normal_scale)+""",solid,draw=black,thin,fill=blue}}
            \\addlegendimage{mark=o, solid, mark options={scale="""+str(normal_scale)+""",draw=black,thin,fill=white}}
            \\addlegendimage{only marks ,mark=o, mark options={scale="""+str(normal_scale)+""",draw=black,thick,fill=white}}
            \\addlegendimage{only marks ,mark=o, mark options={scale="""+str(normal_scale)+""",draw=black,thin,fill=white}}
            \\addlegendimage{only marks ,mark=square, mark options={scale="""+str(normal_scale)+""",draw=black,thin,fill=white}}
            \\addlegendimage{only marks ,mark=triangle, mark options={scale="""+str(normal_scale)+""",draw=black,thin,fill=white}}
            \\addlegendimage{only marks ,mark=diamond, mark options={scale="""+str(normal_scale)+""",draw=black,thin,fill=white}}
"""

for k, nd in enumerate(unique_dist):
    legend += """
            \\addlegendimage{only marks ,mark=*, mark options={scale="""+str(normal_scale)+""",draw=black,thin,fill={color}}}}}
    """.format(color=colorbrew[k])

legend += """
    \\end{customlegend}
\\end{tikzpicture}
"""

with open("asv_legend" + str_oracle + ".tikz", "w") as text_file:
    text_file.write(legend)
