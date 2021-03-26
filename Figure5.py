import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm as cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import plotly.graph_objects as go
import plotly.io as pio
pio.orca.config.use_xvfb = True
import seaborn as sns
import pathlib
from itertools import cycle, islice
mpl.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'

#
center = np.array([[0,2],[0,-2],[3,0]])
ratio = [0.9, 0.5, 0.1]
data = []
label = []
ns = 200
for i in range(3):
	noise = np.random.randn(ns, 2) * 0.3
	data.append(center[i] + noise)
	tmp = np.random.binomial(1, ratio[i], ns)
	label.append(tmp)

data = np.concatenate(data, axis=0)
label = np.concatenate(label)

fig, ax = plt.subplots(figsize=(3,3))
for i in range(2):
	ax.scatter(data[label==i, 0], data[label==i, 1], s=1, label='-> Target '+str(i+1))

ax.set_xlabel('Dim-1', fontsize=15)
ax.set_ylabel('Dim-2', fontsize=15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', length=0)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.legend(markerscale=5)
plt.tight_layout()
plt.savefig(indir + 'plot/simulate.pdf', bbox_inches='tight', transparent=True)
plt.close()






indir = '/gale/netapp/home/tanpengcheng/projects/CEMBA_RS2/matrix/'
outdir = '/gale/netapp/home/zhoujt/project/CEMBA/RS2/merge_RS2/cortex/'

rate_reduce = np.load(indir + 'cell_6985_RS1.2_MOp_posterior_disp2k_pc100.npy')
y = umap.fit_transform(rate_reduce[:,:50])
metatmp = np.load(indir + 'cell_6985_RS1.2_MOp_meta.npy', allow_pickle=True)
metatmp[metatmp[:,2]=='MB',2] = 'VTA'

y = np.loadtxt(indir + 'cell_6985_RS1.2_MOp_posterior_disp2k_pc100_ndim50_p50.txt')
leg = ['L2/3', 'L4', 'L5-IT', 'L6-IT', 'L5-ET', 'L6-CT', 'L6b', 'NP', 'CLA', 'Inh', 'Others']
tar = ['MOp', 'SSp', 'ACA', 'VISp', 'STR', 'SC', 'Pons', 'VTA', 'TH', 'MY']
fig, axes = plt.subplots(1, 2, figsize=(8,3), sharex='all', sharey='all')

ax = axes[0]
for i,xx in enumerate(leg):
	cell = (metatmp[:,-3]==xx)
	ax.scatter(y[cell, 0], y[cell, 1], c=legcolor[i], s=3, edgecolors='none', alpha=0.8, rasterized=True, label=xx.replace('-', ' '))

ax.legend(bbox_to_anchor=(1,1), markerscale=3)

ax = axes[1]
cell = (metatmp[:,-2]=='RS1')
ax.scatter(y[cell, 0], y[cell, 1], c='grey', s=1, edgecolors='none', alpha=0.3, rasterized=True, label='Unbiased')
for i,xx in enumerate(tar):
	cell = (metatmp[:,-4]==xx)
	ax.scatter(y[cell, 0], y[cell, 1], c=tarcolor[xx], s=3, edgecolors='none', alpha=0.8, rasterized=True, label=xx)

ax.legend(bbox_to_anchor=(1,1), markerscale=3)

for i,ax in enumerate(axes):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.tick_params(axis='both', which='both', length=0)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_title(['Subclass', 'Target'][i], fontsize=15)
	ax.set_xlabel('t-SNE-1', fontsize=15)

axes[0].set_ylabel('t-SNE-2', fontsize=15)

plt.tight_layout()
plt.savefig(outdir + 'plot/cell_6986_MOp_RS1_RS2_posterior_pc50_p50.meta.pdf', transparent=True, bbox_inches='tight', dpi=300)
plt.close()

metatmp = np.concatenate((metatmp[metatmp[:,-2]=='RS1'], metaall[metaall[:,-5]=='MOp'][:,[0,6,9,10,8,7]]))
metatmp[metatmp[:,4]=='RS1',4] = 'Unbiased'
tar = ['Unbiased', 'SSp', 'ACA', 'STR', 'TH', 'SC', 'VTA', 'Pons', 'MY']
leg = ['L2/3', 'L4', 'L5-IT', 'L6-IT', 'L5-ET', 'L6-CT', 'L6b', 'NP', 'CLA', 'Inh']
ratio = np.array([[np.sum(np.logical_and(metatmp[:,4]==xx, metatmp[:,3]==yy)) for yy in leg] for xx in tar])
ratio = ratio / np.sum(ratio, axis=1)[:,None]
fig, axes = plt.subplots(3, 3, figsize=(6,4), sharex='all')
for i,ax in enumerate(axes.flatten()):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.bar(range(len(leg)), ratio[i], color=legcolor[:len(leg)], width=0.66)
	ax.set_title('->' + tar[i] + ' (n=' + str(np.sum(metatmp[:,4]==tar[i])) + ')')
	ax.set_xticks(range(len(leg)))
	ax.set_xticklabels([])

for ax in axes[-1]:
	ax.set_xticklabels([x.replace('-', ' ') for x in leg], fontsize=12, rotation=60, ha='right')

plt.tight_layout()
plt.savefig(outdir + 'plot/cell_6982_MOp_RS1_RS2.proj_layer_ratio.pdf', transparent=True)
plt.close()



tar = ['MOp', 'SSp', 'ACA', 'VISp', 'STR', 'SC', 'Pons', 'VTA', 'TH', 'MY']
y = np.loadtxt(indir + 'cell_848_L5ET_MOp_posterior_disp2k_pca50_ndim30_nn15_umap_y.txt')
metatmp = metaall[np.logical_and(metaall[:,-5]=='MOp', metaall[:,-1]=='L5-ET')]
metatmp[:,-1] = np.load(indir + 'cell_848_L5ET_MOp_6cluster_res4.0_label.npy')
fig, axes = plt.subplots(1, 2, figsize=(8,3), sharex='all', sharey='all')

ax = axes[0]
for i in range(6):
	cell = (metatmp[:,-1]==str(i))
	ax.scatter(y[cell, 0], y[cell, 1], c=legcolor[i], s=8, edgecolors='none', alpha=0.8, rasterized=True)
	ax.text(np.median(y[cell, 0]), np.median(y[cell, 1]), str(i), fontsize=15, horizontalalignment='center', verticalalignment='center')

ax = axes[1]
for i,xx in enumerate(tar):
	cell = (metatmp[:,-3]==xx)
	ax.scatter(y[cell, 0], y[cell, 1], c=tarcolor[xx], s=8, edgecolors='none', alpha=0.8, rasterized=True, label=xx)

ax.legend(bbox_to_anchor=(1,1), markerscale=2)

for i,ax in enumerate(axes):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.tick_params(axis='both', which='both', length=0)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_title(['Cluster', 'Target'][i], fontsize=15)
	ax.set_xlabel('UMAP-1', fontsize=15)

axes[0].set_ylabel('UMAP-2', fontsize=15)

plt.tight_layout()
plt.savefig(outdir + 'L5PT/plot/cell_848_L5ET_MOp_RS2_posterior_pc50_p50.meta.pdf', transparent=True, bbox_inches='tight', dpi=300)
plt.close()


metaall = np.load(indir + 'cell_4176_L5ET_meta.npy')
meta = metaall[metaall[:,12]=='MOp']
cluster = np.load(indir + 'cell_848_L5ET_MOp_6cluster_res4.0_label.npy')
allclist = np.loadtxt(outdir + 'matrix/allclist_CG.txt', dtype=np.str)
allcdict = {x.split('/')[-1][5:-7].replace('-', '_'):x for x in allclist}
nc = len(set(cluster))
for i in range(nc):
	tmp = [allcdict[x] for x in meta[cluster==i, 0]]
	np.savetxt(outdir + 'L5PT/MOp/merged_allc_cluster/allclist_cluster'+str(i+1)+'.txt', tmp, fmt='%s', delimiter='\n')
	print(len(tmp))

indir = '/gale/netapp/home/zhoujt/project/CEMBA/RS2/merge_RS2/cortex/L5PT/MOp/merged_allc_cluster/DMR_CG_comb/'
fin = open(indir + 'DMR_rms_results_collapsed.tsv')
tmp = fin.readline().strip().split('\t')[6:]
clist = [x[18:] for x in tmp]
dict = {clist[i]:i for i in range(len(clist))}
bed, matrix = [], []
tot = 0
for line in fin:
	tmp = line.strip().split('\t')
	ind = [0 for i in range(len(clist))]
	if len(tmp[5])>0 and int(tmp[3])>1:
		bed.append(['chr'+tmp[0], str(int(tmp[1])-1), tmp[2], tot, tmp[3]])
		tot += 1
		for k in tmp[5].split(','):
			ind[dict[k]] = 1
		matrix.append(ind)

fin.close()
matrix = np.array(matrix)
bed = np.array(bed)
np.save(indir + 'DMR_cluster_hypo.npy', matrix)
np.savetxt(indir + 'DMR_all_hypo.bed', bed, delimiter='\t', fmt='%s')
for i, x in enumerate(clist):
	np.savetxt(indir + 'DMR_'+x+'_hypo.bed', bed[matrix[:, i]==1], delimiter='\t', fmt='%s')


indir = './'
fin = open(indir + 'DMR_rms_results_collapsed.tsv')
tmp = fin.readline().strip().split('\t')[6:]
clist = [x[18:] for x in tmp]
dict = {clist[i]:i for i in range(len(clist))}
bed, matrix = [], []
tot = 0
for line in fin:
	tmp = line.strip().split('\t')
	ind = [0 for i in range(len(clist))]
	if 'NA' in tmp:
		continue
	mc = np.array(tmp[6:]).astype(float)
	if len(tmp[5])>0 and np.mean(mc)>np.min(mc)+0.3:
		bed.append(['chr'+tmp[0], str(int(tmp[1])-1), tmp[2], tot, tmp[3]])
		tot += 1
		for k in tmp[5].split(','):
			if (np.mean(mc)-0.3) > mc[dict[k]]:
				ind[dict[k]] = 1
		matrix.append(ind)

fin.close()
matrix = np.array(matrix)
bed = np.array(bed)
np.save(indir + 'DMR_cluster_hypo.npy', matrix)
np.savetxt(indir + 'DMR_all_diff30_hypo.bed', bed, delimiter='\t', fmt='%s')
for i, x in enumerate(clist):
	np.savetxt(indir + 'DMR_'+x+'_diff30_hypo.bed', bed[matrix[:, i]==1], delimiter='\t', fmt='%s')




python ~/project/VC_Methyl/q8/proj/merge_allc/DMR/flankbed.py DMR_all_hypo 500 5
sort -k1,1 -k2,2n DMR_all_hypo.flank.bed -o DMR_all_hypo.flank.bed


nc = 6
leg = np.array([str(i+1) for i in range(nc)])
indir = '/gale/netapp/home/zhoujt/project/CEMBA/RS2/merge_RS2/cortex/L5PT/MOp/'
data = np.array([np.load(indir + 'merged_allc_cluster/DMR_CG_comb/methyl_ratio/cluster'+xx+'_CG_comb_DMR_all_hypo.flank.npy') for xx in leg])
meth = np.array([data[i,5,:] for i in range(len(leg))]).T

seldmr = []
for i in range(6):
	# dmrtmp = (np.sum(matrix, axis=1)==matrix[:,i])
	diff = np.min(meth[:,np.arange(nc)!=i], axis=1) - meth[:,i]
	idx = np.argsort(diff)[::-1]
	seldmr = seldmr + idx[:100].tolist()

leg = np.array([str(i) for i in range(nc)])
fig, ax = plt.subplots(figsize=(5, 6))
plot = ax.imshow(-np.concatenate(data[:,:,seldmr], axis=0).T, cmap='cividis', vmin=-1.0, vmax=-0.3, aspect='auto')
# plot = ax.imshow(-np.concatenate(data[:,:,seldmr][corder][:,:,rorder], axis=0).T, cmap='cividis', vmin=-1.0, vmax=-0.3, aspect='auto')
for i in range(nc-1):
	ax.plot([11*(i+1)-0.5, 11*(i+1)-0.5], [-0.5, len(seldmr)-0.5], c=[1,1,1])

ax.set_xlim([-0.5, 11*len(leg)-0.5])
ax.set_ylim([len(seldmr)-0.5, -0.5])
ax.set_xticks([i*11+5 for i in range(len(leg))])
# ax.set_yticklabels(leg, fontsize=15, rotation=60, rotation_mode='anchor', ha='right')
ax.set_xticklabels(leg, fontsize=15)
ax.set_xlabel('Clusters', fontsize=15)
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_ylabel(str(data.shape[2]) + ' CG-DMRs', fontsize=15)
cbar = plt.colorbar(plot, ax=ax, shrink=0.3, fraction=0.05, orientation='horizontal')
cbar.solids.set_clim([-0.3, -1.0])
cbar.set_ticks([-0.3, -1.0])
cbar.set_ticklabels([0.3, 1.0])
cbar.set_label('mCG', fontsize=15)
cbar.ax.yaxis.set_label_position('left')
cbar.draw_all()

plt.tight_layout()
plt.savefig(indir + 'plot/MOp_L5-ET_6cluster_DMR_CG_comb_flank.pdf', transparent=True)
plt.close()

count = np.sum(matrix, axis=0)

fig, ax = plt.subplots(figsize=(2,6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.barh(range(nc), np.log(count), color='b', height=0.66)
ax.plot(np.log([10000,10000]), [-0.5, nc-0.5], 'k--', linewidth=0.5)
ax.plot(np.log([100000,100000]), [-0.5, nc-0.5], 'k--', linewidth=0.5)
ax.set_xlim(np.log([5000, 200000]))
ax.set_xticks(np.log([10000, 100000]))
ax.set_xticklabels(['10 k', '100 k'], fontsize=15)
ax.set_xlabel('# hypo-CG-DMRs', fontsize=15)
ax.set_ylim([nc-0.5, -0.5])
# ax.yaxis.tick_right()
ax.set_yticks(range(nc))
ax.set_yticklabels(range(nc), fontsize=15)
plt.tight_layout()
plt.savefig(indir + 'plot/L5-ET_15cluster_DMR_CG_comb_count_bar.pdf', transparent=True)
plt.close()


outdir = '/gale/netapp/home/tanpengcheng/projects/CEMBA_RS2/matrix/'
cluster = np.load(outdir + 'cell_11827_posterior_disp2k_pc100_nd50_p50.pc50.knn25.louvain_res1.2_nc25_cluster_label.npy')
rate_gene = np.load(outdir + 'cell_11827_rate_gene.mCH.npy')
gene_all = np.loadtxt('/gale/netapp/home/zhoujt/resource/gene/GENCODE/gencode.vM10.bed', dtype=np.str)
genefilter = np.load('/gale/netapp/home/zhoujt/project/CEMBA/RS2/merge_RS2/cortex/matrix/cell_11827_gene_filter80_autosomal.npy')
gene = gene_all[genefilter]
meta = np.load(outdir + 'cell_11827_meta.npy')

rateg = rate_gene[:, genefilter] / meta[:,8].astype(float)[:,None]
rateg[rateg>10] = 10
meta = np.concatenate((meta[:,7:16], cluster[:,None]), axis=1)

cellfilter = (meta[:,5]=='MOp')
rateg = rateg[cellfilter]
meta = meta[cellfilter]

# cluster = np.load(indir + 'cell_848_L5ET_MOp_6cluster_res4.0_label.npy')
# nc = len(set(cluster))

indir = '/gale/netapp/home/zhoujt/project/CEMBA/RS2/merge_RS2/cortex/L5PT/MOp/'
f = h5py.File(indir + 'cell_2111_MOp.hdf5', 'w')
tmp = f.create_dataset('rate_gene_ch', rateg.shape, dtype=float, compression='gzip')
tmp[()] = rateg
tmp = f.create_dataset('meta', (meta.shape[0]+1, meta.shape[1]), dtype=h5py.string_dtype(encoding='utf-8'), compression='gzip')
tmp[()] = np.concatenate([np.array(['mCCC', 'mCH', 'mCG', '# Non-clonal reads', 'Experiment', 'Source', 'Slice', 'Target', 'Gender', 'Major Type'])[None,:] ,meta], axis=0)
tmp = f.create_dataset('genes', gene.shape, dtype=h5py.string_dtype(encoding='utf-8'), compression='gzip')
tmp[()] = gene
f.close()

cellfilter = np.logical_and(meta[:,4]=='L5-ET', meta[:,0]=='MOp')
rateg = rateg[cellfilter]
meta = meta[cellfilter]
cluster = np.load(indir + 'cell_848_L5ET_MOp_6cluster_res4.0_label.npy')
nc = len(set(cluster))
meta[:,4] = cluster[:]

outdir = '/gale/netapp/home/zhoujt/project/CEMBA/RS2/merge_RS2/cortex/L5PT/MOp/cluster_dmg/'
data = pd.DataFrame(data=rateg, columns=gene[:,-1])
data['target'] = meta[:,2]
data['gender'] = meta[:,3]
data['cluster'] = meta[:,4]
data['global'] = meta[:,5]
data.to_csv(outdir + 'cell_848_rateg.meta.txt', index=None, sep='\t')
para_list = np.array([[i,j] for i in range(nc-1) for j in range(i+1, nc)])
np.savetxt(outdir + 'para_list.txt', para_list, fmt='%s', delimiter='\t')


require(data.table)
library(lmerTest)
args = commandArgs(trailingOnly=TRUE)
indir <- '/gale/netapp/home/zhoujt/project/CEMBA/RS2/merge_RS2/cortex/L5PT/cluster_dmg/'
fin <- paste(indir, 'cell_4176_rateg.meta.txt', sep='')
data <- fread(fin)
data$mice <- apply(data[,c('gender', 'target')], 1, paste, collapse="-")
data <- data[which((data$cluster==args[1]) | (data$cluster==args[2]))]
pv <- c()
ngene <- 12261
for (i in 1:ngene){
	datatmp = data[,c(i, ngene+1, ngene+2, ngene+3, ngene+4, ngene+5), with=FALSE]
	colnames(datatmp)[1] = 'mCH'
	model <- lmer(mCH ~ cluster + gender + global + (1 | mice), data=datatmp)
	pv <- append(pv, anova(model)[6])
}
fout <- paste(indir, paste('cluster', args[1], args[2], sep='_'), '.ga.m.pvalue.txt', sep='')
write.table(pv, file=fout, quote=FALSE, sep='\t', row.names=FALSE, col.names=FALSE)


def rs_roc_pr_ovo(args):
	global rateg, cluster, gene
	i, j = args
	print(i, j)
	rate1 = rateg[cluster==i]
	rate2 = rateg[cluster==j]
	pv = np.array([ranksums(rate1[:,k], rate2[:,k])[1] for k in range(len(gene))])
	fdr = FDR(pv, 0.01, 'fdr_bh')[1]
	cellfilter = np.logical_or(cluster==i, cluster==j)
	rate = rateg[cellfilter]
	label = (cluster[cellfilter]==i)
	roc = np.array([roc_auc_score(label, rate[:,k]) for k in range(len(gene))])
	npos, nneg = np.sum(label), np.sum(~label)
	if npos > nneg:
		label = ~label
	else:
		rate = -rate
	pr1 = np.array([average_precision_score(label, rate[:,k]) for k in range(len(gene))])
	pr2 = np.array([average_precision_score(label, -rate[:,k]) for k in range(len(gene))])
	return [i,j,fdr,roc,pr1,pr2]

def rs_roc_pr_ovr(i):
	print(i)
	global rateg, cluster, gene
	rate1 = rateg[cluster==i]
	rate2 = rateg[cluster!=i]
	pv = np.array([ranksums(rate1[:,k], rate2[:,k])[1] for k in range(len(gene))])
	fdr = FDR(pv, 0.01, 'fdr_bh')[1]
	label = (cluster==i)
	npos, nneg = np.sum(label), np.sum(~label)
	roc = np.array([roc_auc_score(label, rateg[:,k]) for k in range(len(gene))])
	if npos > nneg:
		label = ~label
	pr = np.array([average_precision_score(label, -rateg[:,k]) for k in range(len(gene))])
	return [i,fdr,roc,pr]

p = Pool(5)
result = p.map(rs_roc_pr_ovo, para_list)
p.close()

for x in result:
	np.savetxt(outdir + 'cluster_'+str(x[0])+'_'+str(x[1])+'.rs.roc.pr.txt', np.array([x[2], x[3], x[4], x[5]]).T, fmt='%s', delimiter='\t')


p = Pool(5)
result = p.map(rs_roc_pr_ovr, np.arange(nc))
p.close()

for x in result:
	print(x[0])
	idx = np.argsort(x[2])
	print(gene[idx[:10], -1])
	idx = np.argsort(x[3])[::-1]
	print(gene[idx[:10], -1])

0
['Ptprg' 'St6galnac3' 'Kcnd3' 'Nfia' 'Fam196a' 'Kctd1' 'Grik4' 'Plekha2'
 'Cacna1e' 'Tenm4']
['Ptprg' 'St6galnac3' 'Kcnd3' 'Fam196a' 'Pid1' 'Ddah1' 'Dgkg' 'Cacna1e'
 'Nfia' 'Fhit']
1
['Ccbe1' 'Wbscr17' 'Stxbp6' 'Ppm1h' 'Rasgrf2' 'Kctd16' 'Lrrtm4' 'Rab3b'
 'Fam189a1' 'Gria4']
['Kirrel3' 'Ccbe1' 'Lrrtm4' 'Kctd16' 'Gria4' 'Stxbp6' 'Ppm1h' 'Wbscr17'
 'Fam189a1' 'Sntb1']
2
['Camta1' 'Atrnl1' 'Grid1' 'Nckap5' 'Slc4a4' 'Ephb1' 'Ano6' 'Kcnip4'
 'Alcam' 'Ust']
['Camta1' 'Atrnl1' 'Grid1' 'Slc4a4' 'Kcnip4' 'Ephb1' 'Ano6' 'Nckap5'
 'Lsamp' 'Ust']
3
['Syn2' 'Cdh13' 'Dner' 'RP23-253I14.4' 'Map3k5' 'A930011G23Rik' 'Ptpre'
 'Slit3' 'Phlpp1' 'Naaladl2']
['Cntnap5a' 'Hs3st5' 'Syn2' 'Ext1' 'Cdh13' 'Map3k5' 'Dner' 'Naaladl2'
 'Slit3' 'RP23-253I14.4']
4
['Galntl6' 'Negr1' 'Gpc6' 'Kif16b' 'Galnt13' 'Cacna2d3' 'Sorcs1' 'Cntnap2'
 'Ptpro' 'Usp6nl']
['Galntl6' 'Negr1' 'Gpc6' 'Ntm' 'Cntnap2' 'Cacna2d3' 'Sorcs1' 'Galnt13'
 'Kif16b' 'Gm20754']
5
['Klf12' 'Tmtc2' 'Dscam' 'Tox2' 'Foxp1' 'Dpy19l1' 'Il1rap' 'Sorcs3'
 'Agbl4' 'Lhfp']
['Agbl4' 'Foxp1' 'Klf12' 'Dscam' 'Tmtc2' 'Slc24a3' 'Lrp1b' 'Pcdh9' 'Tox2'
 'Gm20696']

for i in range(6):
	print(i)
	roc, pr = [], []
	for j in range(6):
		if i<j:
			data = np.loadtxt(outdir + 'cluster_'+str(i)+'_'+str(j)+'.rs.roc.pr.txt')
			roc.append(data[:,1])
			pr.append(data[:,2])
		elif i>j:
			data = np.loadtxt(outdir + 'cluster_'+str(j)+'_'+str(i)+'.rs.roc.pr.txt')
			roc.append(1-data[:,1])
			pr.append(data[:,3])
	roc = np.max(roc, axis=0)
	idx = np.argsort(roc)
	print(gene[idx[:10], -1])
	pr = np.min(pr, axis=0)
	idx = np.argsort(pr)[::-1]
	print(gene[idx[:10], -1])

0
['Ptprg' 'Kcnd3' 'Fam196a' 'Qk' 'Plekha2' 'Skap1' 'Grem2' 'Cadm1' 'Ddah1'
 'St6galnac3']
['Cntnap2' 'Sorcs1' 'Auts2' 'Trpc4' 'Ptpro' 'Sema6a' 'Cdh20' 'Nup93'
 'Dscam' 'Robo2']
1
['Ccbe1' 'Ppm1h' 'Sntb1' 'Rab3b' 'Gria4' 'Pag1' 'Wbscr17' 'Prex1' 'Kctd16'
 'Stxbp6']
['Tmtc2' 'Arhgap15' 'Efna5' 'Ubl3' 'Naaladl2' 'RP23-357C16.1' 'Tiam2'
 'Prex2' 'Ext1' 'Lnx2']
2
['Camta1' 'Slc4a4' 'Atrnl1' 'Grid1' 'Ano6' 'Ephb1' 'Ust' 'Lsamp' 'Alcam'
 'Ltbp1']
['Kcnt2' 'Efna5' 'Plcxd2' 'Fam20a' 'Lrba' 'Adcy7' 'Ubl3' 'Grm7' 'Nav3'
 'Lrfn5']
3
['Syn2' 'Slc22a22' 'Hs3st5' 'Top3a' 'Ext1' 'Plcxd2' 'Samd12' 'Nav3'
 'Cpne2' 'Ttc6']
['Ext1' 'Cntnap5a' 'Hs3st5' 'Naaladl2' 'Efna5' 'Lace1' 'Cdh20' 'Grm7'
 'Syn2' 'Unc5d']
4
['Galntl6' 'Negr1' 'Dlg2' 'Cacna2d3' 'Synpr' 'Sorcs1' 'Zfyve28' 'Fam222a'
 'Gm20754' 'Grik2']
['Sdccag8' 'Ctbp2' 'Elovl6' 'Lrfn5' 'Efna5' 'Ddx25' 'Nubpl' 'Plcxd2'
 'Bcar3' 'Pdzd8']
5
['Klf12' 'Tox2' 'Bace2' 'Dscam' 'Foxp1' 'Tmtc2' 'Lhfp' 'Dpy19l1' 'Il1rap'
 'Kctd8']
['Efna5' 'Naaladl2' 'Cntnap5a' 'Pde6c' 'Lrfn5' 'Gtdc1' 'Dab1' 'Plcxd2'
 'Ankrd55' 'Jam2']

rateg_cluster = np.array([np.mean(rateg[cluster==i], axis=0) for i in range(nc)])
selg = np.zeros(len(gene))
for i in range(nc-1):
	for j in range(i+1, nc):
		pv = np.loadtxt(outdir + 'cluster_'+str(i)+'_'+str(j)+'.ga.m.wald.pvalue.txt')
		fdr = FDR(pv, 0.01, 'fdr_bh')[1]
		fc = np.log2((rateg_cluster[i] + 0.5) / (rateg_cluster[j] + 0.5))
		tmp = np.logical_and(fdr<0.01, np.abs(fc)>np.log2(1.5))
		# df = rateg_cluster[i] - rateg_cluster[j]
		# tmp = np.logical_and(fdr<0.01, np.abs(df)>0.5)
		selg = np.logical_or(selg, tmp)
		print(i, j, np.sum(fdr<0.01), np.sum(np.abs(fc)>np.log2(1.5)), np.sum(tmp))

print(np.sum(selg))

selg = np.zeros(len(gene))
i = 0
for j in range(1,nc):
	pv = np.loadtxt(outdir + 'cluster_'+str(i)+'_'+str(j)+'.ga.m.wald.pvalue.txt')
	fdr = FDR(pv, 0.01, 'fdr_bh')[1]
	fc = np.log2((rateg_cluster[i] + 0.5) / (rateg_cluster[j] + 0.5))
	tmp = np.logical_and(fdr<0.01, fc<-np.log2(1.5))
	# df = rateg_cluster[i] - rateg_cluster[j]
	# tmp = np.logical_and(fdr<0.01, np.abs(df)>0.5)
	selg = np.logical_or(selg, tmp)
	print(i, j, np.sum(fdr<0.01), np.sum(fc<-np.log2(1.5)), np.sum(tmp))

print(np.sum(selg))

# rateg_cluster_neg = np.array([np.mean(rateg[cluster!=i], axis=0) for i in range(nc)])
# selg = np.zeros(len(gene))
# for i in range(nc-1):
# 	pv = np.loadtxt(outdir + 'cluster_'+str(i)+'.ga.m.ovr.wald.pvalue.txt')
# 	fdr = FDR(pv, 0.01, 'fdr_bh')[1]
# 	fc = np.log2((rateg_cluster[i] + 0.5) / (rateg_cluster_neg[i] + 0.5))
# 	tmp = np.logical_and(fdr<0.01, fc<-np.log2(1.5))
# 	# df = rateg_cluster[i] - rateg_cluster[j]
# 	# tmp = np.logical_and(fdr<0.01, np.abs(df)>0.5)
# 	selg = np.logical_or(selg, tmp)
# 	print(i, j, np.sum(fdr<0.01), np.sum(fc<-np.log2(1.5)), np.sum(tmp))

# print(np.sum(selg))

np.savetxt(indir + 'cell_4176_L5ET_15cluster_ovo_ga_m_pv01_fc50_pseudo50.txt', gene[selg,-1], fmt='%s', delimiter='\n')

mch = rateg_cluster.T
leg = [str(i) for i in range(nc)]
cg = clustermap(mch[selg], vmin=-2, vmax=2, cmap='bwr', xticklabels=leg, yticklabels=[], metric='cosine', cbar_kws={'ticks':[-2, 2]}, z_score=0)
cg.ax_heatmap.tick_params(axis='both', which='both', length=0)
plt.setp(cg.ax_heatmap.get_xticklabels(), fontsize=12, rotation=60, ha='right', rotation_mode='anchor')
cg.cax.tick_params(labelsize=12)
cg.ax_row_dendrogram.set_visible(False)
cg.ax_col_dendrogram.set_visible(False)
hm = cg.ax_heatmap.get_position()
cg.ax_heatmap.set_position([hm.x0, hm.y0+hm.height*0.6, hm.width*0.5, hm.height*0.4])
cg.ax_heatmap.set_ylabel(np.str(np.sum(selg)) + ' CH-DMG', fontsize=12)
cg.ax_heatmap.yaxis.set_label_position('left')
cb = cg.cax.get_position()
cg.cax.set_position([hm.x0+0.55*hm.width, hm.y0+hm.height*0.6, cb.width, cb.height])
cg.cax.yaxis.set_label_position('left')
cg.cax.set_ylabel('mCH z-score', fontsize=12)
cg.savefig(indir + 'plot/MOp_L5-ET_6cluster_ovo_ga_m_pv01_fc50_pseudo50_mch.pdf', transparent=True)

genedict = {x:i for i,x in enumerate(gene[:,-1])}
marker = ['Astn2', 'Ccbe1', 'Camta1', 'Syn2', 'Galntl6', 'Foxp1']
marker = ['Kcnd3', 'Ccbe1', 'Grid1', 'Ext1', 'Galntl6', 'Dscam']
color = np.array(list(islice(cycle(['#CB4335','#F39C12','#F4D03F','#27AE60','#76D7C4','#3498DB','#8E44AD','#6D4C41','#FFAB91','#1A237E','#FFFF66','#004D40','#00FF33','#CCFFFF','#A1887F','#FFCDD2','#999966','#212121','#FF00FF']), 100)))
fig, axes = plt.subplots(len(marker)//2, 2, figsize=(3,3))
for k,ax in zip(marker, axes.flatten()):
	ax = sns.boxplot(x=cluster, y=rateg[:,genedict[k]], palette={i:x for i,x in enumerate(color[:nc])}, order=np.arange(nc), showfliers=False, ax=ax)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	if not ax in axes[-1]:
		ax.set_xticklabels([])
	else:
		ax.set_xlabel('Clusters')
	ax.set_title(k)

axes[1,0].set_ylabel('mCH')
plt.tight_layout()
plt.savefig(indir + 'plot/MOp_L5-ET_6cluster_marker_boxplot_new.pdf', transparent=True)
plt.close()

