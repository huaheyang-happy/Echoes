import anndata as ad
import torch
import scanpy as sc
import episcanpy as epi
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
import scipy
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import anndata as ad
import scipy
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

atac_file_path = "/home/zzy/python/数据/Yao-2021-ATAC.h5ad"
rna_file_path = "/home/zzy/python/数据/Yao-2021-RNA.h5ad"

def tfidf1(count_mat): 
    epsilon = 1e-12
    nfreqs = 1.0 * count_mat / (np.tile(np.sum(count_mat,axis=0), (count_mat.shape[0], 1)) + epsilon)
    tfidf_mat = np.multiply(nfreqs, np.tile(np.log(1 + 1.0 * count_mat.shape[1] / np.sum(count_mat,axis=1)).reshape(-1,1), (1,count_mat.shape[1])))
    return scipy.sparse.csr_matrix(tfidf_mat)
def tfidf2(count_mat): 
    tf_mat = 1.0 * count_mat / np.tile(np.sum(count_mat,axis=0), (count_mat.shape[0],1))
    signac_mat = np.log(1 + np.multiply(1e4*tf_mat,  np.tile((1.0 * count_mat.shape[1] / np.sum(count_mat,axis=1)).reshape(-1,1), (1,count_mat.shape[1]))))
    return scipy.sparse.csr_matrix(signac_mat)
def tfidf3(count_mat): 
    model = TfidfTransformer(smooth_idf=False, norm="l2")
    model = model.fit(np.transpose(count_mat))
    model.idf_ -= 1
    tf_idf = np.transpose(model.transform(np.transpose(count_mat)))
    return scipy.sparse.csr_matrix(tf_idf)

def compute_pairwise_distances(x, y):
    """计算平方欧氏距离矩阵"""
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def _gaussian_kernel_matrix(x, y, sigmas):
    """计算多高斯核矩阵"""
    dist = compute_pairwise_distances(x, y)
    beta = 1. / (2. * sigmas[:, None]**2 + 1e-8)
    s = -beta @ dist.view(1, -1)
    return torch.sum(torch.exp(s), dim=0).view_as(dist)

def mmd_loss_latent(x, y, device, kernel_type='gaussian_multi'):
    """计算 MMD 损失"""
    sigmas_val = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    sigmas = torch.tensor(sigmas_val, device=device)
    K_xx = _gaussian_kernel_matrix(x, x, sigmas)
    K_yy = _gaussian_kernel_matrix(y, y, sigmas)
    K_xy = _gaussian_kernel_matrix(x, y, sigmas)
    loss_mmd = torch.mean(K_xx) + torch.mean(K_yy) - 2.0 * torch.mean(K_xy)
    return torch.clamp(loss_mmd, min=0.0)

adata_atac = sc.read_h5ad(atac_file_path)
print(f"原始 ATAC 数据维度: {adata_atac.shape}")
fpeak = 0.01
epi.pp.binarize(adata_atac)
epi.pp.filter_features(adata_atac, min_cells = np.ceil(fpeak*adata_atac.shape[0]))
tfidf_res = tfidf1(adata_atac.X.T).T
adata_atac.X = tfidf_res.copy()
max_temp = np.max(adata_atac.X)
adata_atac.X = adata_atac.X / max_temp

X_atac = adata_atac.X.toarray()#稀疏矩阵转成密集矩阵
atac_dict = {}
y_atac = adata_atac.obs["cell_type"].unique().tolist()
for i in range(len(y_atac)):
    atac_dict[y_atac[i]] = i
y_atac = [atac_dict[i] for i in adata_atac.obs["cell_type"]]
print(f"预处理后的形状：{adata_atac.shape}")



adata_rna = ad.read_h5ad(rna_file_path)
print(f"原始 RNA 数据维度: {adata_rna.shape}")
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi = 80, facecolor = 'white')
sc.pp.calculate_qc_metrics(adata_rna, percent_top = None, log1p = False, inplace = True)
sc.pp.filter_genes(adata_rna, min_cells = 3)
sc.pp.filter_cells(adata_rna, min_genes = 200)
adata_rna = adata_rna[adata_rna.obs.n_genes_by_counts < 5000, :]
sc.pp.normalize_total(adata_rna, target_sum = 1e4)
sc.pp.log1p(adata_rna)
sc.pp.highly_variable_genes(adata_rna, flavor = "seurat", n_top_genes = 3000, subset = True)
scaler = MinMaxScaler()
adata_rna.X = scaler.fit_transform(adata_rna.X.toarray())
print(f"原始 ATAC 数据维度: {adata_atac.shape}")

X_rna = adata_rna.X#已经是密集矩阵了，不需要再转成密集矩阵
rna_dict = {}
y_rna = adata_rna.obs["cell_type"].unique().tolist()
for i in range(len(y_rna)):
    rna_dict[y_rna[i]] = i
y_rna = [rna_dict[i] for i in adata_rna.obs["cell_type"]]
print(f"预处理后的形状：{adata_rna.shape}")

class Dataset(Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        assert self.X.shape[0] == len(self.Y)
        return self.X.shape[0]
    def __getitem__(self,index):
        x_tensor = torch.tensor(self.X[index], dtype=torch.float32)
        y_tensor = torch.tensor(self.Y[index],dtype = torch.int32)#要.squeeze()因为要去掉维度为1的那个维度

        return x_tensor,y_tensor


atac_train, atac_test, atac_train_labels, atac_test_labels = train_test_split(X_atac, y_atac, test_size = 0.3,random_state = 33)
atac_train_dataset = Dataset(atac_train,atac_train_labels)
atac_test_dataset = Dataset(atac_test,atac_test_labels)
atac_train_dataloader = DataLoader(atac_train_dataset, batch_size = 128, shuffle = True)
atac_test_dataloader = DataLoader(atac_test_dataset, batch_size = 128, shuffle = False)

rna_train, rna_test, rna_train_labels, rna_test_labels = train_test_split(X_rna, y_rna, test_size = 0.3,random_state = 33)
rna_train_dataset = Dataset(rna_train,rna_train_labels)
rna_test_dataset = Dataset(rna_test,rna_test_labels)
rna_train_dataloader = DataLoader(rna_train_dataset, batch_size = 128, shuffle = True)
rna_test_dataloader = DataLoader(rna_test_dataset, batch_size = 128, shuffle = False)

#pos weight自己定记得加
hidden_dim1 = 1024
hidden_dim2 = 512
hidden_dim3 = 256
z_dim = 64


total_epochs = 200
warmup_epochs = 20

batch_size = 128

lr_vae = 1e-3

rna_kl_weight = 1
atac_kl_weight = 1
mmd_weight = 1

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

    
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim2, hidden_dim1, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, output_dim),
            #nn.Sigmoid()  # Sigmoid for RNA, Logits for ATAC
        )

    def forward(self, z):
        return self.decoder(z)
    

class VAE(nn.Module):
    def __init__(self, rna_input_dim,atac_input_dim, hidden_dim1, hidden_dim2,hidden_dim3,z_dim):
        super().__init__()
        self.encoder_rna = Encoder(rna_input_dim, hidden_dim1, hidden_dim2,hidden_dim3)
        self.encoder_atac = Encoder(atac_input_dim, hidden_dim1, hidden_dim2, hidden_dim3)
        self.fc_mu = nn.Linear(hidden_dim3, z_dim)
        self.fc_log_var = nn.Linear(hidden_dim3, z_dim)
        self.decoder_rna = Decoder(z_dim, hidden_dim2, hidden_dim1, rna_input_dim)
        self.decoder_atac = Decoder(z_dim, hidden_dim2, hidden_dim1, atac_input_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode_rna(self, x_rna):
        hidden_rna = self.encoder_rna(x_rna)
        mu_rna = self.fc_mu(hidden_rna)
        log_var_rna = self.fc_log_var(hidden_rna)
        return mu_rna, log_var_rna
    
    def encode_atac(self, x_atac):
        hidden_atac = self.encoder_atac(x_atac)
        mu_atac = self.fc_mu(hidden_atac)
        log_var_atac = self.fc_log_var(hidden_atac)
        return mu_atac, log_var_atac
    
    def forward_rna(self, x_rna):
        mu_rna, log_var_rna = self.encode_rna(x_rna)
        z_rna = self.reparameterize(mu_rna, log_var_rna)
        x_rna_recon = F.sigmoid(self.decoder_rna(z_rna))
        return x_rna_recon, mu_rna, log_var_rna
    
    def forward_atac(self, x_atac):
        mu_atac, log_var_atac = self.encode_atac(x_atac)
        z_atac = self.reparameterize(mu_atac, log_var_atac)
        x_atac_recon = self.decoder_atac(z_atac)
        return x_atac_recon, mu_atac, log_var_atac
    
    def get_embeddings(self, x_rna, x_atac):
        mu_rna, _ = self.encode_rna(x_rna)
        mu_atac, _ = self.encode_atac(x_atac)
        return mu_rna, mu_atac
    

num_positives = atac_train.sum()
total_elements = atac_train.shape[0] * atac_train.shape[1]
num_negatives = total_elements - num_positives
pos_weight_value = num_negatives / num_positives
print(f"Calculated pos_weight: {pos_weight_value:.4f}")
pos_weight_tensor = torch.tensor(pos_weight_value, device=device)


def vae_loss_rna(x_rna, x_rna_recon, mu_rna, log_var_rna):
    recon_loss = F.mse_loss(x_rna_recon, x_rna, reduction='mean')#这里用sum或者mean
    kl_div = -0.5 * torch.mean(1 + log_var_rna - mu_rna.pow(2) - log_var_rna.exp())
    vae_loss_rna = recon_loss + kl_div * rna_kl_weight
    return vae_loss_rna

def vae_loss_atac(x_atac, x_atac_recon, mu_atac, log_var_atac):
    recon_loss =  F.binary_cross_entropy_with_logits(x_atac_recon, x_atac, reduction='mean', pos_weight=pos_weight_tensor)
    kl_div = -0.5 * torch.mean(1 + log_var_atac - mu_atac.pow(2) - log_var_atac.exp())
    vae_loss_atac = recon_loss + kl_div * atac_kl_weight
    return vae_loss_atac


model_vae = VAE(X_rna.shape[1], X_atac.shape[1], hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, hidden_dim3 = hidden_dim3, z_dim=z_dim)
model_vae.to(device)

optim_vae = torch.optim.AdamW(model_vae.parameters(), lr=lr_vae )
#Warmup + CosineAnnealingLR

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optim_vae, start_factor=0.1, total_iters=warmup_epochs
    )
main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim_vae, T_max=(total_epochs - warmup_epochs), eta_min=5e-5
    )
scheduler_vae = torch.optim.lr_scheduler.SequentialLR(
    optim_vae,
    schedulers=[warmup_scheduler, main_scheduler],
    milestones=[warmup_epochs]
)

import itertools
import matplotlib.pyplot as plt
history = {key: [] for key in ["total", "rna", "atac", "mmd"]}

print("Starting model training...")
for epoch in range(total_epochs):
    model_vae.train()
    epoch_losses = {key: 0.0 for key in history}
    
    rna_loader_iter = itertools.cycle(rna_train_dataloader)
    atac_loader_iter = itertools.cycle(atac_train_dataloader)
    num_batches = max(len(rna_train_dataloader), len(atac_train_dataloader))

    for _ in range(num_batches):
        x_rna, _ = next(rna_loader_iter)
        x_atac, _ = next(atac_loader_iter)
        x_rna, x_atac = x_rna.to(device), x_atac.to(device)

        x_rna_recon, mu_rna, log_var_rna = model_vae.forward_rna(x_rna)
        l_r = vae_loss_rna(x_rna, x_rna_recon, mu_rna, log_var_rna)
        
        x_atac_recon, mu_atac, log_var_atac = model_vae.forward_atac(x_atac)
        l_a = vae_loss_atac(x_atac, x_atac_recon, mu_atac, log_var_atac)

        mmd_l = mmd_loss_latent(mu_rna, mu_atac, device)
        
        #total_loss = (l_r * (l_a.item() / (l_r.item() + 1e-8))) + l_a + (mmd_weight * mmd_l)
        total_loss = (l_r * (l_a.item() / (l_r.item() + 1e-8))) + l_a + (mmd_weight * mmd_l)
        
        optim_vae.zero_grad()
        total_loss.backward()
        optim_vae.step()

        epoch_losses["total"] += total_loss.item()
        epoch_losses["rna"] += l_r.item()
        epoch_losses["atac"] += l_a.item()
        epoch_losses["mmd"] += mmd_l.item()
    
    scheduler_vae.step()

    for key in history:
        history[key].append(epoch_losses[key] / num_batches)
    
    print(
        f"Epoch [{epoch+1}/{total_epochs}] | "
        f"Total: {history['total'][-1]:.4f} | "
        f"RNA: {history['rna'][-1]:.4f} | "
        f"ATAC: {history['atac'][-1]:.4f} | "
        f"MMD: {history['mmd'][-1]:.4f}"
    )

print("\nTraining finished!")

plt.figure(figsize=(10, 6))
plt.plot(history["total"], label='Total Loss', color='black', lw=2)
plt.plot(history["rna"], label='RNA VAE Loss', ls='--')
plt.plot(history["atac"], label='ATAC VAE Loss', ls='--')
plt.plot(history["mmd"], label='MMD Alignment Loss', ls=':')
plt.title('Training Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

model_vae.eval()
X_rna = torch.tensor(X_rna, dtype=torch.float32).to(device)
X_atac = torch.tensor(X_atac, dtype=torch.float32).to(device)
with torch.no_grad():
    rna_embeddings, atac_embeddings = model_vae.get_embeddings(X_rna,X_atac)
adata_atac.obsm["embedding"] = atac_embeddings.cpu().numpy()
adata_rna.obsm["embedding"] = rna_embeddings.cpu().numpy()

sc.pp.neighbors(adata_atac, n_neighbors=15, use_rep='embedding')
sc.tl.leiden(adata_atac, key_added='leiden_clusters')

sc.pp.neighbors(adata_rna, n_neighbors=15, use_rep='embedding')
sc.tl.leiden(adata_rna, key_added='leiden_clusters')

atac_labels_true = adata_atac.obs["cell_type"]
atac_labels_pred = adata_atac.obs["leiden_clusters"]
nmi_atac = normalized_mutual_info_score(atac_labels_true, atac_labels_pred)
ari_atac = adjusted_rand_score(atac_labels_true, atac_labels_pred)
ami_atac = adjusted_mutual_info_score(atac_labels_true, atac_labels_pred)
homo_atac = homogeneity_score(atac_labels_true, atac_labels_pred)
print(f"ATAC.  NMI: {nmi_atac:.4f}, ARI: {ari_atac:.4f}, AMI: {ami_atac:.4f}, Homo: {homo_atac:.4f}")

rna_labels_true = adata_rna.obs["cell_type"]
rna_labels_pred = adata_rna.obs["leiden_clusters"]
nmi_rna = normalized_mutual_info_score(rna_labels_true, rna_labels_pred)
ari_rna = adjusted_rand_score(rna_labels_true, rna_labels_pred)
ami_rna = adjusted_mutual_info_score(rna_labels_true, rna_labels_pred)
homo_rna = homogeneity_score(rna_labels_true, rna_labels_pred)
print(f"RNA.  NMI: {nmi_rna:.4f}, ARI: {ari_rna:.4f}, AMI: {ami_rna:.4f}, Homo: {homo_rna:.4f}")


# 在合并之前，分别为两个 anndata 对象添加 'domain' 列
adata_rna.obs['domain'] = 'RNA'
adata_atac.obs['domain'] = 'ATAC'
combined = ad.concat([adata_rna, adata_atac])
sc.pp.neighbors(combined, use_rep="embedding", metric="cosine")
sc.tl.umap(combined)
sc.pl.umap(combined, color=["cell_type"], wspace=0.3)
sc.pl.umap(combined, color=[ "domain"], wspace=0.3)
