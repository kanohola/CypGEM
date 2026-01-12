import os, math, time, json, copy, random, pickle, csv
from dataclasses import dataclass, field, fields, is_dataclass 
from typing import List, Optional, Dict, Any, Iterator, Set, Tuple
from pathlib import Path
from collections import deque
from tqdm import tqdm
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import global_mean_pool, global_max_pool
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix, dropout_edge, softmax
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.norm import BatchNorm as PGBatchNorm, LayerNorm as PGLayerNorm, GraphNorm
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, MaxAggregation

from scipy.sparse.csgraph import shortest_path

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    matthews_corrcoef, jaccard_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings("ignore", message=".*torch-scatter.*")
# ------------------------- Global Configuration -------------------------
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED); random.seed(GLOBAL_SEED)


NODE_IN_DIM       = 47
EXPECTED_EDGE_DIM = 15

# ------------------------- Loss Function -------------------------
def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    pos_weight: torch.Tensor
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1 - p).clamp(1e-6, 1 - 1e-6)
    focal_weight = (1 - pt) ** gamma
    per_node_focal_loss = focal_weight * bce
    weight_mask = torch.where(targets > 0.5, pos_weight, torch.ones_like(targets))
    final_loss_per_node = per_node_focal_loss * weight_mask
    return final_loss_per_node.mean()

# ------------------------- Data Function -------------------------
def sanitize_dataset(lst: List[Data], name: str, expected_edge_dim: int) -> List[Data]:
    keep, drop = [], 0
    for i, d in enumerate(lst):
        try:
            if hasattr(d, 'dist_3d'):
                del d.dist_3d 
            

            n_nodes = d.x.size(0); d.y = torch.as_tensor(d.y).view(-1).to(torch.long)
            if d.x.dtype != torch.float32: d.x = d.x.to(torch.float32)
            if d.x.size(-1) != NODE_IN_DIM: 
                raise ValueError(f"x dim {d.x.size(-1)} != {NODE_IN_DIM}")
                
            if not isinstance(d.edge_index, torch.Tensor): d.edge_index = torch.as_tensor(d.edge_index, dtype=torch.long)
            E = int(d.edge_index.size(1)); ea = getattr(d, "edge_attr", None)
            
            if ea is None: d.edge_attr = torch.zeros((E, expected_edge_dim), dtype=torch.float32)
            else:
                if ea.dtype != torch.float32: ea = ea.to(torch.float32)
                rows, feat_dim = int(ea.size(0)), int(ea.size(1))
                if rows != E:
                    raise ValueError(f"edge_attr rows {rows} != edge_index E {E}")
                if feat_dim != expected_edge_dim:
                    raise ValueError(f"edge_attr dim {feat_dim} != {expected_edge_dim}")

                d.edge_attr = ea.contiguous()
            
            if not hasattr(d, 'spd') or d.spd.size(0) != E:
                raise ValueError(f"spd size {d.spd.size(0) if hasattr(d, 'spd') else 'N/A'} != edge_index E {E}")
                
            keep.append(d)
        except Exception as e: 
            print(f"[{name}] drop idx={i}: {e}", flush=True); 
            drop += 1
            
    if drop: 
        print(f"[{name}] kept={len(keep)} dropped={drop}", flush=True)
    return keep

def compute_class_pos_weight(datalist: List[Data]) -> float:
    pos=neg=0
    for d in datalist: 
        y=torch.as_tensor(d.y).view(-1).to(torch.long)
        pos+=int((y==1).sum().item())
        neg+=int((y==0).sum().item())
    return float(max(neg,1)/max(pos,1))

class DropPath(nn.Module):
    def __init__(self, p: float = 0.0): super().__init__(); self.p = p
    def forward(self, x):
        if self.p==0. or not self.training: return x
        keep=1-self.p; shape=(x.shape[0],)+(1,)*(x.ndim-1); rand=keep+torch.rand(shape,dtype=x.dtype,device=x.device)
        return x.div(keep)*torch.floor(rand)

class MoreConfidentReadout(nn.Module):
    def __init__(self, d_e, cls_hidden, dropout):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_e * 2, cls_hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden * 2, cls_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, 1)
        )
    
    def forward(self, x):
        return self.head(x).view(-1)

# ------------------------- Model Components -------------------------
@dataclass
class EvalResult:
    roc_auc: float; auprc: float; mcc: float; jacc: float; prec: float; rec: float
    top1: float; top1_all: float
    top2: float; top2_all: float
    top3: float; top3_all: float
    thr: float 

class EarlyStopperAUC:
    def __init__(self, patience=20, warmup=12, alpha=0.25, min_delta=0.003):
        self.patience,self.warmup,self.alpha,self.min_delta=patience,warmup,alpha,min_delta
        self.best_smooth,self.bad_epochs = 0.5, 0
        self.smooth_prev,self.buf=None,deque(maxlen=7)
    
    def step(self, epoch_idx, raw_auc):
        self.buf.append(float(raw_auc)); improved=False
        smooth=raw_auc if self.smooth_prev is None else self.alpha*raw_auc+(1-self.alpha)*self.smooth_prev
        self.smooth_prev=smooth
        thr=max(self.min_delta,0.5*np.std(self.buf) if len(self.buf)>=2 else 0.0)
        
        if epoch_idx<=self.warmup:
            if smooth>self.best_smooth: self.best_smooth=smooth
        elif smooth>=self.best_smooth+thr: 
            self.best_smooth,self.bad_epochs,improved=smooth,0,True
        else: 
            self.bad_epochs+=1
        return improved,(self.bad_epochs>=self.patience)

class EMAWrapper:
    def __init__(self,model,decay=0.995): self.decay,self.ema_model=decay,copy.deepcopy(model); [p.requires_grad_(False) for p in self.ema_model.parameters()]
    def to(self,device): self.ema_model.to(device)
    @torch.no_grad()
    def update(self,src_model):
        d=self.decay
        for (_,p_e),(_,p_s) in zip(self.ema_model.named_parameters(),src_model.named_parameters()):
            p_e.data.mul_(d).add_(p_s.data,alpha=1.0-d)
        for (_,b_e),(_,b_s) in zip(self.ema_model.named_buffers(),src_model.named_buffers()):
            if b_e.data.dtype.is_floating_point: b_e.data.mul_(d).add_(b_s.data,alpha=1.0-d)
            else: b_e.data.copy_(b_s.data)
    def eval_model(self): return self.ema_model

def _per_graph_slices(batch):
    if getattr(batch,"batch",None) is None: return [slice(0,batch.num_nodes)]
    b=batch.batch.cpu().numpy(); splits=[]; start=0
    for gid in range(int(b.max())+1): 
        end=start+int((b==gid).sum())
        splits.append(slice(start,end))
        start=end
    return splits

def _compute_topk(scores, y_true, graph_slices, k_values=[1, 2, 3]):
    results = {}
    ng = len(graph_slices)
    if ng == 0:
        for k in k_values:
            results[f'top{k}'] = 0.0
            results[f'top{k}_all'] = 0.0
        return results
        
    hits = {k: 0 for k in k_values}
    sel_trues = {k: 0 for k in k_values}
    sel_tots = {k: 0 for k in k_values}
    
    for sl in graph_slices:
        p, y = scores[sl], y_true[sl]
        if p.size == 0:
            continue
            
        
        top_indices = np.argsort(p)[::-1]
       

        for k in k_values:
            k_eff = min(k, p.size)
            if k_eff <= 0:
                continue
            
            current_top_indices = top_indices[:k_eff]
            chosen = (y[current_top_indices] == 1)
            
            hits[k] += int(chosen.any())
            sel_trues[k] += int(chosen.sum())
            sel_tots[k] += k_eff
            
    for k in k_values:
        results[f'top{k}'] = hits[k] / ng
        results[f'top{k}_all'] = sel_trues[k] / max(sel_tots[k], 1)
        
    return results

@torch.no_grad()
def evaluate(model, loader, thr=None, device=None):
    model.eval()
    Z_raw, Y, S = [], [], []
    
    for batch in loader:
        batch=batch.to(device,non_blocking=True)
        logits = model.forward_logits(batch)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        Z_raw.append(logits.detach().cpu())
        Y.append(batch.y.view(-1).cpu())
        S.extend(_per_graph_slices(batch))

    z_raw=torch.cat(Z_raw)
    labels=torch.cat(Y).numpy().astype(int)
    probs=torch.sigmoid(z_raw).cpu().numpy()

    if thr is None:
        cand=np.unique(np.clip(probs,1e-6,1-1e-6)); best_thr,best_mcc=0.5,-1
        if cand.size>2000: cand=np.quantile(cand,np.linspace(0,1,2001))
        for t in cand:
            pred=(probs>=t).astype(int)
            if len(np.unique(pred))<2: continue
            if (m:=matthews_corrcoef(labels,pred))>best_mcc: best_mcc,best_thr=m,float(t)
        thr=best_thr
    
    preds=(probs>=thr).astype(int)

    roc_auc=roc_auc_score(labels,probs) if len(np.unique(labels))>1 else float("nan")
    auprc=average_precision_score(labels,probs) if len(np.unique(labels))>1 else float("nan")
    mcc=matthews_corrcoef(labels,preds) if len(np.unique(preds))>1 else 0.0
    jacc=jaccard_score(labels,preds,zero_division=0)
    prec=precision_score(labels,preds,zero_division=0)
    rec=recall_score(labels,preds,zero_division=0)
    
    scores_for_rank = z_raw.cpu().numpy()
    topk_results = _compute_topk(scores_for_rank, labels, S, k_values=[1, 2, 3])

    return EvalResult(
        roc_auc=roc_auc, auprc=auprc, mcc=mcc, jacc=jacc, prec=prec, rec=rec,
        top1=topk_results['top1'], top1_all=topk_results['top1_all'],
        top2=topk_results['top2'], top2_all=topk_results['top2_all'],
        top3=topk_results['top3'], top3_all=topk_results['top3_all'],
        thr=thr
    ), probs

class NodeProjector(nn.Module):
    def __init__(self, in_dim, hidden, feat_drop): super().__init__(); self.proj=nn.Sequential(nn.Linear(in_dim,hidden),nn.ReLU(),nn.BatchNorm1d(hidden),nn.Dropout(feat_drop))
    def forward(self, x): return self.proj(x)
class EdgeProjector(nn.Module):
    def __init__(self, in_dim, d_e, feat_drop): super().__init__(); self.proj=nn.Sequential(nn.Linear(in_dim,d_e),nn.ReLU(),nn.BatchNorm1d(d_e),nn.Dropout(feat_drop))
    def forward(self, e): return self.proj(e)
class EdgeFusion(nn.Module):
    def __init__(self, d_e):
        super().__init__(); self.gate_mlp=nn.Sequential(nn.Linear(d_e*2,d_e),nn.Sigmoid()); self.norm=nn.LayerNorm(d_e)
    def forward(self, handcraft_feat, spd_feat):
        gate=self.gate_mlp(torch.cat([handcraft_feat,spd_feat],dim=-1)); fused=gate*handcraft_feat+(1.0-gate)*spd_feat; return self.norm(fused)

class EdgeUpdate(nn.Module):
    def __init__(self, node_dim, edge_dim, use_norm=True): 
        super().__init__()
        self.node_norm = PGLayerNorm(node_dim, mode="node")
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, edge_dim * 2),
            nn.ReLU()
        )
        self.edge_norm = nn.LayerNorm(edge_dim) if use_norm else nn.Identity()

    def forward(self, x, edge_index, edge_attr): 
        src, dst = edge_index
        xn = self.node_norm(x)
        m = torch.cat([xn[src], xn[dst], edge_attr], dim=-1)
        raw_output = self.mlp(m)
        delta, gate_logit = torch.chunk(raw_output, 2, dim=-1)
        gate = torch.sigmoid(gate_logit)
        out = edge_attr + gate * delta 
        return self.edge_norm(out)

class NodeEdgeCrossAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.d_head = dim // heads
        self.scale = self.d_head ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.aggr = SumAggregation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_nodes, k_edges, v_edges, edge_index):
        N = q_nodes.size(0)
        E = k_edges.size(0)
        dst, src = edge_index
        
        if E == 0:
            return torch.zeros((N, self.dim), device=q_nodes.device)
        
        q = self.q_proj(q_nodes).view(N, self.heads, self.d_head)
        k = self.k_proj(k_edges).view(E, self.heads, self.d_head)
        v = self.v_proj(v_edges).view(E, self.heads, self.d_head)

        q_i = q[dst]
        k_e = k
        
        attn_score = (q_i * k_e).sum(dim=-1) * self.scale
        attn_weights = softmax(attn_score, dst, num_nodes=N)
        attn_weights = self.dropout(attn_weights)
        
        weighted_v = v * attn_weights.unsqueeze(-1)
        weighted_v = weighted_v.view(E, self.dim)
        
        aggr_out = self.aggr(weighted_v, dst, dim_size=N)
        return self.out_proj(aggr_out)

class GraphormerBlock(nn.Module):
    def __init__(self, dim, edge_dim, heads, dropout, norm_type, droppath, max_degree):
        super().__init__(); assert dim%heads==0; self.norm=make_norm(norm_type,dim)
        self.attn=TransformerConv(dim,dim//heads,heads,edge_dim=edge_dim,dropout=dropout)
        self.ffn=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*4,dim),nn.Dropout(dropout))
        self.stoch=DropPath(droppath); self.max_deg=int(max_degree); self.deg_emb=nn.Embedding(self.max_deg,dim)
        self.edge_upd = EdgeUpdate(node_dim=dim, edge_dim=edge_dim, use_norm=True)
    def _degree_embed(self, num_nodes, edge_index, device):
        deg=torch.zeros(num_nodes,dtype=torch.long,device=device).scatter_add_(0,edge_index[0],torch.ones_like(edge_index[0]))
        return self.deg_emb(deg.clamp_max(self.max_deg-1))
    def forward(self, x, edge_index, edge_attr, norm_type):
        x_in = x + self._degree_embed(x.size(0), edge_index, x.device)
        h = self.norm(x_in)
        h_attn = self.attn(h, edge_index, edge_attr) 
        x_attn_res = x_in + self.stoch(h_attn)
        h_ffn = self.norm(x_attn_res) if norm_type in ["layer","batch"] else x_attn_res
        h_ffn_out = self.ffn(h_ffn)
        x_new = x_attn_res + self.stoch(h_ffn_out)
        ea_new = self.edge_upd(x_new, edge_index, edge_attr)
        return x_new, ea_new

# ------------------------- Global Transformer Layer with Bias -------------------------
class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=20.0, num_gaussians=32):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1))**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        # dist: [Batch, N, N] -> [Batch, N, N, num_gaussians]
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))

class GlobalTransformerLayer(nn.Module):
    def __init__(self, dim, heads, max_dist=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout)
        
        self.dist_bias = nn.Embedding(max_dist, heads)
        
        self.gaussian = GaussianSmearing(start=0.0, stop=20.0, num_gaussians=32)
        self.rbf_proj = nn.Linear(32, heads)
        
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.num_heads = heads

    
    def forward(self, x, batch, dist_matrix, dist_3d=None):
        """
        x: [Batch*N, Dim]
        batch: [Batch*N]
        dist_matrix: [Batch, Max_N, Max_N] 
        dist_3d: [Batch, Max_N, Max_N] 
        """
        x_dense, mask = to_dense_batch(x, batch)
        
        
        batch_size, max_n = x_dense.size(0), x_dense.size(1)
        
        
        dist_clamped = dist_matrix.clamp(0, self.dist_bias.num_embeddings - 1).long()
        bias_spd = self.dist_bias(dist_clamped).permute(0, 3, 1, 2)
        
        attn_bias = bias_spd
        
        if dist_3d is not None:
            
            rbf_feat = self.gaussian(dist_3d)
            
            bias_3d = self.rbf_proj(rbf_feat).permute(0, 3, 1, 2)
            
            
            attn_bias = attn_bias + bias_3d

        
        attn_bias = attn_bias.reshape(batch_size * self.num_heads, max_n, max_n)
        
        
        padding_mask = ~mask 
        padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_heads, max_n, max_n)
        padding_mask_expanded = padding_mask_expanded.reshape(batch_size * self.num_heads, max_n, max_n)
        
        
        attn_bias = attn_bias.masked_fill(padding_mask_expanded, float("-inf"))
        
        
        h = self.norm1(x_dense)
        
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_bias)
        
        h = h + attn_out
        h = h + self.ffn(self.norm2(h))
        
        x_out = h[mask]
        return x_out

class GraphormerModel(nn.Module):
    def __init__(self, node_in, edge_in, hidden, heads, n_layers, 
                 attn_heads=4, cls_hidden=128, dropout=0.3, d_e=128, norm_type="batch", 
                 droppath=0.1, feat_drop=0.2, max_degree=32, max_spd=64, drop_edge_p=0.0, **kwargs):
        super().__init__(); assert hidden % heads == 0; self.max_spd=max_spd; self.drop_edge_p=drop_edge_p
        
        self.node_proj=NodeProjector(node_in,hidden,feat_drop)
        self.edge_proj=EdgeProjector(edge_in,d_e,feat_drop)
        self.spatial_emb=nn.Embedding(max_spd, d_e)
        self.edge_fusion=EdgeFusion(d_e)
        self.layers=nn.ModuleList([GraphormerBlock(hidden,d_e,heads,dropout,norm_type,droppath*(i+1)/max(n_layers,1),max_degree) for i in range(n_layers)])
        
        self.global_layer = GlobalTransformerLayer(dim=hidden, heads=heads, max_dist=max_spd, dropout=dropout)
        
        self.edge_aggr = NodeEdgeCrossAttention(d_e, heads=attn_heads, dropout=dropout)
        self.d_e = d_e
        self.drop=nn.Dropout(dropout)
        
        self.node_readout_proj = nn.Sequential(
            nn.Linear(hidden * 2, d_e), 
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.global_pool_proj = nn.Sequential(
            nn.Linear(hidden * 2, d_e),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.global_edge_proj = nn.Sequential(
            nn.Linear(d_e, d_e),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        input_dim = int(4 * d_e) 
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, cls_hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden * 2, cls_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, 1)
        )
    
    def forward_logits(self, data):
        x, ei, ea, spd = data.x, data.edge_index, data.edge_attr, data.spd
        
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if self.training and self.drop_edge_p > 0:
            ei, edge_mask = dropout_edge(ei, p=self.drop_edge_p, force_undirected=False)
            ea = ea[edge_mask]
            spd = spd[edge_mask]
            
        x = self.node_proj(x)
        ea_handcraft = self.edge_proj(ea)
        spd_embedding = self.spatial_emb(spd.clamp_max(self.max_spd-1))
        
        ea = self.edge_fusion(ea_handcraft, spd_embedding)
        
        x0 = x.clone()
        x_history = []
        ea_history = []
        
        for blk in self.layers: 
            x, ea = blk(x, ei, ea, norm_type="layer")
            x_history.append(x)
            ea_history.append(ea)

        n_layers = len(x_history)
        if n_layers == 0: x_combined = x
        elif n_layers == 1: x_combined = x_history[-1]
        elif n_layers == 2: x_combined = x_history[-1] + x_history[-2]
        else: x_combined = x_history[-1] + x_history[-2] + x_history[-3]

        n_edge_layers = len(ea_history)
        if n_edge_layers == 0: ea_combined = ea
        elif n_edge_layers == 1: ea_combined = ea_history[-1]
        elif n_edge_layers == 2: ea_combined = ea_history[-1] + ea_history[-2]
        else: ea_combined = ea_history[-1] + ea_history[-2] + ea_history[-3]
        

        dist_dense_spd = to_dense_adj(ei, batch, edge_attr=spd, max_num_nodes=None)
        
        pos_dense, mask = to_dense_batch(data.pos, batch)
        
        dist_3d_batch = torch.cdist(pos_dense, pos_dense, p=2.0)
        
        x_global_refined = self.global_layer(x_combined, batch, dist_dense_spd, dist_3d=dist_3d_batch)
        x_combined = x_combined + x_global_refined 
        
        h_node = torch.cat([x_combined, x0], dim=-1)
        h_node_proj = self.node_readout_proj(h_node) 

        ea_smart_aggr = self.edge_aggr(
            q_nodes=h_node_proj, 
            k_edges=ea_combined,
            v_edges=ea_combined,
            edge_index=ei
        ) 

        g_mean = global_mean_pool(x_combined, batch) 
        g_max = global_max_pool(x_combined, batch)   
        g_feat = torch.cat([g_mean, g_max], dim=-1)  
        g_feat_proj = self.global_pool_proj(g_feat)  
        g_feat_lifted = g_feat_proj[batch]           

        edge_batch = batch[ei[0]] 
        g_edge_mean = global_mean_pool(ea_combined, edge_batch) 
        g_edge_proj = self.global_edge_proj(g_edge_mean)       
        g_edge_lifted = g_edge_proj[batch]                     

        x_out = torch.cat([h_node_proj, ea_smart_aggr, g_feat_lifted, g_edge_lifted], dim=-1)
        
        h_dropped = self.drop(x_out)
        logits = self.head(h_dropped).view(-1)

        if self.training:
            return logits, x_combined
        else:
            return logits, None

def load_pickle_list(path):
    with open(path,"rb") as f: obj=pickle.load(f)
    if isinstance(obj,list): return obj
    if isinstance(obj,dict):
        for k in ["data_list","dataset","data"]:
            if k in obj and isinstance(obj[k],list): return obj[k]
    if isinstance(obj,Data): return [obj]
    return list(obj)

def make_norm(norm_type: str, dim: int):
    t=(norm_type or "none").lower()
    if t=="batch": return PGBatchNorm(dim)
    if t=="layer": return PGLayerNorm(dim, mode="node")
    return nn.Identity()

# ------------------------- Training Function -------------------------
def train_one_run(cfg, train_loader, valid_loader, test_loader, external_loader, pos_weight, device, save_dir):
    
    model_cfg = {k:v for k,v in cfg.items() if k in [
        "hidden","heads","n_layers","cls_hidden","dropout",
        "norm_type","droppath","feat_drop","max_degree","max_spd",
        "drop_edge_p", "attn_heads", "d_e"
    ]}
    
    model=GraphormerModel(node_in=NODE_IN_DIM, edge_in=EXPECTED_EDGE_DIM, **model_cfg).to(device)
    
    decay,no_decay=[],[]
    for n,p in model.named_parameters():
        if p.requires_grad:
            if any(k in n for k in ["norm","bn","bias","deg_emb","spatial_emb", "temperature"]): no_decay.append(p)
            else: decay.append(p)
    optimizer=torch.optim.AdamW([
        {"params":decay,"weight_decay":cfg['weight_decay']},
        {"params":no_decay,"weight_decay":0.0}
    ], lr=cfg['lr'], betas=(0.9,0.95))
    
    def lr_lambda(epoch):
        if epoch<cfg['warmup_epochs']: return (epoch+1)/max(1,cfg['warmup_epochs'])
        t=(epoch-cfg['warmup_epochs'])/max(1,cfg['max_epochs']-cfg['warmup_epochs'])
        return cfg['cosine_min_lr']/cfg['lr']+0.5*(1-cfg['cosine_min_lr']/cfg['lr'])*(1+math.cos(math.pi*t))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_lambda)
    
    ema=EMAWrapper(model,decay=cfg['ema_decay']) if cfg['use_ema'] else None
    if ema: ema.to(device)
    
    stopper=EarlyStopperAUC(patience=cfg['patience'],warmup=cfg['es_warmup'])
    
    
    best_state = None
    best_val_auc = 0.5
    fixed_thr = None       
    best_thr_to_save = 0.5 

    print("\n" + "="*60)
    print("Start Training...")
    print("="*60)
    
    for epoch in range(1,cfg['max_epochs']+1):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            batch=batch.to(device,non_blocking=True)
            logits, _ = model.forward_logits(batch)
            y_float=batch.y.view(-1).float()
            
            pos_mask = (y_float == 1)
            neg_mask = (y_float == 0)
            num_pos = pos_mask.sum().item()
            
            if num_pos == 0:
                continue
            
            num_neg_to_keep = min(neg_mask.sum().item(), max(1, int(num_pos * cfg['target_pos_neg_ratio'])))
            neg_indices = neg_mask.nonzero(as_tuple=False).view(-1)
            keep_indices = neg_indices[torch.randperm(neg_indices.size(0))[:num_neg_to_keep]]
            
            mask = pos_mask.clone()
            mask[keep_indices] = True
            
            masked_logits = logits[mask]
            masked_targets = y_float[mask]

            if cfg['label_smoothing'] > 0: 
                masked_targets = masked_targets*(1-cfg['label_smoothing'])+0.5*cfg['label_smoothing']
            
            loss = focal_bce_loss(
                masked_logits, masked_targets,
                gamma=cfg['focal_gamma'],
                pos_weight=pos_weight
            )
            
            if loss > 0:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
                optimizer.step()
                if ema and (epoch>cfg['ema_warmup_epochs']): ema.update(model)
                
                epoch_loss += loss.item()
                num_batches += 1
        
        scheduler.step()
        eval_model=ema.eval_model() if (ema and epoch>cfg['ema_warmup_epochs']) else model
        
        if cfg['thr_lock_after'] and epoch==cfg['thr_lock_after']: 
            lock_res,_=evaluate(eval_model,valid_loader,thr=None,device=device)
            fixed_thr=float(lock_res.thr)
            print(f"  >> Epoch {epoch:3d} | threshold lock @ {fixed_thr:.3f}")
        
        dyn_res,_=evaluate(eval_model,valid_loader,thr=fixed_thr,device=device)
        cur_auc = dyn_res.roc_auc
        improved,should_stop=stopper.step(epoch, cur_auc)
        
        if epoch % 10 == 0 or improved:
            print(f"  >> Epoch {epoch:3d} | Loss: {epoch_loss/num_batches:.4f} | Val AUC: {cur_auc:.4f} | "
                  f"Best: {stopper.best_smooth:.4f} | Bad Epochs: {stopper.bad_epochs}")
        
        if improved or (cur_auc > best_val_auc and epoch<=stopper.warmup):
            best_val_auc = cur_auc
            src_state=model.state_dict()
            best_state={k:v.cpu().clone() for k,v in src_state.items()}
            
            best_thr_to_save = float(dyn_res.thr)
        # ========================================================
            
        if should_stop:
            print(f"  >> Early stop trigger @ Epoch {epoch}")
            break
    
    if best_state:
        model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    
    save_path = save_dir / "best_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'val_auc': best_val_auc,
        'threshold': best_thr_to_save 
    }, save_path)
    print(f"\n  >> The best model has been saved to: {save_path}")
    print(f"  >> Corresponding threshold: {best_thr_to_save:.4f}")
    # ========================================================
    def full_evaluate(model, loader, split_name, thr=None):
        print(f"\n  >> {split_name} Evaluating...")
        res, probs = evaluate(model, loader, thr=thr, device=device)
        print(f"     AUC: {res.roc_auc:.4f} | AUPRC: {res.auprc:.4f} | MCC: {res.mcc:.4f}")
        print(f"     Top1: {res.top1:.4f} | Top2: {res.top2:.4f} | Top3: {res.top3:.4f}")
        print(f"     Precision: {res.prec:.4f} | Recall: {res.rec:.4f} | Jaccard: {res.jacc:.4f}")
        print(f"     Threshold: {res.thr:.3f}")
        return res, probs
    
    val_res, _ = full_evaluate(model, valid_loader, "Validation set", thr=best_thr_to_save)

    if test_loader is not None:
        test_res, _ = full_evaluate(model, test_loader, "Test set", thr=best_thr_to_save)
    else:
        test_res = EvalResult(0,0,0,0,0,0,0,0,0,0,0,0,best_thr_to_save)
    
    if external_loader:
        external_res, external_probs = full_evaluate(model, external_loader, "External test set", thr=best_thr_to_save)
    else:
        external_res = EvalResult(0,0,0,0,0,0,0,0,0,0,0,0,best_thr_to_save)
    # =======================================================================
    

    metrics = {
        "val_top1":val_res.top1, "val_top1_all":val_res.top1_all,
        "val_top2":val_res.top2, "val_top2_all":val_res.top2_all,
        "val_top3":val_res.top3, "val_top3_all":val_res.top3_all,
        "val_auc":val_res.roc_auc, "val_ap":val_res.auprc, "val_mcc":val_res.mcc,
        "val_jacc":val_res.jacc, "val_prec":val_res.prec, "val_rec":val_res.rec, 
        "val_thr":val_res.thr,
        
        "test_top1":test_res.top1, "test_top1_all":test_res.top1_all,
        "test_top2":test_res.top2, "test_top2_all":test_res.top2_all,
        "test_top3":test_res.top3, "test_top3_all":test_res.top3_all,
        "test_auc":test_res.roc_auc, "test_ap":test_res.auprc, "test_mcc":test_res.mcc,
        "test_jacc":test_res.jacc, "test_prec":test_res.prec, "test_rec":test_res.rec, 
        "test_thr":test_res.thr,
        
        "external_top1":external_res.top1, "external_top1_all":external_res.top1_all,
        "external_top2":external_res.top2, "external_top2_all":external_res.top2_all,
        "external_top3":external_res.top3, "external_top3_all":external_res.top3_all,
        "external_auc":external_res.roc_auc, "external_ap":external_res.auprc, "external_mcc":external_res.mcc,
        "external_jacc":external_res.jacc, "external_prec":external_res.prec, "external_rec":external_res.rec, 
        "external_thr":external_res.thr,
    }
    
    return metrics, model

# ------------------------- Main Function -------------------------
DATA_DIR = Path("/data1/zyx/all")
TRAIN_PKL = DATA_DIR/"train.pkl"
VALID_PKL = DATA_DIR/"valid.pkl"
TEST_PKL = DATA_DIR/"test.pkl"

def main(external_test_path=None, gpu_id=1, save_dir="./model_save"):
    assert torch.cuda.is_available(),"CUDA not available"
    torch.cuda.set_device(gpu_id)
    device=torch.device(f"cuda:{gpu_id}")
    print("[CUDA]",torch.cuda.get_device_name(gpu_id),flush=True)
    
    # best config
    BEST_CONFIG = {
        'max_epochs': 160,
        'cosine_min_lr': 5e-07,
        'use_ema': True,
        'ema_warmup_epochs': 5,
        'patience': 10,
        'es_warmup': 12,
        'attn_heads': 4,
        'batch_size': 128,
        'cls_hidden': 128,
        'd_e': 128,
        'drop_edge_p': 0.0,
        'dropout': 0.3,
        'droppath': 0.3,  
        'ema_decay': 0.99,
        'feat_drop': 0.1,
        'focal_gamma': 3.0,
        'grad_clip': 1.5,
        'heads': 8,
        'hidden': 1024,
        'label_smoothing': 0.1,
        'lr': 3e-05,  
        'max_degree': 64,
        'max_spd': 32,
        'n_layers': 6,
        'norm_type': 'batch',
        'pos_weight': 20.0,
        'target_pos_neg_ratio': 2.0,
        'thr_lock_after': None,
        'warmup_epochs': 15,   
        'weight_decay': 0.0001  
    }
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Load dataset...")
    print("="*60)
    
    train_list=sanitize_dataset(load_pickle_list(TRAIN_PKL),"train",EXPECTED_EDGE_DIM)
    valid_list=sanitize_dataset(load_pickle_list(VALID_PKL),"valid",EXPECTED_EDGE_DIM)
    test_list=sanitize_dataset(load_pickle_list(TEST_PKL),"test",EXPECTED_EDGE_DIM)
    
    print(f"[Info] |train|={len(train_list)} |valid|={len(valid_list)} |test|={len(test_list)}",flush=True)
    
    pos_weight=torch.tensor([compute_class_pos_weight(train_list)],dtype=torch.float32,device=device)
    print(f"[Info] pos_weight = {pos_weight.item():.3f}",flush=True)
    
    dl_kw={"pin_memory":True,"persistent_workers":True}
    train_loader=DataLoader(train_list,batch_size=BEST_CONFIG['batch_size'],shuffle=True,num_workers=4,**dl_kw)
    valid_loader=DataLoader(valid_list,batch_size=BEST_CONFIG['batch_size'],shuffle=False,num_workers=2,**dl_kw)
    test_loader=DataLoader(test_list,batch_size=BEST_CONFIG['batch_size'],shuffle=False,num_workers=2,**dl_kw)
    
    external_loader = None
    if external_test_path and Path(external_test_path).exists():
        print(f"[Info] Loading external test set: {external_test_path}")
        external_list = sanitize_dataset(load_pickle_list(Path(external_test_path)),"external_test",EXPECTED_EDGE_DIM)
        print(f"[Info] |external_test|={len(external_list)}")
        external_loader=DataLoader(external_list,batch_size=BEST_CONFIG['batch_size'],shuffle=False,num_workers=2,**dl_kw)
    else:
        print(f"[Warning] If no external test suite path is provided or the file does not exist, external test suite evaluation will be skipped.")
        
        external_loader = None
    
    
    metrics, trained_model = train_one_run(
        BEST_CONFIG, train_loader, valid_loader, test_loader, external_loader,
        pos_weight, device, save_dir
    )
    
    results_path = save_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": BEST_CONFIG,
            "metrics": metrics
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("Training and assessment complete!")
    print("="*60)
    print(f"The model has been saved to: {save_dir}")
    print(f"Detailed results have been saved to: {results_path}")
    
    print("\n" + "="*60)
    print("Final performance summary")
    print("="*60)
    
    splits = ["val", "test", "external"]
    split_names = ["Validation Set", "Test Set", "External Test Set"]
    
    for split, name in zip(splits, split_names):
        if split == "external" and not external_loader:
            continue
            
        print(f"\n{name}:")
        print(f"  AUC: {metrics[f'{split}_auc']:.4f} | AUPRC: {metrics[f'{split}_ap']:.4f} | MCC: {metrics[f'{split}_mcc']:.4f}")
        print(f"  Top1: {metrics[f'{split}_top1']:.4f} | Top2: {metrics[f'{split}_top2']:.4f} | Top3: {metrics[f'{split}_top3']:.4f}")
        print(f"  Precision: {metrics[f'{split}_prec']:.4f} | Recall: {metrics[f'{split}_rec']:.4f} | Jaccard: {metrics[f'{split}_jacc']:.4f}")
        print(f"  Threshold: {metrics[f'{split}_thr']:.3f}")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--gpu",type=int,default=0,help="GPU device ID")
    p.add_argument("--external_test",type=str,default="",help="Path to external test suite pkl file")
    p.add_argument("--save_dir",type=str,default="",help="Model save directory")
    a=p.parse_args()
    main(external_test_path=a.external_test, gpu_id=a.gpu, save_dir=a.save_dir)