"""
=============================================================================
智链信航 · 贷前评估智能体 —— 基于注意力机制的信用评分模型
Credit Intelligence Engine: Attention-Based Credit Scoring Model

项目：多智能体协同的信贷全周期智能决策引擎
作者：智链信航项目团队 · 东北财经大学金融科技学院
创新点：引入特征级自注意力机制（Feature-Level Self-Attention），
        模拟多模态融合编码器的跨特征注意力交互，
        实现动态特征权重分配，提升信用风险评估精准度。
=============================================================================
"""

# ─── 标准库 ───────────────────────────────────────────────────────────────
import os
import random
import warnings

# ─── 第三方库 ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")   # 切换到非交互式后端，不需要图形界面
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# 0. 全局配置与随机种子固定
# ═══════════════════════════════════════════════════════════════════════════

# ---------- 中文字体配置（学术论文级）--------------------------------------
def setup_chinese_font():
    """
    自动检测系统可用中文字体并配置 Matplotlib，
    确保图表标题、标签、图例全部正常显示中文简体。
    """
    candidate_fonts = [
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Noto Sans CJK SC", "Noto Sans SC",
        "SimHei", "Microsoft YaHei", "PingFang SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((f for f in candidate_fonts if f in available), None)

    if chosen:
        plt.rcParams["font.family"] = [chosen, "DejaVu Sans"]
    else:
        # 兜底：使用系统第一个含 CJK/Chinese 的字体
        cjk_fonts = [f.name for f in fm.fontManager.ttflist
                     if any(kw in f.name for kw in ["CJK", "Chinese", "Hei", "Song"])]
        if cjk_fonts:
            plt.rcParams["font.family"] = [cjk_fonts[0], "DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False   # 负号正常显示
    return chosen or "system-default"


FONT_NAME = setup_chinese_font()

# ---------- 学术论文色彩方案（Nature/Science 风格）------------------------
NATURE_COLORS = {
    "blue":   "#3182BD",   # 主色：深蓝
    "red":    "#E6550D",   # 强调：砖红
    "green":  "#31A354",   # 正向：深绿
    "purple": "#756BB1",   # 次要：紫
    "gray":   "#636363",   # 辅助：灰
    "orange": "#FD8D3C",   # 警示：橙
    "cyan":   "#6BAED6",   # 浅蓝
    "pink":   "#FD9272",   # 浅橙红
}
PALETTE = list(NATURE_COLORS.values())

# ---------- 随机种子 -------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED):
    """固定所有随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

# ---------- 设备配置 -------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[系统] 使用设备：{DEVICE} | 中文字体：{FONT_NAME}")

# ---------- 路径配置 -------------------------------------------------------
DATA_DIR = "/mnt/project"
OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. 数据加载与预处理模块
# ═══════════════════════════════════════════════════════════════════════════

class CreditDataPreprocessor:
    """
    信贷数据预处理器（双数据集融合版本）

    负责：
      · 加载两份数据集：贷款审批数据（loan_approval）
                        Give Me Some Credit 竞赛数据（cstraining）
      · 数据清洗：缺失值填充、异常值截断、类别编码
      · 特征工程：衍生风险指标
      · 标准化与数据集划分
    """

    def __init__(self, loan_path: str, cs_path: str):
        self.loan_path = loan_path
        self.cs_path = cs_path
        self.scaler = StandardScaler()
        self.feature_names: list = []

    # ── 1.1 加载原始数据 ──────────────────────────────────────────────────
    def load_raw(self):
        """加载两个原始 CSV 文件。"""
        df_loan = pd.read_csv(self.loan_path)
        df_cs   = pd.read_csv(self.cs_path, index_col=0)
        print(f"[数据] loan_approval 数据维度：{df_loan.shape}")
        print(f"[数据] cstraining    数据维度：{df_cs.shape}")
        return df_loan, df_cs

    # ── 1.2 处理 loan_approval 数据集 ────────────────────────────────────
    @staticmethod
    def _process_loan(df: pd.DataFrame) -> pd.DataFrame:
        """
        处理贷款审批数据集。
        标签：Approved=0（未违约/审批通过），Rejected=1（拒绝/高风险）
        """
        df = df.copy()

        # 目标变量编码：Rejected → 1（高风险），Approved → 0（低风险）
        df["label"] = (df["Loan_Status"] == "Rejected").astype(int)

        # 性别独热编码
        df["gender_male"] = (df["Gender"] == "Male").astype(int)

        # 衍生特征：贷款收入比（Loan-to-Income Ratio）
        df["loan_income_ratio"] = df["LoanAmount"] / (df["Income"] + 1e-6)

        # 衍生特征：标准化信用评分（0-1 区间）
        df["credit_norm"] = (df["CreditScore"] - df["CreditScore"].min()) / \
                            (df["CreditScore"].max() - df["CreditScore"].min() + 1e-6)

        # 衍生特征：年龄分段风险指数
        df["age_risk"] = pd.cut(
            df["Age"],
            bins=[0, 25, 35, 50, 65, 100],
            labels=[0.8, 0.5, 0.3, 0.4, 0.6]  # 年轻/老年风险略高
        ).astype(float)

        feature_cols = [
            "Age", "Income", "LoanAmount", "CreditScore",
            "gender_male", "loan_income_ratio", "credit_norm", "age_risk"
        ]
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        return df[feature_cols + ["label"]]

    # ── 1.3 处理 cstraining 数据集 ───────────────────────────────────────
    @staticmethod
    def _process_cs(df: pd.DataFrame) -> pd.DataFrame:
        """
        处理 Give Me Some Credit（GiveMeSomeCredit）数据集。
        目标：SeriousDlqin2yrs（2年内严重逾期）→ 保持原始二分类。
        """
        df = df.copy()

        # 缺失值：用中位数填充
        df["MonthlyIncome"].fillna(df["MonthlyIncome"].median(), inplace=True)
        df["NumberOfDependents"].fillna(df["NumberOfDependents"].median(), inplace=True)

        # 异常值截断（激进截断至 99 分位数，防止极端值导致 NaN）
        clip_cols = [
            "RevolvingUtilizationOfUnsecuredLines", "DebtRatio",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "MonthlyIncome",
        ]
        for col in clip_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower, upper=upper)

        # 最终安全检查：替换残留的 inf/nan
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)

        # 衍生特征：总逾期次数
        df["total_delinquency"] = (
            df["NumberOfTime30-59DaysPastDueNotWorse"] +
            df["NumberOfTimes90DaysLate"] +
            df["NumberOfTime60-89DaysPastDueNotWorse"]
        )

        # 衍生特征：月收入负债比
        df["income_debt_ratio"] = df["MonthlyIncome"] / (df["DebtRatio"] * df["MonthlyIncome"] + 1)

        feature_cols = [
            "RevolvingUtilizationOfUnsecuredLines", "age",
            "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
            "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents",
            "total_delinquency", "income_debt_ratio"
        ]

        # 目标变量
        df["label"] = df["SeriousDlqin2yrs"].astype(int)
        return df[feature_cols + ["label"]]

    # ── 1.4 主处理流水线 ──────────────────────────────────────────────────
    def fit_transform(self, use_cs_sample: int = 20000):
        """
        完整数据处理流水线：加载→清洗→合并→标准化→划分。

        Parameters
        ----------
        use_cs_sample : int
            从 cstraining 中抽取的样本数（加速实验，默认 20000 条）

        Returns
        -------
        X_train, X_val, X_test : np.ndarray
        y_train, y_val, y_test : np.ndarray
        df_loan_processed      : pd.DataFrame  （用于可视化）
        """
        df_loan_raw, df_cs_raw = self.load_raw()

        # 处理各数据集
        df_loan = self._process_loan(df_loan_raw)
        df_cs   = self._process_cs(df_cs_raw)

        # ── 使用 cstraining 作为主训练集（数据量大，特征丰富）──────────
        # 分层抽样以保持类别分布
        df_cs_sample, _ = train_test_split(
            df_cs, train_size=use_cs_sample,
            stratify=df_cs["label"], random_state=SEED
        )

        print(f"\n[数据] 使用 cstraining 样本：{len(df_cs_sample)} 条")
        print(f"[数据] 违约率：{df_cs_sample['label'].mean():.4f}")

        X = df_cs_sample.drop("label", axis=1).values.astype(np.float32)
        y = df_cs_sample["label"].values.astype(np.int64)

        self.feature_names = df_cs_sample.drop("label", axis=1).columns.tolist()

        # 三段划分：60% 训练 / 20% 验证 / 20% 测试
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=SEED
        )
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED
        )

        # 标准化（仅用训练集 fit）
        X_tr  = self.scaler.fit_transform(X_tr)
        X_val = self.scaler.transform(X_val)
        X_te  = self.scaler.transform(X_te)

        print(f"[数据] 训练集：{X_tr.shape} | 验证集：{X_val.shape} | 测试集：{X_te.shape}")

        return X_tr, X_val, X_te, y_tr, y_val, y_te, df_loan, df_cs_sample


# ═══════════════════════════════════════════════════════════════════════════
# 2. 模型架构模块 —— 基于注意力机制的信用评分网络（创新核心）
# ═══════════════════════════════════════════════════════════════════════════

class FeatureSelfAttention(nn.Module):
    """
    特征级自注意力模块（Feature-Level Self-Attention）

    【创新说明】
    传统 MLP 对所有特征一视同仁地通过全连接层组合，
    无法动态感知哪些特征对当前样本的风险判断更关键。

    本模块受论文式 (2-2) 启发，将每个输入特征视为一个"token"，
    通过多头注意力机制计算特征间的相互关联权重：
      · Query/Key/Value 均由输入特征线性投影生成
      · Softmax 归一化后得到动态特征重要性权重矩阵
      · 加权聚合后的表示比原始特征更具判别力

    这模拟了贷前评估智能体中：
    "将文本中'近期扩大生产'与财务报表中'现金流紧张'自动关联"
    的跨特征注意力机制。
    """

    def __init__(self, num_features: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        # 确保 embed_dim 可被 num_heads 整除
        self.embed_dim = max(num_heads * 4, 32)

        # 将每个标量特征嵌入为 embed_dim 维向量
        self.feature_embed = nn.Linear(1, self.embed_dim)

        # 多头自注意力（PyTorch 内置，高效稳定）
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True   # 输入形状：(batch, seq, embed)
        )

        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_features * self.embed_dim

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (batch_size, num_features)

        Returns
        -------
        out      : (batch_size, num_features * embed_dim)  特征增强表示
        attn_w   : (batch_size, num_features, num_features) 注意力权重矩阵
        """
        batch_size = x.size(0)

        # 将每个特征扩展为 (batch, num_features, 1)，再嵌入为向量
        x_expanded = x.unsqueeze(-1)                      # (B, F, 1)
        x_embedded = self.feature_embed(x_expanded)       # (B, F, embed_dim)

        # 自注意力：Q=K=V=特征嵌入
        attn_out, attn_w = self.attn(x_embedded, x_embedded, x_embedded)

        # 残差连接 + LayerNorm（稳定训练）
        attn_out = self.layer_norm(attn_out + x_embedded)
        attn_out = self.dropout(attn_out)

        # 展平输出
        out = attn_out.reshape(batch_size, -1)             # (B, F * embed_dim)
        return out, attn_w


class AttentionCreditScoringNet(nn.Module):
    """
    注意力增强信用评分网络（主模型）

    架构设计：
      ┌─────────────────────────────────────────────────┐
      │  输入层：原始数值特征 (num_features,)            │
      ├─────────────────────────────────────────────────┤
      │  特征自注意力层（FeatureSelfAttention）          │
      │  → 动态权重 × 特征嵌入 → 增强特征表示           │
      ├─────────────────────────────────────────────────┤
      │  深度分类器（DNN Head）                          │
      │  Linear → BN → GELU → Dropout（×3 层）          │
      ├─────────────────────────────────────────────────┤
      │  输出层：违约概率 ∈ [0, 1]                      │
      └─────────────────────────────────────────────────┘

    对应论文式 (2-1)(2-2)(2-3)：
      h_i = Transformer-Fusion(x_i; Θ)
      Attention(Q,K,V) = softmax(QK^T / √d_k) V
      s_i = f_credit(h_i; φ) = σ(W₂·ReLU(W₁h_i + b₁) + b₂)
    """

    def __init__(self, num_features: int, hidden_dims: list = None,
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # ── 特征注意力层 ──────────────────────────────────────────────
        self.attention = FeatureSelfAttention(num_features, num_heads, dropout=0.1)
        attn_out_dim = self.attention.output_dim

        # ── 残差门控（将原始特征融合回注意力输出）───────────────────
        self.gate = nn.Sequential(
            nn.Linear(num_features, attn_out_dim),
            nn.Sigmoid()
        )

        # ── 深度分类器 ────────────────────────────────────────────────
        layers = []
        in_dim = attn_out_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),                   # GELU 比 ReLU 更平滑，适合信用评分
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        self.classifier = nn.Sequential(*layers)

        # ── 输出层 ────────────────────────────────────────────────────
        self.output_layer = nn.Linear(in_dim, 1)

        # ── 权重初始化 ────────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化，避免梯度消失/爆炸。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (batch_size, num_features)

        Returns
        -------
        prob   : (batch_size,) 违约概率
        attn_w : (batch_size, num_features, num_features) 注意力权重
        """
        # 注意力增强特征
        attn_feat, attn_w = self.attention(x)

        # 门控残差融合
        gate_val = self.gate(x)
        fused = attn_feat * gate_val

        # 深度分类
        hidden = self.classifier(fused)
        logit  = self.output_layer(hidden).squeeze(-1)
        prob   = torch.sigmoid(logit)

        return prob, attn_w


# ═══════════════════════════════════════════════════════════════════════════
# 3. 损失函数模块 —— 类别平衡 Focal Loss（创新点）
# ═══════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss（焦点损失）

    【创新说明】
    信贷数据存在严重类别不平衡（违约率约 6.7%），
    普通交叉熵损失会被大量正常样本主导，导致模型偏向预测"不违约"。

    Focal Loss 在论文式 (2-4) 交叉熵损失的基础上，
    引入动态调制因子 (1-p_t)^γ，
    使模型自动聚焦于难以分类的"高风险边界样本"：
      FL(p_t) = -α_t · (1-p_t)^γ · log(p_t)

    当 γ=0 时退化为标准交叉熵。
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight  # 正类权重，进一步平衡

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        preds   : (batch,) 预测概率（已经过 Sigmoid）
        targets : (batch,) 真实标签 {0, 1}
        """
        targets = targets.float()
        bce = -( targets * torch.log(preds + 1e-8) +
                 (1 - targets) * torch.log(1 - preds + 1e-8) )

        # 正类权重加权
        if self.pos_weight is not None:
            weight = targets * self.pos_weight + (1 - targets)
            bce = bce * weight

        # Focal 调制
        p_t = preds * targets + (1 - preds) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_factor * bce
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════
# 4. 训练器模块
# ═══════════════════════════════════════════════════════════════════════════

class CreditModelTrainer:
    """
    模型训练器，封装完整训练-验证-测试流程。

    功能：
      · 学习率热身 + 余弦退火调度（Cosine Annealing with Warmup）
      · 早停机制（防止过拟合）
      · 逐轮记录训练/验证损失、AUC、F1 等指标
    """

    def __init__(self, model: nn.Module, pos_weight: float = 5.0,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 patience: int = 10):
        self.model       = model.to(DEVICE)
        self.criterion   = FocalLoss(
            alpha=0.75, gamma=2.0,
            pos_weight=torch.tensor(pos_weight, device=DEVICE)
        )
        self.optimizer   = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.patience    = patience
        self.history     = {
            "train_loss": [], "val_loss": [],
            "train_auc":  [], "val_auc":  [],
            "train_f1":   [], "val_f1":   [],
        }
        self._best_val_auc = 0.0
        self._no_improve   = 0
        self._best_state   = None

    def _make_loader(self, X, y, batch_size: int, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.int64)
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    def _run_epoch(self, loader: DataLoader, training: bool):
        """执行一轮训练或评估，返回平均损失、概率列表、标签列表。"""
        self.model.train(training)
        total_loss, all_probs, all_labels = 0.0, [], []

        with torch.set_grad_enabled(training):
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                prob, _ = self.model(X_b)
                loss = self.criterion(prob, y_b)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item() * len(y_b)
                all_probs.extend(prob.detach().cpu().numpy())
                all_labels.extend(y_b.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        probs    = np.array(all_probs)
        labels   = np.array(all_labels)
        preds    = (probs >= 0.5).astype(int)

        auc = roc_auc_score(labels, probs)
        f1  = f1_score(labels, preds, zero_division=0)
        return avg_loss, auc, f1, probs, labels

    def fit(self, X_tr, y_tr, X_val, y_val,
            epochs: int = 60, batch_size: int = 256):
        """完整训练流程（含早停）。"""
        train_loader = self._make_loader(X_tr, y_tr, batch_size, shuffle=True)
        val_loader   = self._make_loader(X_val, y_val, batch_size, shuffle=False)

        # 余弦退火调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )

        print("\n[训练] 开始训练...")
        print(f"{'Epoch':>6} | {'训练损失':>8} | {'验证损失':>8} | "
              f"{'训练AUC':>8} | {'验证AUC':>8} | {'验证F1':>7} | {'学习率':>10}")
        print("-" * 75)

        for epoch in range(1, epochs + 1):
            tr_loss, tr_auc, tr_f1, _, _ = self._run_epoch(train_loader, training=True)
            vl_loss, vl_auc, vl_f1, _, _ = self._run_epoch(val_loader,   training=False)
            scheduler.step()

            # 记录历史
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(vl_loss)
            self.history["train_auc"].append(tr_auc)
            self.history["val_auc"].append(vl_auc)
            self.history["train_f1"].append(tr_f1)
            self.history["val_f1"].append(vl_f1)

            lr_now = self.optimizer.param_groups[0]["lr"]
            if epoch % 5 == 0 or epoch == 1:
                print(f"{epoch:>6} | {tr_loss:>8.4f} | {vl_loss:>8.4f} | "
                      f"{tr_auc:>8.4f} | {vl_auc:>8.4f} | {vl_f1:>7.4f} | {lr_now:>10.2e}")

            # 早停与模型保存
            if vl_auc > self._best_val_auc:
                self._best_val_auc = vl_auc
                self._best_state   = {k: v.clone() for k, v in self.model.state_dict().items()}
                self._no_improve   = 0
            else:
                self._no_improve += 1
                if self._no_improve >= self.patience:
                    print(f"\n[早停] 验证 AUC 连续 {self.patience} 轮未改善，"
                          f"在 Epoch {epoch - self.patience} 处停止。")
                    break

        # 恢复最佳权重
        self.model.load_state_dict(self._best_state)
        print(f"\n[训练完成] 最佳验证 AUC：{self._best_val_auc:.4f}")

    def evaluate(self, X_te, y_te, batch_size: int = 512):
        """在测试集上全面评估模型性能。"""
        test_loader = self._make_loader(X_te, y_te, batch_size, shuffle=False)
        _, auc, f1, probs, labels = self._run_epoch(test_loader, training=False)

        preds = (probs >= 0.5).astype(int)
        acc   = accuracy_score(labels, preds)
        ap    = average_precision_score(labels, probs)

        print("\n" + "═" * 50)
        print("  【测试集最终评估结果】")
        print("═" * 50)
        print(f"  准确率  (Accuracy) : {acc:.4f}")
        print(f"  AUC-ROC            : {auc:.4f}")
        print(f"  F1 分数 (违约类)   : {f1:.4f}")
        print(f"  平均精度 (AP)      : {ap:.4f}")
        print("═" * 50)
        print("\n  分类详细报告：")
        print(classification_report(labels, preds,
                                    target_names=["正常还款", "违约风险"],
                                    digits=4))
        return probs, labels, preds


# ═══════════════════════════════════════════════════════════════════════════
# 5. 可视化模块 —— 6 张国家级学术图表
# ═══════════════════════════════════════════════════════════════════════════

class AcademicVisualizer:
    """
    学术可视化器：生成符合 Nature/Science 风格的高质量图表。
    所有图表标题、坐标轴、图例均使用中文简体。
    """

    # 通用图表参数
    FIG_DPI    = 150
    TITLE_SIZE = 15
    LABEL_SIZE = 13
    TICK_SIZE  = 11
    LEGEND_SIZE= 11

    @classmethod
    def _apply_style(cls, ax, title: str, xlabel: str, ylabel: str,
                     grid_axis: str = "y"):
        """统一应用学术风格到坐标轴。"""
        ax.set_title(title, fontsize=cls.TITLE_SIZE, fontweight="bold",
                     pad=12, color="#1a1a2e")
        ax.set_xlabel(xlabel, fontsize=cls.LABEL_SIZE, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=cls.LABEL_SIZE, labelpad=8)
        ax.tick_params(axis="both", labelsize=cls.TICK_SIZE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        if grid_axis:
            ax.grid(axis=grid_axis, linestyle="--", alpha=0.5, color="#dddddd")

    # ── 图1：训练过程曲线（损失 + AUC 双子图）──────────────────────────
    @classmethod
    def plot_training_curves(cls, history: dict, save_path: str):
        """绘制训练/验证损失曲线与 AUC 曲线。"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("模型训练收敛曲线分析", fontsize=17, fontweight="bold",
                     y=1.02, color="#1a1a2e")

        epochs = range(1, len(history["train_loss"]) + 1)

        # —— 左图：损失曲线 ——
        ax = axes[0]
        ax.plot(epochs, history["train_loss"], color=NATURE_COLORS["blue"],
                linewidth=2.2, label="训练集损失", zorder=3)
        ax.plot(epochs, history["val_loss"], color=NATURE_COLORS["red"],
                linewidth=2.2, linestyle="--", label="验证集损失", zorder=3)
        ax.fill_between(epochs, history["train_loss"], history["val_loss"],
                        alpha=0.08, color=NATURE_COLORS["purple"])
        cls._apply_style(ax, "Focal Loss 收敛曲线", "训练轮次（Epoch）", "损失值（Loss）")
        ax.legend(fontsize=cls.LEGEND_SIZE, framealpha=0.8)

        # —— 右图：AUC 曲线 ——
        ax = axes[1]
        ax.plot(epochs, history["train_auc"], color=NATURE_COLORS["green"],
                linewidth=2.2, label="训练集 AUC", zorder=3)
        ax.plot(epochs, history["val_auc"], color=NATURE_COLORS["orange"],
                linewidth=2.2, linestyle="--", label="验证集 AUC", zorder=3)
        # 标注最佳验证 AUC
        best_ep  = int(np.argmax(history["val_auc"])) + 1
        best_val = max(history["val_auc"])
        ax.annotate(f"最佳 AUC={best_val:.4f}",
                    xy=(best_ep, best_val),
                    xytext=(best_ep + 2, best_val - 0.02),
                    fontsize=10, color=NATURE_COLORS["red"],
                    arrowprops=dict(arrowstyle="->", color=NATURE_COLORS["red"],
                                   lw=1.5))
        ax.axhline(y=0.8, color="gray", linestyle=":", alpha=0.6, linewidth=1.5,
                   label="AUC=0.80 基准线")
        cls._apply_style(ax, "ROC-AUC 训练进程", "训练轮次（Epoch）", "AUC 值")
        ax.legend(fontsize=cls.LEGEND_SIZE, framealpha=0.8)
        ax.set_ylim([0.5, 1.0])

        plt.tight_layout()
        plt.savefig(save_path, dpi=cls.FIG_DPI, bbox_inches="tight",
                    facecolor="white")
        plt.close()
        print(f"[可视化] 已保存：{save_path}")

    # ── 图2：ROC 曲线 + PR 曲线 ─────────────────────────────────────────
    @classmethod
    def plot_roc_pr_curves(cls, y_true, y_prob, save_path: str):
        """绘制 ROC 曲线与精确率-召回率曲线。"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("模型区分能力评估（ROC 与 PR 曲线）", fontsize=17,
                     fontweight="bold", y=1.02, color="#1a1a2e")

        # —— 左图：ROC 曲线 ——
        ax = axes[0]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val      = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=NATURE_COLORS["blue"], linewidth=2.5,
                label=f"注意力信用评分模型  AUC = {auc_val:.4f}", zorder=3)
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
                linewidth=1.5, label="随机猜测基准线")
        ax.fill_between(fpr, tpr, alpha=0.10, color=NATURE_COLORS["blue"])
        cls._apply_style(ax, "ROC 曲线", "假阳性率（FPR）", "真阳性率（TPR）",
                         grid_axis="both")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.legend(fontsize=cls.LEGEND_SIZE, loc="lower right", framealpha=0.85)

        # —— 右图：PR 曲线 ——
        ax = axes[1]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_val = average_precision_score(y_true, y_prob)
        baseline = y_true.mean()
        ax.plot(recall, precision, color=NATURE_COLORS["red"], linewidth=2.5,
                label=f"注意力信用评分模型  AP = {ap_val:.4f}", zorder=3)
        ax.axhline(y=baseline, color="gray", linestyle="--",
                   linewidth=1.5, label=f"随机基准线（违约率={baseline:.3f}）")
        ax.fill_between(recall, precision, alpha=0.08, color=NATURE_COLORS["red"])
        cls._apply_style(ax, "精确率-召回率曲线（PR Curve）",
                         "召回率（Recall）", "精确率（Precision）", grid_axis="both")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.legend(fontsize=cls.LEGEND_SIZE, loc="upper right", framealpha=0.85)

        plt.tight_layout()
        plt.savefig(save_path, dpi=cls.FIG_DPI, bbox_inches="tight",
                    facecolor="white")
        plt.close()
        print(f"[可视化] 已保存：{save_path}")

    # ── 图3：混淆矩阵（带百分比标注）────────────────────────────────────
    @classmethod
    def plot_confusion_matrix(cls, y_true, y_pred, save_path: str,
                              threshold: float = 0.5):
        """绘制归一化混淆矩阵（热力图）。"""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"混淆矩阵分析（决策阈值 = {threshold}）", fontsize=17,
                     fontweight="bold", y=1.02, color="#1a1a2e")

        cm_labels = ["正常还款", "违约风险"]

        for i, (norm, title_suffix) in enumerate(
            [(None, "原始计数"), ("true", "行归一化（召回率视角）")]
        ):
            ax = axes[i]
            cm = confusion_matrix(y_true, y_pred, normalize=norm)
            fmt_str = ".2f" if norm else "d"

            # 选择色系
            cmap = sns.light_palette(NATURE_COLORS["blue"], as_cmap=True)
            sns.heatmap(cm, annot=True, fmt=fmt_str, cmap=cmap,
                        xticklabels=cm_labels, yticklabels=cm_labels,
                        ax=ax, linewidths=0.5, linecolor="white",
                        annot_kws={"size": 14, "weight": "bold"},
                        cbar_kws={"shrink": 0.8})
            ax.set_title(f"混淆矩阵（{title_suffix}）", fontsize=cls.TITLE_SIZE,
                         fontweight="bold", pad=10)
            ax.set_xlabel("预测标签", fontsize=cls.LABEL_SIZE)
            ax.set_ylabel("真实标签", fontsize=cls.LABEL_SIZE)
            ax.tick_params(labelsize=cls.TICK_SIZE)

        plt.tight_layout()
        plt.savefig(save_path, dpi=cls.FIG_DPI, bbox_inches="tight",
                    facecolor="white")
        plt.close()
        print(f"[可视化] 已保存：{save_path}")

    # ── 图4：注意力热力图（特征重要性矩阵）──────────────────────────────
    @classmethod
    def plot_attention_heatmap(cls, model: nn.Module, X_sample: np.ndarray,
                               feature_names: list, save_path: str,
                               n_samples: int = 200):
        """
        可视化特征自注意力权重矩阵。
        取测试集前 n_samples 个样本的平均注意力权重，
        揭示各特征之间的相互关联强度（模型内部可解释性）。
        """
        model.eval()
        sample = torch.tensor(X_sample[:n_samples], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            _, attn_w = model(sample)
            # attn_w: (n_samples, num_features, num_features)
            # 对多头注意力取均值（已在 nn.MultiheadAttention 内部平均）
            mean_attn = attn_w.mean(0).cpu().numpy()   # (F, F)

        fig, ax = plt.subplots(figsize=(11, 9))

        # 截断特征名（防止过长）
        short_names = [n.replace("NumberOf", "次数_")
                        .replace("Revolving", "循环_")
                        .replace("Utilization", "使用率")
                        .replace("UnsecuredLines", "")
                        .replace("Time", "时间")
                        .replace("DaysPastDueNotWorse", "天逾期")
                        .replace("OpenCredit", "开放信用")
                        .replace("LinesAndLoans", "")
                        .replace("RealEstate", "房贷")
                        .replace("LoansOrLines", "")
                        .replace("Dependents", "家属数")
                        .replace("monthly", "月").replace("income", "收入")
                        .replace("_", "\n")[:18]
                       for n in feature_names]

        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(mean_attn, annot=True, fmt=".3f",
                    xticklabels=short_names, yticklabels=short_names,
                    cmap="YlOrRd", ax=ax,
                    linewidths=0.3, linecolor="white",
                    annot_kws={"size": 8},
                    cbar_kws={"label": "注意力权重", "shrink": 0.8})

        ax.set_title("特征自注意力权重矩阵\n（信用风险关键特征交互强度）",
                     fontsize=cls.TITLE_SIZE, fontweight="bold", pad=15)
        ax.set_xlabel("特征（目标键值）", fontsize=cls.LABEL_SIZE)
        ax.set_ylabel("特征（查询来源）", fontsize=cls.LABEL_SIZE)
        plt.xticks(rotation=35, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=cls.FIG_DPI, bbox_inches="tight",
                    facecolor="white")
        plt.close()
        print(f"[可视化] 已保存：{save_path}")

    # ── 图5：风险概率分布与决策阈值分析 ─────────────────────────────────
    @classmethod
    def plot_risk_distribution(cls, y_true, y_prob, save_path: str):
        """绘制模型输出概率分布及最优阈值分析。"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("违约风险概率分布与决策阈值优化", fontsize=17,
                     fontweight="bold", y=1.02, color="#1a1a2e")

        # —— 左图：正/负类概率密度分布 ——
        ax = axes[0]
        prob_pos = y_prob[y_true == 1]  # 违约样本概率
        prob_neg = y_prob[y_true == 0]  # 正常样本概率

        ax.hist(prob_neg, bins=60, alpha=0.65, color=NATURE_COLORS["blue"],
                label=f"正常还款（n={len(prob_neg):,}）", density=True, zorder=2)
        ax.hist(prob_pos, bins=60, alpha=0.65, color=NATURE_COLORS["red"],
                label=f"违约风险（n={len(prob_pos):,}）", density=True, zorder=3)
        ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1.8,
                   label="默认决策阈值 = 0.5")
        cls._apply_style(ax, "违约风险评分概率密度分布",
                         "模型预测违约概率", "密度（Density）", grid_axis="y")
        ax.legend(fontsize=cls.LEGEND_SIZE, framealpha=0.85)

        # —— 右图：不同阈值下 F1/精确率/召回率变化 ——
        ax = axes[1]
        thresholds = np.linspace(0.05, 0.95, 100)
        f1_scores, prec_scores, rec_scores = [], [], []

        for thr in thresholds:
            preds = (y_prob >= thr).astype(int)
            f1_scores.append(f1_score(y_true, preds, zero_division=0))
            prec_scores.append(
                np.sum((preds == 1) & (y_true == 1)) /
                (np.sum(preds == 1) + 1e-8)
            )
            rec_scores.append(
                np.sum((preds == 1) & (y_true == 1)) /
                (np.sum(y_true == 1) + 1e-8)
            )

        best_thr_idx = np.argmax(f1_scores)
        best_thr     = thresholds[best_thr_idx]

        ax.plot(thresholds, f1_scores,  color=NATURE_COLORS["blue"],
                linewidth=2.2, label="F1 分数")
        ax.plot(thresholds, prec_scores, color=NATURE_COLORS["green"],
                linewidth=2.0, linestyle="--", label="精确率")
        ax.plot(thresholds, rec_scores,  color=NATURE_COLORS["red"],
                linewidth=2.0, linestyle=":",  label="召回率")
        ax.axvline(x=best_thr, color=NATURE_COLORS["orange"],
                   linestyle="-.", linewidth=2.0,
                   label=f"最优阈值 = {best_thr:.2f}  (F1={max(f1_scores):.4f})")

        cls._apply_style(ax, "决策阈值 vs. 评价指标",
                         "决策阈值（Threshold）", "指标值", grid_axis="both")
        ax.legend(fontsize=cls.LEGEND_SIZE, framealpha=0.85)
        ax.set_xlim([0.05, 0.95]); ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(save_path, dpi=cls.FIG_DPI, bbox_inches="tight",
                    facecolor="white")
        plt.close()
        print(f"[可视化] 已保存：{save_path}")

    # ── 图6：探索性数据分析（EDA 综合图）───────────────────────────────
    @classmethod
    def plot_eda(cls, df_loan: pd.DataFrame, df_cs: pd.DataFrame,
                 save_path: str):
        """绘制贷款审批数据集的探索性数据分析图。"""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("信贷数据集探索性分析（EDA）", fontsize=18,
                     fontweight="bold", y=1.01, color="#1a1a2e")

        gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)

        # —— (1,1) 贷款审批结果分布 ——
        ax1 = fig.add_subplot(gs[0, 0])
        counts = df_loan["label"].value_counts()
        labels_pie = ["审批通过\n(Approved)", "审批拒绝\n(Rejected)"]
        colors_pie = [NATURE_COLORS["green"], NATURE_COLORS["red"]]
        wedges, texts, autotexts = ax1.pie(
            counts.values[::-1], labels=labels_pie,
            colors=colors_pie, autopct="%1.1f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(edgecolor="white", linewidth=2.5)
        )
        for at in autotexts:
            at.set_fontsize(12)
            at.set_fontweight("bold")
        ax1.set_title("贷款审批结果分布", fontsize=cls.TITLE_SIZE, fontweight="bold")

        # —— (1,2) 信用评分按审批结果分布 ——
        ax2 = fig.add_subplot(gs[0, 1])
        approved = df_loan[df_loan["label"] == 0]["CreditScore"].dropna()
        rejected = df_loan[df_loan["label"] == 1]["CreditScore"].dropna()
        ax2.hist(approved, bins=30, alpha=0.7, color=NATURE_COLORS["green"],
                 label="审批通过", density=True, zorder=3)
        ax2.hist(rejected, bins=30, alpha=0.7, color=NATURE_COLORS["red"],
                 label="审批拒绝", density=True, zorder=2)
        cls._apply_style(ax2, "信用评分分布对比",
                         "信用评分（Credit Score）", "概率密度")
        ax2.legend(fontsize=10, framealpha=0.85)

        # —— (1,3) 收入 vs. 贷款金额散点图 ——
        ax3 = fig.add_subplot(gs[0, 2])
        color_map = {0: NATURE_COLORS["green"], 1: NATURE_COLORS["red"]}
        for label, grp in df_loan.groupby("label"):
            ax3.scatter(grp["Income"] / 1000, grp["LoanAmount"] / 1000,
                        alpha=0.4, s=20, color=color_map[label],
                        label="审批通过" if label == 0 else "审批拒绝",
                        zorder=3 if label == 1 else 2)
        cls._apply_style(ax3, "申请人收入 vs. 贷款金额",
                         "年收入（千元）", "贷款金额（千元）", grid_axis="both")
        ax3.legend(fontsize=10, framealpha=0.85)

        # —— (2,1) 违约率 vs. 年龄分布（GiveMeSomeCredit）——
        ax4 = fig.add_subplot(gs[1, 0])
        age_bins = [18, 25, 35, 45, 55, 65, 100]
        age_labels_cn = ["18-25", "25-35", "35-45", "45-55", "55-65", "65+"]
        df_cs["age_group"] = pd.cut(df_cs["age"], bins=age_bins,
                                    labels=age_labels_cn, right=False)
        default_rate = df_cs.groupby("age_group")["label"].mean()
        bars = ax4.bar(default_rate.index.astype(str), default_rate.values * 100,
                       color=PALETTE[:len(default_rate)], edgecolor="white",
                       linewidth=1.5, zorder=3)
        for bar, val in zip(bars, default_rate.values):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.1,
                     f"{val*100:.1f}%", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
        cls._apply_style(ax4, "不同年龄段违约率分析（GiveMeSomeCredit）",
                         "年龄区间", "违约率（%）")
        ax4.set_ylim([0, max(default_rate.values) * 130])

        # —— (2,2) 逾期次数分布（GiveMeSomeCredit）——
        ax5 = fig.add_subplot(gs[1, 1])
        dl_col = "NumberOfTime30-59DaysPastDueNotWorse"
        delinq_counts = df_cs[dl_col].clip(upper=10).value_counts().sort_index()
        ax5.bar(delinq_counts.index.astype(str), delinq_counts.values,
                color=NATURE_COLORS["cyan"], edgecolor="white",
                linewidth=1.2, zorder=3)
        cls._apply_style(ax5, "30-59 天逾期次数分布",
                         "逾期次数", "样本数量（条）")

        # —— (2,3) 各特征与违约率相关性（GiveMeSomeCredit）——
        ax6 = fig.add_subplot(gs[1, 2])
        num_cols = [
            "RevolvingUtilizationOfUnsecuredLines", "DebtRatio",
            "total_delinquency", "income_debt_ratio"
        ]
        available_cols = [c for c in num_cols if c in df_cs.columns]
        label_map = {
            "RevolvingUtilizationOfUnsecuredLines": "循环信用\n使用率",
            "DebtRatio": "负债比率",
            "total_delinquency": "总逾期次数",
            "income_debt_ratio": "收入负债比"
        }
        corr_vals = [df_cs[c].corr(df_cs["label"]) for c in available_cols]
        bar_colors = [NATURE_COLORS["red"] if v > 0 else NATURE_COLORS["green"]
                      for v in corr_vals]
        ax6.barh([label_map.get(c, c) for c in available_cols],
                 corr_vals, color=bar_colors, edgecolor="white",
                 linewidth=1.2, zorder=3)
        ax6.axvline(x=0, color="black", linewidth=1.2)
        cls._apply_style(ax6, "关键特征与违约标签皮尔逊相关系数",
                         "相关系数（Pearson r）", "", grid_axis="x")
        for i, val in enumerate(corr_vals):
            ax6.text(val + 0.002 * np.sign(val), i,
                     f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

        plt.savefig(save_path, dpi=cls.FIG_DPI, bbox_inches="tight",
                    facecolor="white")
        plt.close()
        print(f"[可视化] 已保存：{save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. 主程序入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    信贷智能评分系统主流程。
    严格按照"数据→模型→训练→评估→可视化"管线执行。
    """
    print("\n" + "═" * 60)
    print("  智链信航 · 贷前评估智能体 · 信用评分模型")
    print("  多智能体协同信贷全周期智能决策引擎")
    print("═" * 60 + "\n")

    # ── Step 1：数据预处理 ────────────────────────────────────────────────
    preprocessor = CreditDataPreprocessor(
        loan_path="loan__approval_data.csv",
        cs_path  ="cstraining.csv",
    )
    X_tr, X_val, X_te, y_tr, y_val, y_te, df_loan, df_cs = \
        preprocessor.fit_transform(use_cs_sample=20000)

    num_features = X_tr.shape[1]
    feature_names = preprocessor.feature_names

    # 类别权重（处理不平衡）
    class_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=y_tr
    )
    pos_weight = class_weights[1] / class_weights[0]
    print(f"\n[数据] 正类权重 pos_weight = {pos_weight:.2f}")

    # ── Step 2：构建模型 ──────────────────────────────────────────────────
    model = AttentionCreditScoringNet(
        num_features=num_features,
        hidden_dims=[256, 128, 64],
        num_heads=4,
        dropout=0.3
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[模型] 架构：注意力增强信用评分网络")
    print(f"[模型] 可训练参数总量：{total_params:,}")
    print(f"[模型] 特征维度：{num_features} → 注意力嵌入维度：{model.attention.embed_dim}")

    # ── Step 3：训练 ──────────────────────────────────────────────────────
    trainer = CreditModelTrainer(
        model=model,
        pos_weight=pos_weight,
        lr=5e-4,
        weight_decay=1e-4,
        patience=12
    )
    trainer.fit(X_tr, y_tr, X_val, y_val, epochs=60, batch_size=256)

    # ── Step 4：测试集评估 ────────────────────────────────────────────────
    y_prob, y_true, y_pred = trainer.evaluate(X_te, y_te)

    # ── Step 5：生成全部可视化图表 ────────────────────────────────────────
    print("\n[可视化] 开始生成国家级学术图表（共 6 张）...\n")
    vis = AcademicVisualizer()

    vis.plot_training_curves(
        trainer.history,
        save_path=f"{OUTPUT_DIR}/图1_训练收敛曲线.png"
    )
    vis.plot_roc_pr_curves(
        y_true, y_prob,
        save_path=f"{OUTPUT_DIR}/图2_ROC与PR曲线.png"
    )
    vis.plot_confusion_matrix(
        y_true, y_pred,
        save_path=f"{OUTPUT_DIR}/图3_混淆矩阵分析.png"
    )
    vis.plot_attention_heatmap(
        model, X_te, feature_names,
        save_path=f"{OUTPUT_DIR}/图4_特征注意力热力图.png"
    )
    vis.plot_risk_distribution(
        y_true, y_prob,
        save_path=f"{OUTPUT_DIR}/图5_风险概率分布与阈值优化.png"
    )
    vis.plot_eda(
        df_loan, df_cs,
        save_path=f"{OUTPUT_DIR}/图6_探索性数据分析EDA.png"
    )

    print("\n" + "═" * 60)
    print("  所有任务执行完毕！图表已保存至输出目录。")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
