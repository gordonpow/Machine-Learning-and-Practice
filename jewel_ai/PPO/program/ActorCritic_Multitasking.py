import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

# ==========================================
# 🔥 新增：自注意力模組 (Self-Attention)
# ==========================================
class SelfAttention(nn.Module):
    """
    Self-Attention Layer for 2D inputs (Feature Maps).
    讓模型學會「關注」盤面上重要的區域，而不僅僅是看局部。
    """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        # 1. Query, Key, Value 的卷積層 (1x1 Conv)
        # 為了節省運算，通常會把 Query 和 Key 的通道數縮小 (例如除以 8)
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        # 2. Gamma 參數：控制 Attention 的強度
        # 初始設為 0，代表一開始模型只用原本的 CNN 特徵，慢慢學會加入 Attention
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x shape: (Batch, Channel, Height, Width)
        """
        m_batchsize, C, width, height = x.size()
        
        # --- Step 1: 計算 Query 和 Key ---
        # proj_query: (B, C/8, W, H) -> view -> (B, C/8, N) -> permute -> (B, N, C/8)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # proj_key:   (B, C/8, W, H) -> view -> (B, C/8, N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        
        # --- Step 2: 計算 Attention Map (能量分數) ---
        # energy: (B, N, N) -> 代表第 i 個像素與第 j 個像素的關聯度
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy) # 歸一化
        
        # --- Step 3: 將 Attention 加權到 Value 上 ---
        # proj_value: (B, C, W, H) -> view -> (B, C, N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        
        # out: (B, C, N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # 還原形狀: (B, C, W, H)
        out = out.view(m_batchsize, C, width, height)
        
        # --- Step 4: Residual Connection (殘差連接) ---
        # 結果 = 原始特徵 + (gamma * 注意力特徵)
        out = self.gamma * out + x
        return out

# ==========================================
# 修改後的 Actor-Critic 模型
# ==========================================
class Match3ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Match3ActorCritic, self).__init__()
        c, h, w = input_shape
        
        # 1. CNN 特徵提取層
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # 這裡不加 Flatten，因為 Attention 需要保留空間結構 (H, W)
        )
        
        # 🔥 2. 插入 Self-Attention 層
        # 輸入通道數必須對應上一層 CNN 的輸出 (這裡是 64)
        self.attention = SelfAttention(in_dim=64)
        
        flatten_dim = 64 * h * w
        
        # 3. Actor Head
        self.actor = nn.Sequential(
            nn.Flatten(), # Flatten 移到這裡
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        # 4. Critic Head
        self.critic = nn.Sequential(
            nn.Flatten(), # Flatten 移到這裡
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, action_mask=None):
        # 1. 提取特徵
        features = self.cnn(state)
        
        # 🔥 2. 經過 Attention 機制
        # 這會讓 features 包含全域資訊
        features = self.attention(features)
        
        # 3. 進入 Actor / Critic
        logits = self.actor(features)
        
        # Action Masking
        if action_mask is not None:
            logits[action_mask == 0] = -1e9

        dist = Categorical(logits=logits)
        action = dist.sample()
        
        return (
            action.detach(),
            dist.log_prob(action).detach(),
            self.critic(features).detach(),
            dist.entropy().detach()
        )

    def evaluate(self, state, action, action_mask=None):
        features = self.cnn(state)
        
        # 🔥 加入 Attention
        features = self.attention(features)
        
        logits = self.actor(features)
        
        if action_mask is not None:
            logits[action_mask == 0] = -1e9
            
        dist = Categorical(logits=logits)
        
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)
        
        return action_log_probs, state_values.squeeze(), dist_entropy