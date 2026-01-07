import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# ==========================================
# 1. 神經網路模型 (Match3ActorCritic)
# ==========================================
class Match3ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Match3ActorCritic, self).__init__()
        c, h, w = input_shape
        
        # CNN 骨幹
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        flatten_dim = 64 * h * w
        
        # Actor: 動作機率
        self.actor = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        # Critic: 狀態價值
        self.critic = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, action_mask=None):
        """
        Inference 階段：選擇動作
        """
        features = self.cnn(state)
        logits = self.actor(features)
        
        # 應用 Mask
        if action_mask is not None:
            logits[action_mask == 0] = -1e9

        dist = Categorical(logits=logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(features)
        
        # 全部回傳 Tensor (Detached)，格式轉換交給外層 Agent
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action, action_mask=None):
        """
        Training 階段：評估動作 (計算 Loss 用)
        """
        features = self.cnn(state)
        logits = self.actor(features)
        
        if action_mask is not None:
            logits[action_mask == 0] = -1e9

        dist = Categorical(logits=logits)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)
        
        # Squeeze 修正維度
        return action_logprobs, state_values.squeeze(), dist_entropy

# ==========================================
# 2. PPO Agent (負責互動與更新)
# ==========================================
class PPOAgent:
    def __init__(self, input_shape, num_actions, device, lr, gamma, eps_clip, k_epochs):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.policy = Match3ActorCritic(input_shape, num_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        
        self.buffer = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'is_terminals': [], 'values': []}

    def select_action(self, state, action_mask=None):
        with torch.no_grad():
            # 1. 轉為 Tensor 並移至裝置
            # 注意：這裡不要直接 unsqueeze(0)，先檢查維度
            state_tensor = torch.FloatTensor(state).to(self.device)

            # 2. 判斷維度
            # 如果是 (10, 9, 6) -> 3維 -> 代表單一環境 -> 加 Batch 維度變成 (1, 10, 9, 6)
            # 如果是 (24, 10, 9, 6) -> 4維 -> 代表多環境 -> 保持原樣
            if state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)
            
            # 3. 處理 Mask
            mask_tensor = None
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                # 同理，如果是單一 mask (46,) -> 加 batch -> (1, 46)
                if mask_tensor.dim() == 1:
                    mask_tensor = mask_tensor.unsqueeze(0)
            
            # 4. 進模型
            action, logprob, state_val = self.policy.act(state_tensor, mask_tensor)
            
        # 5. 回傳處理 (還原格式)
        # 如果原本輸入是 Numpy 且是 3維 (單環境)，我們就回傳 純量 (Scalar)
        if isinstance(state, np.ndarray) and state.ndim == 3:
            return action.cpu().numpy()[0], logprob.cpu().numpy()[0], state_val.cpu().numpy()[0]
        else:
            # 多環境 (24環境)，回傳 陣列 (Array)
            return action.cpu().numpy(), logprob.cpu().numpy(), state_val.cpu().numpy()

    def update(self):
            # 1. 準備資料
            rewards = torch.tensor(self.buffer['rewards'], dtype=torch.float32).to(self.device)
            is_terminals = torch.tensor(self.buffer['is_terminals'], dtype=torch.bool).to(self.device)
            old_states = torch.cat(self.buffer['states'], dim=0).to(self.device)
            old_actions = torch.cat(self.buffer['actions'], dim=0).to(self.device)
            old_logprobs = torch.cat(self.buffer['log_probs'], dim=0).to(self.device)
            old_state_values = torch.cat(self.buffer['values'], dim=0).to(self.device).squeeze()
            
            # 2. 計算 Returns & Advantages (保持不變)
            returns = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                returns.insert(0, discounted_reward)
            
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)
            advantages = returns - old_state_values.detach()

            # 3. PPO Update Loop
            total_loss = 0
            total_kl = 0  # 🔥 新增：累計 KL

            for _ in range(self.k_epochs):
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
                
                # 計算 Ratio
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # 🔥🔥🔥 新增：計算 Approximate KL Divergence 🔥🔥🔥
                # 公式：(log_ratio - 1) + 1/ratio (較精確) 或簡單版 mean((old - new)^2)
                # 這裡用簡單且常用的 log_ratio 近似法
                with torch.no_grad():
                    # KL = old_log_p - new_log_p
                    # 這裡計算 batch 的平均 KL
                    approx_kl = (old_logprobs.detach() - logprobs).mean().item()
                total_kl += approx_kl

                # PPO Loss (保持不變)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, returns) - 0.01 * dist_entropy
                
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                
                total_loss += loss.mean().item()
                
            avg_loss = total_loss / self.k_epochs
            avg_kl = total_kl / self.k_epochs # 🔥 計算平均 KL

            # 🔥 回傳 Loss 和 KL
            return avg_loss, avg_kl

    def clear_buffer(self):
        for key in self.buffer:
            self.buffer[key] = []