import os
import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from collections import deque
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 引入您的模組
from jewel_env.GAME_jewel_env_blacklist import JewelEnv
from BFS_Solver import JewelSolver
from PPO_Agent_Multitasking import PPOAgent

# ================= 參數設定 =================
# 🔥 多核心設定
NUM_ENVS = 18             # 同時跑幾個環境 (建議設為 CPU 核心數的一半或 3/4)
MAX_TRAINING_STEPS = 10_000_000
UPDATE_TIMESTEP = 2000   # 這是總步數
SAVE_INTERVAL = 50000

# 引導設定
SOLVER_DEPTH = 2
EXPERT_BONUS = 0.5
USE_TEACHER_FORCING = False
TEACHER_FORCING_RATE = 0.1

# PPO 參數
LR = 0.0001
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4

# 自適應 LR
LR_PATIENCE = 500
LR_FACTOR = 0.5
MIN_LR = 1e-6

CHECKPOINT_DIR = "training_checkpoints"
LOG_DIR = "training_logs"
PRETRAINED_MODEL = None

# ==========================================
# 🛠️ 多進程 Worker (獨立運作的環境)
# ==========================================
def worker_process(remote, parent_remote, env_idx):
    parent_remote.close()
    
    env = JewelEnv(reward_mode="advanced")
    solver = JewelSolver(env)
    
    obs, _ = env.reset()
    
    current_solver_act = None
    try:
        sol = solver.solve(max_depth=SOLVER_DEPTH)
        if sol: current_solver_act = sol[0]
    except: pass

    try:
        while True:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                action = data
                
                try:
                    next_obs, reward, done, _, info = env.step(action)
                except IndexError:
                    safe_action = env.board.shape[0] * (env.board.shape[1] - 1)
                    try:
                        next_obs, reward, done, _, info = env.step(safe_action)
                    except:
                        next_obs, _ = env.reset()
                        reward, done, info = 0, True, {}

                if done:
                    real_next_obs = next_obs
                    next_obs, _ = env.reset()
                    info['terminal_observation'] = real_next_obs

                next_solver_act = None
                try:
                    sol = solver.solve(max_depth=SOLVER_DEPTH)
                    if sol: next_solver_act = sol[0]
                except: pass

                remote.send({
                    'next_obs': next_obs,
                    'reward': reward,
                    'done': done,
                    'info': info,
                    'solver_act': current_solver_act
                })
                current_solver_act = next_solver_act

            elif cmd == 'reset':
                obs, _ = env.reset()
                current_solver_act = None
                try:
                    sol = solver.solve(max_depth=SOLVER_DEPTH)
                    if sol: current_solver_act = sol[0]
                except: pass
                
                remote.send({
                    'obs': obs,
                    'solver_act': current_solver_act
                })

            elif cmd == 'close':
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        remote.close()

# ==========================================
# 🎮 Vector Environment 管理器
# ==========================================
class SubprocVecEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = []
        
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            p = mp.Process(target=worker_process, args=(work_remote, remote, i))
            p.daemon = True
            p.start()
            self.ps.append(p)
            work_remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs = np.stack([r['obs'] for r in results])
        solver_acts = [r['solver_act'] for r in results]
        return obs, solver_acts

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        
        obs = np.stack([r['next_obs'] for r in results])
        rewards = np.stack([r['reward'] for r in results])
        dones = np.stack([r['done'] for r in results])
        infos = [r['info'] for r in results]
        solver_acts = [r['solver_act'] for r in results]
        
        return obs, rewards, dones, infos, solver_acts

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

# ==========================================
# 🏋️ 主訓練邏輯
# ==========================================
def train():
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(LOG_DIR, f"run_multicore_{timestamp}"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用裝置: {device} | 並行環境數: {NUM_ENVS}")

    dummy_env = JewelEnv(reward_mode="advanced")
    rows, cols = dummy_env.board.shape
    try: num_actions = dummy_env.action_space.n
    except: num_actions = rows * cols
    input_shape = (10, rows, cols)
    del dummy_env

    print("⚙️ 正在啟動多核心環境 (這可能需要幾秒)...")
    envs = SubprocVecEnv(NUM_ENVS)
    print("✅ 環境啟動完成！")

    agent = PPOAgent(input_shape, num_actions, device, LR, GAMMA, EPS_CLIP, K_EPOCHS)
    
    if PRETRAINED_MODEL and os.path.exists(PRETRAINED_MODEL):
        print(f"📥 載入權重: {PRETRAINED_MODEL}")
        agent.policy.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=device))
    else:
        print("🆕 從零開始訓練")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        agent.optimizer, mode='max', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR
    )

    time_step = 0
    i_episode = 0
    
    obs, solver_acts = envs.reset()
    
    score_history = deque(maxlen=100)
    current_ep_scores = np.zeros(NUM_ENVS)
    current_ep_matches = np.zeros(NUM_ENVS)
    current_ep_steps = np.zeros(NUM_ENVS)
    
    pbar = tqdm(total=MAX_TRAINING_STEPS, desc="🚀 Multi-Core Training", dynamic_ncols=True)
    last_loss = 0.0

    try:
        while time_step < MAX_TRAINING_STEPS:
            actions, log_probs, state_vals = agent.select_action(obs, action_mask=None)
            
            final_actions = actions.copy()
            for i in range(NUM_ENVS):
                s_act = solver_acts[i]
                if USE_TEACHER_FORCING and s_act is not None:
                    if np.random.rand() < TEACHER_FORCING_RATE:
                        final_actions[i] = s_act

            next_obs, rewards, dones, infos, next_solver_acts = envs.step(final_actions)

            for i in range(NUM_ENVS):
                bonus = 0.0
                if solver_acts[i] is not None and final_actions[i] == solver_acts[i]:
                    bonus = EXPERT_BONUS
                    current_ep_matches[i] += 1
                
                total_reward = rewards[i] + bonus
                
                # 🔥🔥🔥【關鍵修正點】🔥🔥🔥
                # 這裡必須將資料變成 1 維的 Tensor (使用 view(-1) 或 unsqueeze(0))
                # 才能讓 PPOAgent 裡的 torch.cat 正常運作
                
                # State: (C, H, W) -> (1, C, H, W)
                agent.buffer['states'].append(torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(device))
                
                # Action: () -> (1)
                agent.buffer['actions'].append(torch.tensor(final_actions[i], dtype=torch.long).view(-1).to(device))
                
                # Log Prob: () -> (1)
                agent.buffer['log_probs'].append(torch.tensor(log_probs[i], dtype=torch.float32).view(-1).to(device))
                
                # Value: () -> (1)
                agent.buffer['values'].append(torch.tensor(state_vals[i], dtype=torch.float32).view(-1).to(device))
                
                # Rewards / Terminals 保持 float/bool 即可，Agent 內部會轉
                agent.buffer['rewards'].append(total_reward)
                agent.buffer['is_terminals'].append(dones[i])
                
                current_ep_scores[i] += rewards[i]
                current_ep_steps[i] += 1
                
                if dones[i]:
                    score = infos[i].get('total_score', current_ep_scores[i])
                    score_history.append(score)
                    
                    match_rate = current_ep_matches[i] / current_ep_steps[i] if current_ep_steps[i] > 0 else 0
                    
                    current_ep_scores[i] = 0
                    current_ep_matches[i] = 0
                    current_ep_steps[i] = 0
                    i_episode += 1

                    if i_episode % 10 == 0:
                        avg = np.mean(score_history) if score_history else 0
                        lr = agent.optimizer.param_groups[0]['lr']
                        writer.add_scalar("Training/Avg_Score", avg, time_step)
                        scheduler.step(avg)
                        
                        pbar.set_postfix({
                            'Ep': i_episode,
                            'Avg': f"{avg:.1f}",
                            'Loss': f"{last_loss:.3f}",
                            'LR': f"{lr:.1e}"
                        })

            obs = next_obs
            solver_acts = next_solver_acts
            time_step += NUM_ENVS
            pbar.update(NUM_ENVS)

            if len(agent.buffer['states']) >= UPDATE_TIMESTEP:
                loss = agent.update()
                agent.clear_buffer()
                last_loss = loss
                writer.add_scalar("Training/Loss", loss, time_step)

            if time_step % SAVE_INTERVAL < NUM_ENVS:
                path = os.path.join(CHECKPOINT_DIR, f"ppo_multi_{time_step}.pth")
                torch.save(agent.policy.state_dict(), path)
                pbar.write(f"💾 Saved: {path}")

    except KeyboardInterrupt:
        print("🛑 停止訓練...")
    finally:
        envs.close()
        writer.close()
        pbar.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train()