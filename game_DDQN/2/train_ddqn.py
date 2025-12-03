import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import subprocess
import webbrowser
import time
import os

from ddqn_agent import DDQNAgent
from jewel_env.GAME_jewel_env_cursor import JewelCursorEnv


# ================================
# 🔥 自動建立獨立的 log 與 model 資料夾
# ================================
timestamp = time.strftime("%Y%m%d-%H%M%S")

BASE_LOG_DIR = "runs/DDQN_JewelCursor"
LOG_DIR = os.path.join(BASE_LOG_DIR, f"run_{timestamp}")

BASE_MODEL_DIR = "saved_models/DDQN_JewelCursor"
MODEL_DIR = os.path.join(BASE_MODEL_DIR, f"run_{timestamp}")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ================================
# 🔥 其他超參數
# ================================
EPISODES = 20000
BATCH_SIZE = 64
TARGET_UPDATE = 200
ACTION_SIZE = 7
LR = 1e-4
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.9995
TENSORBOARD_PORT = 6006


# ================================
# 🔥 自動啟動 TensorBoard
# ================================
def launch_tensorboard(logdir=LOG_DIR, port=TENSORBOARD_PORT):

    print("🔄 啟動 TensorBoard ...")

    tb_process = subprocess.Popen(
        ["tensorboard", f"--logdir={logdir}", f"--port={port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    time.sleep(5)

    url = f"http://localhost:{port}"
    print(f"🌐 自動開啟 TensorBoard: {url}")
    webbrowser.open(url)

    return tb_process


# ================================
# 🔥 主訓練
# ================================
def main():
    tb = launch_tensorboard()

    writer = SummaryWriter(LOG_DIR)
    env = JewelCursorEnv(reward_mode="simple")
    agent = DDQNAgent(action_size=ACTION_SIZE, lr=LR, gamma=GAMMA)

    agent.epsilon = EPS_START
    agent.epsilon_min = EPS_END
    agent.epsilon_decay = EPS_DECAY

    print("CUDA:", torch.cuda.is_available())
    print("Model device:", next(agent.online_net.parameters()).device)

    best_reward = -1e9
    global_step = 0

    for ep in range(EPISODES):
        obs, info = env.reset()
        state = obs

        done = False
        total_reward = 0
        ep_steps = 0
        total_cleared = 0
        max_combo = 0

        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            ep_steps += 1
            total_cleared += info["cleared"]
            max_combo = max(max_combo, info["combo"])

            agent.replay.push(state, action, reward, next_obs, done)
            agent.update(BATCH_SIZE)

            state = next_obs
            global_step += 1

            if hasattr(agent, "loss_value"):
                writer.add_scalar("Train/Loss", agent.loss_value, global_step)
            if hasattr(agent, "last_q_mean"):
                writer.add_scalar("Train/Q_mean", agent.last_q_mean, global_step)

        # Episode summary
        writer.add_scalar("Episode/Reward", total_reward, ep)
        writer.add_scalar("Episode/Steps", ep_steps, ep)
        writer.add_scalar("Episode/Cleared", total_cleared, ep)
        writer.add_scalar("Episode/MaxCombo", max_combo, ep)
        writer.add_scalar("Train/Epsilon", agent.epsilon, ep)

        # reward component breakdown
        rb = info["reward_sum_components"]
        for key, val in rb.items():
            writer.add_scalar(f"Reward/{key}", val, ep)

        if ep % TARGET_UPDATE == 0:
            agent.update_target()

        print(f"EP {ep:5d} | R={total_reward:.2f} | Steps={ep_steps} | eps={agent.epsilon:.3f}")

        # ⭐ 每 1000 回合存一次
        if ep % 1000 == 0:
            save_path = os.path.join(MODEL_DIR, f"ddqn_ep{ep}.pth")
            torch.save(agent.online_net.state_dict(), save_path)
            print(f"💾 模型已儲存：{save_path}")

        # ⭐ 自動存最佳模型
        if total_reward > best_reward:
            best_reward = total_reward
            best_path = os.path.join(MODEL_DIR, "ddqn_best.pth")
            torch.save(agent.online_net.state_dict(), best_path)
            print(f"🌟 新最佳模型：{best_path}")

    writer.close()


if __name__ == "__main__":
    main()
