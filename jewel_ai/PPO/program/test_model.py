import torch
import numpy as np
import time
import os

# 引入你的環境與 Agent
from jewel_env.GAME_jewel_env import JewelEnv
from PPO_Agent_Multitasking import PPOAgent

def test():
    # ==========================================
    # 1. 設定參數 (必須與訓練時完全一致)
    # ==========================================
    MODEL_PATH = "pretrained_models\run_20251215_160236\model_interrupted.pth"
    INPUT_SHAPE = (10, 9, 6)   # (Channel, Height, Width)
    NUM_ACTIONS = 46           # 9 * 5 + 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 測試幾局
    TEST_EPISODES = 5
    # 每一動作的延遲時間 (秒)，設為 0.2 ~ 0.5 方便你肉眼觀察
    RENDER_DELAY = 0.01

    # ==========================================
    # 2. 初始化環境與 Agent
    # ==========================================
    # 注意：測試時我們使用 "simple" 或 "advanced" 模式都可以，重點是看動作
    env = JewelEnv(reward_mode="simple") 
    
    agent = PPOAgent(
        input_shape=INPUT_SHAPE,
        num_actions=NUM_ACTIONS,
        device=DEVICE,
        lr=0.0003,      # 測試時 LR 不重要，因為不更新
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4
    )

    # ==========================================
    # 3. 載入模型權重
    # ==========================================
    if os.path.exists(MODEL_PATH):
        print(f"📂 載入模型權重: {MODEL_PATH}")
        agent.policy.load_state_dict(torch.load(MODEL_PATH))
        agent.policy.eval() # 🔥 設定為評估模式 (Evaluation Mode)
    else:
        print(f"❌ 找不到模型檔案 {MODEL_PATH}，請先執行 pretrain_ppo.py")
        return

    # ==========================================
    # 4. 開始測試
    # ==========================================
    print(f"🚀 開始測試 {TEST_EPISODES} 局遊戲...")
    
    total_rewards = []
    total_cleared = []

    for episode in range(TEST_EPISODES):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        print(f"\n🎮 Episode {episode + 1} Start!")
        env.render() # 顯示視窗

        while not done:
            # 取得 AI 的動作
            # 注意：這裡 action_mask 傳 None，測試 AI 是否學會自己避開無效動作
            action, log_prob, val = agent.select_action(obs, action_mask=None)
            
            # 因為 select_action 回傳的是 numpy array，如果是單一數值取出它
            if isinstance(action, np.ndarray):
                action = action[0]

            # 執行動作
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            step_count += 1
            obs = next_obs
            
            # 顯示動作類型 (方便 Debug)
            if action == 45: # Upload
                act_str = "⬆️ Upload"
            else:
                r = action // 5
                c = action % 5
                act_str = f"Swap ({r},{c})"

            print(f"Step {step_count}: {act_str} | Reward: {reward:.2f} | Cleared: {info['cleared']}")
            
            env.render() # 更新畫面
            time.sleep(RENDER_DELAY) # 暫停一下讓你看到動作

        print(f"🏁 Episode {episode + 1} 結束 | 總分: {episode_reward:.2f} | 消除總數: {info['episode_cleared']}")
        total_rewards.append(episode_reward)
        total_cleared.append(info['episode_cleared'])

    # ==========================================
    # 5. 總結
    # ==========================================
    print("\n📊 測試總結:")
    print(f"平均分數: {np.mean(total_rewards):.2f}")
    print(f"平均消除寶石數: {np.mean(total_cleared):.2f}")
    
    # 簡單的基準比較
    print("-" * 30)
    print("💡 觀察重點：")
    print("1. AI 是否會優先消除寶石？(而不是隨機亂點)")
    print("2. AI 是否會試圖消除牆壁？(如果你的專家數據有教它)")
    print("3. AI 是否會頻繁使用無效交換？(如果是，代表預訓練不足或數據品質不佳)")

if __name__ == "__main__":
    test()