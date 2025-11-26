# PingPong AI — DDQN 強化學習專案

本專案示範如何使用 **Double Deep Q-Network (DDQN)** 來訓練代理人遊玩來自 PAIA Arena 的乒乓球遊戲（Game 3）。此專案提供完整專案架構範本，包含環境介面、模型架構、訓練流程及推論示例。

---

## 📁 專案架構

```
pingpong_ai/
└── DDQN/
    ├── readme.md
    ├── config.py            # 超參數設定
    ├── environment.py       # 包裝 PAIA PingPong API 的環境
    ├── model.py             # DDQN 模型架構
    ├── replay_buffer.py     # 經驗回放記憶體
    ├── train.py             # 訓練主程式
    └── inference.py         # 推論 / 競賽用 Agent
```

---

## 🏓 遊戲介紹 — PAIA PingPong

官方連結：<https://app.paia-arena.com/zh/game/3>

此遊戲為一款 **固定視角的 2D 乒乓球對戰遊戲**，玩家可控制球拍移動方向，並透過策略擊球讓球落在對手區域得分。

### **狀態 (State)**
環境通常會提供：
- 球的位置 `(ball_x, ball_y)`
- 球的速度 `(velocity_x, velocity_y)`
- 玩家球拍位置
- 對手球拍位置
- 回合資訊

### **動作 (Action)**
代理行為可設計為：
- `0`: 不動
- `1`: 向上移動
- `2`: 向下移動
- 或者進階版：依球位置決策擊球方向

### **獎勵 (Reward) 設計示例**
- 得分：`+1`
- 失分：`-1`
- 球接觸球拍：`+0.1`
- 球靠近我方邊界：`-0.01`

---

## 🧠 演算法 — Double Deep Q-Network (DDQN)
DDQN 主要用於降低 Q-learning 在 max 動作選擇造成的 **過度估計問題**。

公式：
```
target = r + γ * Q_target(next_state, argmax_a Q_eval(next_state, a))
```

本專案使用：
- **Online Network**（選擇動作）
- **Target Network**（生成 Q 目標）
- **Replay Buffer**（打散樣本關聯性）

---

## ⚙️ 安裝與環境需求
```
pip install torch numpy requests
```

若需連接 PAIA API，請在 `environment.py` 中填入：
```
API_KEY = "your_api_key_here"
```

---

## 📘 使用方式

### **1. 訓練模型**
```
python train.py
```
訓練完成後會輸出：
```
checkpoints/ddqn_model.pth
```

### **2. 推論 / 比賽模式**
```
python inference.py
```
程式會自動載入最新模型並開始與 AI 或環境對戰。

---

## 📊 訓練過程 (Recommend)
你可以將訓練過程紀錄成：
- reward 曲線
- loss 曲線
- win rate 曲線

建議使用 matplotlib 或 tensorboard。

---

## 🔧 可調整超參數（config.py）
範例：
```
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 50000
BATCH_SIZE = 64
MEMORY_SIZE = 50000
TARGET_UPDATE_FREQ = 1000
```

---

## 📦 下一步可以改進
- 使用 CNN 處理影像輸入版本
- 使用 Prioritized Replay Buffer
- 使用 Rainbow DQN 或 PPO
- 加入球軌跡預測模型

---

## 📄 授權（可自由更改）
MIT License

---

如果你需要，我也可以為你自動產生整個專案所有 `.py` 文件的內容（含完整 DDQN 程式碼）。

