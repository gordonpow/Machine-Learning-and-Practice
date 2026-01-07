# 💎 Jewel Puzzle AI: 基於 PPO 的強化學習消除遊戲解題系統

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Reinforcement Learning](https://img.shields.io/badge/Algorithm-PPO-green)

本專案旨在利用 **深度強化學習 (Deep Reinforcement Learning)** 技術，訓練一個能夠精通「橫向卷軸消除遊戲」的 AI 代理 (Agent)。我們採用了 **自適應 PPO (Adaptive Proximal Policy Optimization)** 演算法，並結合了多核心並行訓練 (Multiprocessing) 與啟發式引導 (Heuristic Guidance)，讓 AI 能夠在複雜的隨機環境中學會「分組」、「連鎖」與「危機處理」。

---

## 📖 目錄

1.  [遊戲規則與環境](#1-遊戲規則與環境)
2.  [PPO 演算法介紹](#2-ppo-演算法介紹)
3.  [系統架構與訓練設計](#3-系統架構與訓練設計)
    * [神經網路模型 (CNN + Attention)](#31-神經網路模型-match3actorcritic)
    * [多核心並行訓練](#32-多核心並行訓練-subprocvecenv)
    * [啟發式引導 (Teacher Forcing)](#33-啟發式引導-teacher-forcing)
    * [自適應 PPO 機制](#34-自適應-ppo-機制-adaptive-mechanism)
4.  [訓練成果展示](#4-訓練成果展示)
5.  [如何執行](#5-如何執行)

---

## 1. 遊戲規則與環境

本專案模擬了一個經典的「橫向卷軸三消遊戲」(類似 Panel de Pon / Tetris Attack)。

### 🎮 遊戲機制
* **棋盤大小**：9 行 (Rows) x 6 列 (Cols)。
* **寶石種類**：7 種普通顏色 + 1 種牆壁 (Wall)。
* **動作空間 (Action Space)**：
    * **Swap**：點擊任意寶石，將其與**右邊**的寶石交換位置。
    * **Upload**：強制讓底部上升一層 (當頂層有空位時)。
* **消除規則**：
    * **3消 (Match-3)**：橫向或縱向 3 個同色寶石連線，即可消除。
    * **連鎖 (Chain/Combo)**：消除後上方的寶石會受到重力掉落，若再次形成消除，則觸發 Combo，分數加倍。
    * **牆壁 (Wall)**：無法直接交換，必須消除其周圍的寶石才能將牆壁轉化為普通寶石。
* **失敗條件 (Game Over)**：當寶石或牆壁觸碰到最頂層 (第 0 層) 並停留超過一定時間 (Game Over Timer)，遊戲結束。

### 🌟 環境挑戰
這是一個**部分可觀測 (Partially Observable)** 且 **隨機性極高** 的環境：
1.  **隨機牆壁**：底部會隨機生成無法移動的牆壁，擠壓生存空間。
2.  **連鎖預測**：AI 必須學會「預判」掉落後的盤面，而不僅僅是看眼前的消除。
3.  **危機處理**：在牆壁逼近頂層時，AI 必須從「貪分模式」切換到「生存模式」。

---

## 2. PPO 演算法介紹

我們選擇 **PPO (Proximal Policy Optimization)** 作為核心演算法，這是目前 OpenAI 最推薦的強化學習算法之一。

### 為什麼選擇 PPO？
相較於 DQN (Deep Q-Network) 或 A2C，PPO 有以下優勢：
* **穩定性 (Stability)**：PPO 限制了每次更新的幅度 (Clip)，防止模型因為一次壞的更新而崩潰。
* **樣本效率 (Sample Efficiency)**：支援多個 Epoch 重複使用同一批收集到的數據進行更新。
* **連續動作與離散動作皆適用**：非常適合這種策略梯度 (Policy Gradient) 的場景。

### PPO 核心公式
PPO 的目標函數如下：

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$

其中：
* $r_t(\theta)$ 是新舊策略的機率比值 (Ratio)。
* $\hat{A}_t$ 是優勢函數 (Advantage)，代表該動作比平均好多少。
* $\text{clip}$ 函數確保更新幅度不會超過 $\epsilon$ (通常設為 0.2)。

---

## 3. 系統架構與訓練設計

為了讓 AI 能夠學會如此複雜的遊戲，我們設計了一套完整的訓練系統。

### 3.1 神經網路模型 (Match3ActorCritic)

我們使用了一個共享卷積骨幹 (CNN Backbone) 的 Actor-Critic 架構，並設計了專門的 One-Hot 輸入層來處理遊戲盤面。

#### 📊 輸入層設計 (One-Hot Encoding Table)
原始盤面是 9x6 的整數矩陣（每個格子儲存寶石 ID），我們將其轉換為 **10 層通道 (Channels)** 的 One-Hot 格式。這意味著棋盤狀態被拆解為 **0 與 1 的二進位矩陣**，每一層都代表一種特定的寶石類型或特殊狀態。這樣做能讓 CNN 清楚區分不同的物件屬性，而非將 ID 視為連續數值。

| 通道索引 (Channel) | 代表物件 | 說明 |
| :---: | :--- | :--- |
| **0** | Empty (空) | 若該格為空氣，此層對應位置為 1，否則為 0 |
| **1** | Red Gem | 若該格為紅色寶石，此層對應位置為 1 |
| **2** | Blue Gem | 若該格為藍色寶石，此層對應位置為 1 |
| **3** | Green Gem | 若該格為綠色寶石，此層對應位置為 1 |
| **4** | Yellow Gem | 若該格為黃色寶石，此層對應位置為 1 |
| **5** | Purple Gem | 若該格為紫色寶石，此層對應位置為 1 |
| **6** | Orange Gem | 若該格為橘色寶石，此層對應位置為 1 |
| **7** | Cyan Gem | 若該格為青色寶石，此層對應位置為 1 |
| **8** | Wall (牆壁) | 若該格為牆壁，此層對應位置為 1 |
| **9** | **Action Mask** | **無效動作遮罩** (剛交換過的位置標記為 1，其餘為 0) |

> **設計理念**：將牆壁 (Wall) 與 Mask 獨立成單獨的通道，能讓卷積層更輕易地學會「牆壁不能動」以及「不要重複操作同一格」的規則。

![圖片範例](img/CNN_layer展示.png)
> **圖片說明** :紅色方框依照定義是Empty (空) 所對應位置的值就是1；綠色圓圈依照定義為Red Gem(紅色寶石)其 所對應位置的值就是1
#### 🧠 CNN 網路架構 (Network Architecture)
模型採用 3 層卷積層提取局部特徵，並引入 **Self-Attention** 機制來捕捉全域的連鎖關係。

| 層級 (Layer) | 輸入形狀 (Input Shape) | 操作 (Operation) | 輸出形狀 (Output Shape) | 說明 |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | `(10, 9, 6)` | - | - | 10 層 One-Hot 盤面 |
| **Conv1** | `(10, 9, 6)` | Conv2d(3x3, 32) + ReLU | `(32, 9, 6)` | 提取基礎紋理特徵 |
| **Conv2** | `(32, 9, 6)` | Conv2d(3x3, 64) + ReLU | `(64, 9, 6)` | 提取高階特徵 (如形狀) |
| **Conv3** | `(64, 9, 6)` | Conv2d(3x3, 64) + ReLU | `(64, 9, 6)` | 加深特徵表達 |
| **Attention**| `(64, 9, 6)` | **Self-Attention** | `(64, 9, 6)` | **捕捉全域連鎖與空間關係** |
| **Flatten** | `(64, 9, 6)` | Flatten | `(3456)` | 展平為向量 |
| **FC (Actor)** | `(3456)` | Linear(512) + ReLU | `(Action Dim)` | 輸出動作機率分佈 |
| **FC (Critic)**| `(3456)` | Linear(512) + ReLU | `(1)` | 輸出盤面價值 (Value) |

> **Self-Attention 的作用**：在消除遊戲中，底部的消除往往會引發頂部的掉落。傳統 CNN 的感受野 (Receptive Field) 有限，難以關聯相距較遠的方塊。Attention 機制允許模型「關注」盤面上任意兩個位置的關聯，從而學會預判連鎖。

### 3.2 多核心並行訓練 (SubprocVecEnv)
單一環境的採樣速度太慢，我們使用 `multiprocessing` 開啟 **18 個並行環境**：
* 每個 CPU 核心負責一個獨立的遊戲環境。
* 主進程收集 18 個環境的 `(State, Reward, Done)`，打包成 Batch 送入 GPU 訓練。
* 這讓訓練速度提升了約 **15 倍**。

### 3.3 啟發式引導 (Teacher Forcing)
在訓練初期，AI 只是隨機亂動，很難學會「消除」這個稀疏獎勵 (Sparse Reward)。
我們引入了一個基於 BFS (廣度優先搜索) 的 **Solver (專家系統)**：
1.  **Solver** 會計算當前盤面是否存在 3消、4消或破牆機會。
2.  在收集數據時，AI 有 **10% 的機率 (Teacher Forcing Rate)** 會被強制執行 Solver 建議的動作。
3.  如果 AI 自己選的動作跟 Solver 一樣，我們會給予額外的 **專家獎勵 (Expert Bonus)**。

這就像是教練手把手教 AI 下棋，讓它快速度過初期的迷茫階段。

### 3.4 自適應 PPO 機制 (Adaptive Mechanism)
為了確保模型在長時間訓練下的穩定性，我們實作了 **自適應學習率 (ReduceLROnPlateau)**。系統會持續監控「平均分數」，當 AI 進步停滯時自動降低學習率，進行更精細的優化。

---

## 4. 訓練成果展示

經過 1000 萬步 (約 24 小時) 的訓練，AI 展現出了驚人的策略演化：

### 📈 訓練曲線
*(此處可插入 TensorBoard 的 Reward 曲線截圖)*
* **階段 1 (0-1M 步)**：AI 學會了基本的 3 消，不再隨機亂按。
* **階段 2 (1M-5M 步)**：AI 開始學會「破牆優先」，存活時間大幅延長。
* **階段 3 (5M+ 步)**：AI 展現出「連鎖」行為，懂得先消除底部，引發上方的連鎖掉落。

### 🎥 實際遊玩演示 (GIF)
*(此處可插入 `gui_simulator.py` 錄製的 AI 遊玩 GIF)*

**觀察到的高階技巧**：
1.  **精準破牆**：當牆壁出現時，AI 會優先尋找牆壁旁的消除機會。
2.  **頂層急救**：當方塊堆到頂層時，AI 動作頻率明顯加快，且優先消除高層方塊。
3.  **拒絕無效步**：AI 幾乎不會做出無法消除的無效交換。

---

## 5. 如何執行

### 環境需求
* Python 3.8+
* PyTorch 2.0+
* Gymnasium
* NumPy, Pillow, Matplotlib

### 1. 安裝依賴
```bash
pip install torch gymnasium numpy pillow tensorboard tqdm
