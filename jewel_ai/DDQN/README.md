# 📌 Project Overview
https://youtu.be/uMm5tCExLoE

## 🧠 演算法詳解：Proximal Policy Optimization (PPO)

本專案採用 **PPO (Proximal Policy Optimization)** 演算法，這是一種基於 Actor-Critic 架構的強化學習方法。它透過限制策略更新的幅度，解決了傳統 Policy Gradient 訓練不穩定的問題。

以下是核心公式與每一個數學符號的詳細定義。

### 1. 核心目標函數 (Clipped Surrogate Objective)

PPO 的靈魂在於其目標函數，透過 Clipping (截斷) 機制來防止策略發生劇烈變動：

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta)\hat{A}_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

#### 📝 符號定義與物理意義

| 符號 | 名稱 | 定義與功能解釋 |
| :--- | :--- | :--- |
| $\theta$ | **模型參數** | 代表神經網路 (Actor Network) 當前的權重 (Weights)。我們的目標就是找到一組最佳的 $\theta$。 |
| $\pi_\theta$ | **當前策略** | 由參數 $\theta$ 控制的策略網路。$\pi_\theta(a_t \| s_t)$ 代表在狀態 $s_t$ 下選擇動作 $a_t$ 的機率。 |
| $\hat{\mathbb{E}}_t$ | **期望值** | 表示對從環境中採樣出的數據 (Batch) 取平均值。 |
| $r_t(\theta)$ | **機率比率 (Ratio)** | $$\frac{\pi_\theta(a_t \| s_t)}{\pi_{\theta_{old}}(a_t \| s_t)}$$ <br> **意義**：衡量新策略相對於舊策略，對同一個動作的偏好程度變化。<br>• 若 $>1$：新策略更傾向選此動作。<br>• 若 $<1$：新策略降低選此動作的機率。 |
| $\hat{A}_t$ | **優勢函數 (Advantage)** | **意義**：衡量動作 $a_t$ 「比平均好多少」。<br>• 若 $\hat{A}_t > 0$：這是一個好動作，我們希望增加其機率 ($r_t$ 變大)。<br>• 若 $\hat{A}_t < 0$：這是一個壞動作，我們希望減少其機率 ($r_t$ 變小)。<br>本專案使用 **GAE (Generalized Advantage Estimation)** 來計算此值。 |
| $\epsilon$ | **Epsilon (截斷閾值)** | **超參數 (Hyperparameter)**，本專案設為 `0.2`。<br>它定義了策略更新的「安全範圍」為 $[1-\epsilon, 1+\epsilon]$，防止模型一次更新走太遠導致崩潰。 |
| $\text{clip}(\cdot)$ | **截斷函數** | 強制將 $r_t(\theta)$ 的值限制在 $[0.8, 1.2]$ (當 $\epsilon=0.2$) 之間。這是 PPO 穩定的關鍵。 |
| $\min$ | **最小值** | 取「原始目標」與「截斷後目標」的最小值。這是一種**悲觀下界 (Pessimistic Bound)** 策略，確保我們只在「有把握」的時候進行優化，避免過度樂觀的更新。 |

---

### 2. 總損失函數 (Total Loss Function)

由於我們使用 **Actor-Critic** 共享架構 (Shared Architecture)，模型需要同時學會「怎麼做 (Policy)」和「判斷局勢 (Value)」。因此，最終要最小化的 Loss 包含三部分：

$$L_t^{Total}(\theta) = \underbrace{- L_t^{CLIP}(\theta)}_{\text{Policy Loss}} + c_1 \underbrace{L_t^{VF}(\theta)}_{\text{Value Loss}} - c_2 \underbrace{S[\pi_\theta](s_t)}_{\text{Entropy Bonus}}$$

#### 📝 組成元件詳解

| 項目 | 符號 | 解釋 |
| :--- | :--- | :--- |
| **1. Policy Loss** | $-L^{CLIP}$ | **策略損失**。<br>因為 PyTorch 的優化器是做「最小化」，而 PPO 原意是「最大化」獎勵，所以這裡加上負號。 |
| **2. Value Loss** | $L^{VF}(\theta)$ | **價值損失** (Critic Loss)。<br>通常使用均方誤差 (MSE)：$(V_\theta(s_t) - V_{target})^2$。<br>**功能**：讓 Critic 網路能夠準確預測當前盤面的價值，這是計算 Advantage ($\hat{A}_t$) 的基礎。 |
| **3. Entropy** | $S[\pi_\theta]$ | **熵 (隨機性)**。<br>衡量策略分佈的混亂程度。<br>**功能**：我們希望最大化熵 (減去負的熵)，鼓勵 Agent 在訓練初期多嘗試不同動作 (Exploration)，避免過早收斂到局部最佳解。 |
| **係數** | $c_1, c_2$ | **權重係數**。<br>用來平衡這三項損失的重要性。<br>• $c_1$ (Value coef)：通常設為 `0.5`。<br>• $c_2$ (Entropy coef)：通常設為 `0.01`。 |

---







遊戲具有 **重力、隨機插入寶石、牆壁生成、游標移動、交換寶石** 等特殊規則，比一般 match-3 更複雜。

強化學習的目標是在有限步數內最大化消除寶石的數量並延長遊戲，不讓畫面因為牆壁堆滿而死亡。

---

# 🎮 Game Environment

## 🧩 Board Specification

- 棋盤大小：**9 × 6**
- 共有 **8 個 one-hot 通道** → shape = **(9, 6, 8)**

### Layer 含義

| Layer Index | Meaning |
|-------------|----------|
| 0 | 寶石 1 |
| 1 | 寶石 2 |
| 2 | 寶石 3 |
| 3 | 寶石 4 |
| 4 | 寶石 5 |
| 5 | 寶石 6 |
| 6 | 寶石 7 |
| 7 | 牆壁 |
| 8 | 游標位置 |

> 💡 **補充**  
> 雖然原本定義 9×6×8，但實際上還多一層游標資訊，可依需求調整為 **9×6×9**。

---

## 🎮 Actions (動作空間)

總共 **6 個動作**：

| Action ID | Description |
|-----------|-------------|
| 0 | 無動作 (No-op) |
| 1 | 游標上移 |
| 2 | 游標下移 |
| 3 | 游標左移 |
| 4 | 游標右移 |
| 5 | 點擊（與右邊寶石互換） |

---

# 🧱 Special Rules

## 🔹 初始狀態  
遊戲一開始會生成 **3 行不可消除的寶石**（random but fixed）。

## 🔹 游標移動  
透過上下左右移動游標，移到指定位置後可選擇點擊。

## 🔹 點擊交換寶石  

當執行 `action = 5` 時：

- ✔ 交換 **當前位置** 與 **右邊位置** 的寶石  
- ❗ **注意：只能與右邊交換（遊戲規則自定）**

---

# 🔍 消除判定

交換成功後若形成以下任一狀況會觸發消除：

- 3 連線  
- 4 連線  
- 5 連線  
- L 型  
- T 型  

並會進一步觸發 **重力掉落**。

---

# 🧲 重力系統

任何空格都會使該列上方的寶石自動往下掉落直到填滿。

---

# 📥 插入新行 (Row Insertion)

遊戲中有兩種插入方式：

1. **環境固定每 X 步自動插入一行隨機寶石**  
2. **玩家主動插入一行新寶石**

插入的行會將整個棋盤往上推，使遊戲難度提升。

---

# 🧱 牆壁系統 (Wall Mechanism)

- 當步數達到一定條件後，環境會**隨機產生一組牆壁**  
- 牆壁的 one-hot code = **7**  
- 牆壁會以 **一整組（連續多格且一次一排）** 生成  
- 牆壁 **無法被三消直接消除**

### ✔ 牆壁的正確消除方式  
必須消除 **牆壁相鄰的寶石**  
→ 相鄰寶石被消除後，牆壁會 **轉變成隨機寶石**

### ⚠ 如果牆壁被推到最頂端？
- 若牆壁到達最上層，遊戲將 **無法再插入新行**  
- 玩家需要優先處理牆壁，不然很容易 **Game Over**

---
![image](jewel_ai/DDQN/cleared.png)








