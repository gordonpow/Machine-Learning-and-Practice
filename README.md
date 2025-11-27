# Machine-Learning-and-Practice
# Match-3-game-With-AI
（DQN & PPO）

本專案實作一個三消（Match-3）遊戲環境，並使用兩種常見的強化學習方法訓練Agent：深度 Q-learning（DQN）與近端策略優化（PPO）。<br>此 README 紀錄以 **專案管理** 的模式進行AI專題，並逐步說明 **需求**、**系統分析**、**設計**三大項目及其小項目，希望我們會在無意間完成專案。

## 目標
- 建立可重複、可評估的 Match-3 環境。
- 實作並比較 DQN（value-based）與 DDQN（ Double Deep Q-Network）的行為表現與收斂特性。
- 提供訓練、評估與視覺化工具，便於分析學習過程與策略行為。

## 三消遊戲簡述
- 玩法：交換左右相鄰方塊，若形成 3 個或以上相同顏色連線則消除並獲得分數，消除後方塊下落並補充新方塊，可能產生連鎖。
- 狀態（State）：棋盤格子顏色矩陣（可附加當前分數、剩餘步數或回合資訊）。
- 動作（Action）：選擇一個格子與鄰格交換（視實作可編碼為離散集合，或只允許有效交換）。
- 獎勵（Reward）：每次消除的分數；可額外給予連鎖獎勵或回合結束的總分獎勵以鼓勵長期規劃。

## 為什麼選 DQN 與 DDQN？
- DQN：適合離散動作空間（例如固定的交換動作集合），透過 Q-value 評估行為價值，實作相對直觀。
- DDQN: Double DQN 可以避免Q-value估計過頭，

## DDQN 的 Loss Function
LossFunction是「機率+期望」推導出來的，源自「最大似然估計」或「期望最小化」，所以 **深度學習的 loss = 基於機率的誤差衡量** ，都是希望「最小化期望誤差」形成的。<br>DQN / DDQN 的 Loss，本質上就是**減少 TD 誤差（Temporal Difference Error**的期望值，min E[(TD Error)2]。
---

### 1. 經典 Q-Learning 的 TD-Target（表格形式）
在強化學習（Reinforcement Learning）中，Temporal Difference (TD) Learning 是一種重要的方法，它結合了蒙地卡羅方法（Monte Carlo）和動態規劃（Dynamic Programming）的優點，用來估計狀態-動作價值函數（Q-function）。TD Learning 的核心是透過「TD 誤差」（Temporal Difference Error）來更新價值估計，目的是最小化預測價值與實際觀察價值之間的期望誤差。這與深度學習中的 Loss Function 類似，都是基於機率與期望的框架，源自最大似然估計（Maximum Likelihood Estimation）或期望最小化（Expectation Minimization），目的是讓模型的預測盡可能接近真實分佈，從而最小化 E[(TD Error)^2]，即 TD 誤差的均方期望。
TD Learning 的更新規則基於 Bellman Equation，強調「現在的價值估計」應該等於「即時獎勵 + 未來折扣價值估計」。讓我們從 TD-target 的公式開始，逐步延伸到 DQN 和 DDQN 的 Loss Function。

<br>在 Q-Learning（TD Learning 的一種變體）中，我們使用 Q(s, a) 表示在狀態 s 下執行動作 a 的預期累積折扣獎勵。TD-Target 是用來計算 TD 誤差的「目標值」，它代表了「更好的」Q 值估計。
TD-Target 的基本公式為：

$$
y = r + \gamma \max_{a'} Q(s', a')
$$

其中：

- $r$: 當前獎勵（reward）。
- $\gamma$: 折扣因子（discount factor），通常介於 0 到 1 之間，用來權衡未來獎勵的重要性。
- $s'$: 下一個狀態（next state）。
- $a'$: 下一個狀態可能執行的動作。
- $\max_{a'} Q(s', a')$: 在下一個狀態 s' 中選擇最大 Q 值的動作，這是基於 greedy 策略的估計。

TD 誤差則是當前 Q 估計與 TD-Target 之間的差異：

$$
\delta = y - Q(s, a) = \left(r + \gamma \max_{a'} Q(s', a')\right) - Q(s, a)
$$

在傳統 Q-Learning 中，我們透過更新 Q 值來最小化這個誤差的期望：$Q(s, a) \leftarrow Q(s, a) + \alpha \delta$，其中 $\alpha$ 是學習率。這裡的 Loss 可以視為最小化 $E[\delta^2]$，以機率期望的形式減少誤差。


---

### 2. DQN（Deep Q-Network）的 TD-Target 與 Loss

延伸到 DQN 的 Loss Function
Deep Q-Network (DQN) 是 Q-Learning 的深度學習版本，它使用神經網絡（Q-Network）來近似 Q(s, a)，以處理高維狀態空間（如圖像）。DQN 保留了 TD-Target 的核心，但引入了經驗回放（Experience Replay）和目標網絡（Target Network）來穩定訓練。
在 DQN 中，TD-Target 的公式與上述相同：

**DQN 的 TD-Target**（第 i 筆經驗）：

$$
y_i^{\text{DQN}} = r_i + \gamma \max_{a'} Q(s_i', a'; \theta^-)
$$

這裡：

- $\theta^-$: 目標網絡的參數（Target Network），它是主網絡 $\theta$ 的延遲副本，用來穩定 TD-Target 的計算，避免訓練時的振盪。
- 下標 i 表示從經驗回放緩衝區中抽樣的轉移樣本 (s_i, a_i, r_i, s_i')。

DQN 的 Loss Function 是基於這些樣本的均方誤差（Mean Squared Error），目的是最小化 Q-Network 預測與 TD-Target 之間的期望誤差：
**DQN 的 Loss Function**（小批量均方誤差）：

$$
\boxed{
L_{\text{DQN}}(\theta) = 
\frac{1}{N}\sum_{i}^{N} 
\left(
    \underbrace{r_i + \gamma \max_{a'} Q(s_i', a'; \theta^-)}_{y_i^{\text{DQN}}}
    \;-\;
    Q(s_i, a_i; \theta)
\right)^2
}
$$

其中 N 是小批量樣本數。這裡的 Loss 直接來自 TD 誤差的平方期望，透過梯度下降最小化 E[(TD Error)^2]，讓 Q-Network 的輸出更接近真實的 Q 值分佈。這種設計源自期望最小化，假設誤差服從某種機率分佈（如高斯），最小化 MSE 等同於最大化似然。
然而，DQN 有 overestimation bias 的問題：因為 $\max_{a'} Q(s', a')$ 總是選擇最大值，噪音或初始誤差會導致系統性地高估 Q 值。
> 缺點：max 操作與同一組參數同時負責「選動作」與「評價值」→ 容易產生 **過高估計偏差（overestimation bias）**

---

### 3. DDQN（Double DQN）的 TD-Target 與 Loss
Double Deep Q-Network (DDQN) 是 DQN 的改進版本，專門解決 overestimation 問題。它引入「雙重估計」（Double Q-Learning）的想法，將「動作選擇」和「價值評估」分開：用主網絡選擇最佳動作，但用目標網絡評估其價值。
在 DDQN 中，TD-Target 的公式修改為：

- 用 **線上網絡（Online Network）** $ \theta $ 選出下一個狀態的最佳動作
- 用 **目標網絡（Target Network）** $ \theta^- $ 評估該動作的價值

**DDQN 的 TD-Target**：

$$
y_i^{\text{DDQN}} = r_i + \gamma \; 
Q\!\left(
    s_i', 
    \;\arg\max_{a'} Q(s_i', a'; \theta)\;;
    \theta^-
\right)
$$

這裡：

$\arg\max_{a'} Q(s_i', a'; \theta)$: 用主網絡 $\theta$ 選擇下一個狀態的最佳動作 a'（動作選擇）。
$Q(s_i', a'; \theta^-)$: 用目標網絡 $\theta^-$ 評估該動作的 Q 值（價值評估）。

這避免了單一網絡同時做選擇和評估導致的偏差。DDQN 的 Loss Function 與 DQN 類似，仍是 MSE：

<br>**DDQN 的 Loss Function**：

$$
\boxed{
L_{\text{DDQN}}(\theta) = 
\frac{1}{N}\sum_{i}^{N} 
\left(
    \underbrace{r_i + \gamma \; 
    Q\!\left(s_i', \arg\max_{a'} Q(s_i', a'; \theta);\; \theta^-\right)
    }_{y_i^{\text{DDQN}}}
    \;-\;
    Q(s_i, a_i; \theta)
\right)^2
}
$$

這樣就徹底解決了 DQN 的 overestimation 問題，訓練更穩定，性能大幅提升。
DDQN 的 Loss 同樣旨在最小化 TD 誤差的期望 E[(TD Error)^2]，但透過解耦動作選擇和評估，減少了 overestimation，從而讓訓練更穩定且性能更好。這延續了 DQN 的機率期望框架，但更精確地逼近真實 Q 值分佈。
---

### 總結對照表

| 方法       | TD-Target 公式                                                                                  | 是否有 Overestimation | Loss 本質                     |
|------------|--------------------------------------------------------------------------------------------------|-----------------------|-------------------------------|
| Q-Learning | $ r + \gamma \max_{a'} Q(s', a') $                                                               | 有                    | $ E[(\delta)^2] $             |
| DQN        | $ r + \gamma \max_{a'} Q(s', a'; \theta^-) $                                                     | 有（嚴重）            | $ \mathbb{E}[(y^{\text{DQN}} - Q)^2] $ |
| DDQN       | $ r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-) $                                  | 大幅減輕              | $ \mathbb{E}[(y^{\text{DDQN}} - Q)^2] $ |

三者的 Loss 都是 **最小化 TD 誤差平方的期望**，只是 TD-Target 的計算方式越來越精確，從而讓 Q 值估計更準確、代理表現更好。

## 系統架構拆解
![image](https://github.com/gordonpow/Machine-Learning-and-Practice/blob/main/jewel_ai/DDQN/Screenshot%202025-11-27%20123559.png?raw=true)



## 評估指標
- 平均得分（Mean Score）
- 達成門檻的成功率（例如超過某分數占比）
- 平均連鎖長度（衡量策略誘發連鎖的能力）
- 收斂速度（Reward vs Episodes/Timesteps）
請同時保存隨機種子與訓練設定以確保結果可重現。

## 範例觀察（可放置圖表/影片）
- DQN 在簡化規則下可學會基本消除選擇，但在長期規劃和連鎖策略上表現有限。

## 未來改進方向
- 使用模仿學習或自監督學習預訓練策略以加速收斂。
- 探索層次化強化學習（HRL）以處理長期目標（例如最大化連鎖）。
- 將觀察空間換成影像並採用 CNN/Transformer 進行特徵抽取。
- 結合規則引導（rule-based heuristics）與學習策略進行混合策略訓練。

## 參考文獻
- Mnih et al., 2015 — Human-level control through deep reinforcement learning (DQN)
- Schulman et al., 2017 — Proximal Policy Optimization Algorithms (PPO)
- OpenAI Gym: https://gym.openai.com/

---
