# Machine-Learning-and-Practice
# Match-3-game-With-AI
（DQN & PPO）

本專案實作一個三消（Match-3）遊戲環境，並使用兩種常見的強化學習方法訓練Agent：深度 Q-learning（DQN）與近端策略優化（PPO）。<br>此 README 紀錄以 **專案管理** 的模式進行AI專題，並逐步說明 **需求**、**系統分析**、**設計**三大項目及其小項目，希望我們會在無意間完成專案。

## 目標
- 建立可重複、可評估的 Match-3 環境（類似 OpenAI Gym 介面）。
- 實作並比較 DQN（value-based）與 PPO（policy-gradient）的行為表現與收斂特性。
- 提供訓練、評估與視覺化工具，便於分析學習過程與策略行為。

## 三消遊戲簡述
- 玩法：交換相鄰方塊，若形成 3 個或以上相同顏色連線則消除並獲得分數，消除後方塊下落並補充新方塊，可能產生連鎖。
- 狀態（Observation）：棋盤格子顏色矩陣（可附加當前分數、剩餘步數或回合資訊）。
- 動作（Action）：選擇一個格子與鄰格交換（視實作可編碼為離散集合，或只允許有效交換）。
- 獎勵（Reward）：每次消除的分數；可額外給予連鎖獎勵或回合結束的總分獎勵以鼓勵長期規劃。

## 為什麼選 DQN 與 PPO？
- DQN：適合離散動作空間（例如固定的交換動作集合），透過 Q-value 評估行為價值，實作相對直觀。
- PPO：基於策略梯度，更新穩定且常具較好樣本效率，對需長期規劃（如最大化連鎖）場景通常有優勢。

## 系統架構建議
- env/ — Match-3 環境實作（reset(), step(action), render()），最好遵循 Gym-like API。
- agents/ — DQN 與 PPO 代理程式碼（模型、優化器、回放緩衝、訓練迴圈）。
- train/ — 訓練腳本與超參數設定（可輸入 CLI 參數）。
- eval/ — 評估與視覺化（紀錄 learning curve、重放錄影、行為分析）。
- docs/ — 實驗結果與圖表。

## 快速上手（範例）
1. 建議使用虛擬環境並安裝依賴：
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. 訓練 DQN（範例）：
```bash
python train/train_dqn.py --env Match3Env --episodes 2000 --save-dir models/dqn
```

3. 訓練 PPO（範例）：
```bash
python train/train_ppo.py --env Match3Env --timesteps 200000 --save-dir models/ppo
```

4. 使用已訓練模型測試與渲染：
```bash
python eval/play.py --model models/ppo/best.pt --env Match3Env --render
```

## 超參數建議（起點）
- DQN: lr=1e-4、epsilon 起始 1.0 -> 0.1 緩慢衰減、batch_size=32、memory_size=1e5、target_update=1000 steps。
- PPO: lr=3e-4、clip_eps=0.2、update_epochs=10、batch_size=64、gamma=0.99、gae_lambda=0.95。

注意：若採用 CNN 或 transformer 來處理棋盤影像表示，網路架構與學習率可能須調整。

## 評估指標
- 平均得分（Mean Score）
- 達成門檻的成功率（例如超過某分數占比）
- 平均連鎖長度（衡量策略誘發連鎖的能力）
- 收斂速度（Reward vs Episodes/Timesteps）
請同時保存隨機種子與訓練設定以確保結果可重現。

## 範例觀察（可放置圖表/影片）
- DQN 在簡化規則下可學會基本消除選擇，但在長期規劃和連鎖策略上表現有限。
- PPO 常較穩定收斂，能在某些情況下學到更具全局性的策略以提升連鎖次數。

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

如果你要我把這份 README 寫入你的 repository，我可以代為更新 main 分支（或其他你指定的分支）。請回覆：
- 「更新 README」：我會將上面的內容寫入 README.md。
- 「先不要」或提出修改意見：我會依你的要求調整內容後再更新。若你想要我直接把某些段落改成英語或加入實驗圖表，也請說明。