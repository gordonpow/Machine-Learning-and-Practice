
# 用於給AI訓練的遊戲架構
#加入第10層上一步動作及黑名單
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import copy
import tkinter as tk

ROWS, COLS = 9, 6
GEM_TYPES = 9
WALL = 7
EMPTY = 0
GAMEOVER_T = 20
wall_timer = 10
IDLE_UPLOAD_T = 15
GEM_TYPE_IDS = [1, 2, 3, 4, 5, 6, 8]

class JewelEnv(gym.Env):
    def __init__(self, reward_mode="simple"):
        super().__init__()
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.top_block_timer = 0
        self.combo = 0
        self.total_cleared = 0
        self.no_clear_steps = 0
        self.steps_in_episode = 0
        self.total_cleared_this_episode = 0
        self.last_action_coord = None
        
        # 動作空間：Swap + Upload
        self.action_space = spaces.Discrete(ROWS * (COLS - 1) + 1)
        
        # 觀測空間：10層 (9層寶石 + 1層無效操作遮罩)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10, 9, 6), dtype=np.float32)

        self.wall_interval = wall_timer
        self.wall_timer = self.wall_interval
        self.upload_timer = IDLE_UPLOAD_T
        self.falling_walls = []
        self.wall_cleared_count = 0
        self.flag_upload = False
        self.upload_no_cleared_step = 0
        self.reward_mode = reward_mode
        self.reward_sum_components = {}
        self.episode_score = 0
        self.same_action_count = 0
        self.same_color_swap = 0

        # 🔥 核心修改 1: 初始化第10層遮罩
        self.failed_action_mask = np.zeros((ROWS, COLS), dtype=np.float32)

        self.generate_initial_board()

    def get_state(self):
        return copy.deepcopy({
            "board": self.board.copy(),
            "top_block_timer": self.top_block_timer,
            "combo": self.combo,
            "total_cleared": self.total_cleared,
            "no_clear_steps": self.no_clear_steps,
            "steps_in_episode": self.steps_in_episode,
            "total_cleared_this_episode": self.total_cleared_this_episode,
            "last_action_coord": self.last_action_coord,
            "wall_timer": self.wall_timer,
            "upload_timer": self.upload_timer,
            "falling_walls": copy.deepcopy(self.falling_walls),
            "wall_cleared_count": self.wall_cleared_count,
            "reward_mode": self.reward_mode,
            "failed_action_mask": self.failed_action_mask.copy() # 保存遮罩狀態
        })

    def reset_to(self, state):
        self.board = state["board"].copy().astype(np.int32)
        self.top_block_timer = state["top_block_timer"]
        self.combo = state["combo"]
        self.total_cleared = state["total_cleared"]
        self.no_clear_steps = state["no_clear_steps"]
        self.steps_in_episode = state["steps_in_episode"]
        self.total_cleared_this_episode = state["total_cleared_this_episode"]
        self.last_action_coord = copy.deepcopy(state["last_action_coord"])
        self.wall_timer = state["wall_timer"]
        self.upload_timer = state["upload_timer"]
        self.falling_walls = copy.deepcopy(state["falling_walls"])
        self.wall_cleared_count = state["wall_cleared_count"]
        self.reward_mode = state["reward_mode"]
        self.failed_action_mask = state["failed_action_mask"].copy() # 還原遮罩
        return self._get_obs(), {}

    def _get_obs(self):
        gem_ids = [0, 1, 2, 3, 4, 5, 6, 8, 7]
        H, W = self.board.shape
        obs = np.zeros((len(gem_ids) + 1, H, W), dtype=np.float32)

        # 填入前 9 層 (寶石資訊)
        for i, g in enumerate(gem_ids):
            obs[i] = (self.board == g).astype(np.float32)

        # 🔥 核心修改 2: 第 10 層填入 "無效操作遮罩"
        obs[-1] = self.failed_action_mask
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.top_block_timer = 0
        self.combo = 0
        self.total_cleared = 0
        self.no_clear_steps = 0
        self.upload_timer = IDLE_UPLOAD_T
        self.wall_timer = self.wall_interval
        self.falling_walls.clear()
        self.wall_cleared_count = 0
        self.steps_in_episode = 0
        self.same_action_count = 0
        self.generate_initial_board()
        self.total_cleared_this_episode = 0
        self.flag_upload = False
        self.same_action_count = 0
        self.upload_no_cleared_step = 0
        self.reward_sum_components = {}
        self.episode_score = 0
        
        # 重置遮罩
        self.failed_action_mask = np.zeros((ROWS, COLS), dtype=np.float32)

        return self._get_obs(), {"episode_start": True}

    def upload_row(self):
            """棋盤底部上升一行"""
            possible = GEM_TYPE_IDS.copy()
            if np.any(self.board[0] != EMPTY):
                self.top_block_timer += 1
                return

            # 嘗試多次以生成符合條件的行
            for _ in range(50): 
                new_row = [random.choice(possible) for _ in range(COLS)]
                
                # 🔥 修改重點 1: 檢查該行是否至少有兩個相同的寶石
                # 如果 set(new_row) 的長度等於原長度，代表沒有重複元素 -> 跳過重抽
                if len(set(new_row)) == len(new_row):
                    continue

                simulated_board = np.copy(self.board[1:])
                simulated_board = np.vstack([simulated_board, np.array(new_row)])

                # 🔥 修改重點 2: 檢查是否會造成直接消除 (無法融合)
                if not self.has_immediate_match(simulated_board):
                    # 符合所有條件，執行更新
                    self.board[:-1] = self.board[1:]
                    self.board[-1] = new_row
                    
                    # Mask 跟著上移
                    self.failed_action_mask[:-1] = self.failed_action_mask[1:]
                    self.failed_action_mask[-1] = 0.0

                    self.upload_timer = IDLE_UPLOAD_T
                    return

            # Fallback: 如果隨機生成一直失敗 (機率極低)，強制插入最後一次生成的行
            # 為了符合 "至少兩個相同" 的規則，如果最後生成的行全都不同，強制改一個
            if len(set(new_row)) == len(new_row):
                new_row[random.randint(1, COLS-1)] = new_row[0] # 強制讓某個位置跟第一個一樣

            self.board[:-1] = self.board[1:]
            self.board[-1] = new_row
            
            self.failed_action_mask[:-1] = self.failed_action_mask[1:]
            self.failed_action_mask[-1] = 0.0
            
            self.upload_timer = IDLE_UPLOAD_T

    def step(self, action):
            self.steps_in_episode += 1
            cleared = 0
            done = False
            self.combo = 0
            
            if isinstance(action, (list, np.ndarray)):
                action = action[0]

            reward_components = {
                "same_action": 0, "cleared": 0, "combo": 0, "bonus": 0,
                "wall": 0, "penalty": 0, "gameover": 0, "duration_bonus": 0,
                "no_clear_penalty": 0, "first_combo": 0, "upload": 0,
                "same_color_swap": 0,
                "mask_penalty": 0 # Mask 懲罰
            }

            # 處理 upload
            if action == ROWS * (COLS - 1):
                self.last_action_coord = None
                self.upload_no_cleared_step += 1
                self.flag_upload = True
                is_swap = False
            else:
                is_swap = True
                row = action // (COLS - 1)
                col = action % (COLS - 1)
                self.last_action_coord = (row, col)

                # 🔥 檢查 mask 並扣分
                if self.failed_action_mask[row][col] == 1.0:
                    reward_components["mask_penalty"] = -0.05

                if self.same_action_count >= 1:
                    reward_components["same_action"] = -0.3

                # 執行交換
                if self.board[row][col] == self.board[row][col + 1]:
                    reward_components["same_color_swap"] = -0.2
                else:
                    if self.board[row][col] != WALL and self.board[row][col + 1] != WALL:
                        self.board[row][col], self.board[row][col + 1] = \
                            self.board[row][col + 1], self.board[row][col]

            # Upload Timer
            if self.upload_timer <= 0 or action == (ROWS * (COLS - 1)):
                self.upload_row()
            else:
                self.upload_timer -= 1

            # 結算消除
            cleared = self.resolve()
            self.drop_walls()

            # 🔥 新增：計算牆壁懲罰 (Wall Punishment)
        # ==========================================
            wall_count = np.sum(self.board == WALL)
            if wall_count > 0:
                # 係數可以調整：
                # -0.1 : 輕微壓力
                # -0.5 : 強大壓力 (AI 會發瘋似地想消牆壁)
                reward_components["wall_existence_penalty"] =-0.05





            if self.reward_mode != "simple":
                self.wall_timer -= 1
                if self.wall_timer <= 0:
                    self.insert_wall()
                    self.wall_timer = self.wall_interval
                    self.drop_walls()
                    self.gravity()




            # 🔥 更新 Mask 狀態
            if cleared > 0:
                self.failed_action_mask.fill(0.0) # 有消除全清空
                self.no_clear_steps = 0
            else:
                self.no_clear_steps += 1
                if is_swap:
                    r, c = self.last_action_coord
                    self.failed_action_mask[r][c] = 1.0 # 沒消除標記

            # 🔥 修正這裡！把原本不見的模式判斷加回來
            if self.reward_mode == "simple":
                
                reward_components["cleared"] = 0.5 * cleared 
                if cleared == 0:
                    reward_components["no_clear_penalty"] = -0.01
                else:
                    reward_components["cleared"] += 0.2
                
                if self.top_block_timer >= GAMEOVER_T:
                    reward_components["gameover"] = -10.0
                    done = True
                
                reward_components["survival"] = 0.001

            elif self.reward_mode == "combo":
                reward_components["cleared"] = 1.5 * cleared
                reward_components["combo"] = 0.05 * (self.combo ** 1.5)
                if cleared >= 5:
                    reward_components["bonus"] = 0.3
                reward_components["wall"] = 0.1 * self.wall_cleared_count
                if cleared == 0:
                    reward_components["no_clear_penalty"] = -0.1

                if self.top_block_timer >= GAMEOVER_T:
                    reward_components["gameover"] = -1.0
                    done = True

            elif self.reward_mode == "advanced":
                reward_components["cleared"] = 0.15 * cleared
                # reward_components["combo"] = 0.05 * self.combo
                if cleared >= 5:
                    reward_components["bonus"] = 0.2
                if self.wall_cleared_count >= 1:
                    reward_components["wall"] = 1
                # reward_components["penalty"] = -0.05 * (self.top_block_timer / GAMEOVER_T)

                if self.top_block_timer >= GAMEOVER_T:
                    reward_components["gameover"] = -1.0
                    done = True

            # 加總 Reward
            for key, val in reward_components.items():
                if key not in self.reward_sum_components:
                    self.reward_sum_components[key] = 0.0
                self.reward_sum_components[key] += val

            reward = float(sum(reward_components.values()))
            self.episode_score += reward
            self.total_cleared_this_episode += cleared

            info = {
                "episode_start": False,
                "cleared": cleared,
                "combo": self.combo,
                "gameover": done,
                "wall_cleared_count": self.wall_cleared_count,
                "reward_breakdown": reward_components
            }
            
            return self._get_obs(), float(reward), bool(done), False, info

    # --- 以下為輔助函數 (保持不變) ---
    def gravity(self):
        for j in range(COLS):
            for i in range(ROWS-2, -1, -1):
                if self.board[i][j] != WALL:
                    r = i
                    while r+1 < ROWS and self.board[r+1][j] == EMPTY:
                        self.board[r+1][j], self.board[r][j] = self.board[r][j], EMPTY
                        r += 1

    def drop_walls(self):
        """
        牆壁掉落邏輯：
        1. 找出所有相連的牆壁群組。
        2. 計算每個群組作為一個剛體，最大能向下移動多少距離（遇到任何障礙即停止）。
        3. 移動群組。
        """
        visited = set()
        
        # 從下往上掃描，確保下方的牆先處理（雖然一次計算到底，但順序仍有助於邏輯）
        for r in range(ROWS - 1, -1, -1):
            for c in range(COLS):
                if self.board[r][c] == WALL and (r, c) not in visited:
                    # 1. 找出這塊牆的所有相連部分
                    group = self._get_wall_group(r, c)
                    visited.update(group)
                    
                    # 2. 計算這個群組最大能掉落的距離
                    drop_dist = 0
                    while True:
                        can_fall_further = True
                        # 檢查群組內每一個方塊，如果往下移一格會怎樣？
                        for wr, wc in group:
                            next_r = wr + drop_dist + 1
                            
                            # 條件A: 碰到邊界 -> 不能再掉
                            if next_r >= ROWS:
                                can_fall_further = False
                                break
                            
                            # 條件B: 碰到非空的格子，且該格子不屬於自己這個群組 -> 不能再掉
                            # (如果是自己群組的格子，例如直條牆，當然可以往下穿過自己原來的位置)
                            if self.board[next_r][wc] != EMPTY and (next_r, wc) not in group:
                                can_fall_further = False
                                break
                        
                        if can_fall_further:
                            drop_dist += 1
                        else:
                            break
                    
                    # 3. 執行移動
                    if drop_dist > 0:
                        # 先將原位置清空
                        for wr, wc in group:
                            self.board[wr][wc] = EMPTY
                        # 再填入新位置
                        for wr, wc in group:
                            self.board[wr + drop_dist][wc] = WALL

    def _get_wall_group(self, start_r, start_c):
        """使用 BFS 找出所有相連的牆壁座標"""
        group = set()
        stack = [(start_r, start_c)]
        group.add((start_r, start_c))
        
        while stack:
            r, c = stack.pop()
            # 檢查上下左右 (4-Way Connectivity)
            # 如果希望牆壁只在水平方向連結，這裡可以改成只檢查 (0, -1) 和 (0, 1)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    if self.board[nr][nc] == WALL and (nr, nc) not in group:
                        group.add((nr, nc))
                        stack.append((nr, nc))
        return group

    def insert_wall(self):
        if np.any(self.board[0] != EMPTY):
            self.top_block_timer += 1
            return
        width = random.randint(2, 6)
        start_col = random.randint(0, COLS - width)
        self.falling_walls.append((0, start_col, width))
        for j in range(start_col, start_col + width):
            self.board[0][j] = WALL
            
    def resolve(self):
        cleared_positions = set()
        total_cleared = 0
        self.wall_cleared_count = 0
        while True:
            self.gravity()
            to_clear = set()
            for i in range(ROWS):
                for j in range(COLS):
                    val = self.board[i][j]
                    if val in (EMPTY, WALL): continue
                    hor = [(i, j)]
                    for dj in range(1, COLS - j):
                        if self.board[i][j + dj] == val: hor.append((i, j + dj))
                        else: break
                    if len(hor) >= 3: to_clear.update(hor)
                    ver = [(i, j)]
                    for di in range(1, ROWS - i):
                        if self.board[i + di][j] == val: ver.append((i + di, j))
                        else: break
                    if len(ver) >= 3: to_clear.update(ver)
            if not to_clear: break
            self.combo += 1
            total_cleared += len(to_clear)
            cleared_positions.update(to_clear)
            for i, j in to_clear: self.board[i][j] = EMPTY
            self.gravity()
            triggered_walls = set()
            for i, j in to_clear:
                for dx, dy in [(-1, 0)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < ROWS and 0 <= nj < COLS and self.board[ni][nj] == WALL:
                        triggered_walls.add((ni, nj))
            wall_to_convert = set()
            visited = set()
            for i, j in triggered_walls:
                if (i, j) in visited: continue
                stack = [(i, j)]
                group = set()
                while stack:
                    x, y = stack.pop()
                    if (x, y) in visited or self.board[x][y] != WALL: continue
                    visited.add((x, y))
                    group.add((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < ROWS and 0 <= ny < COLS and self.board[nx][ny] == WALL:
                            stack.append((nx, ny))
                wall_to_convert |= group
            if wall_to_convert:
                self.wall_cleared_count = len(wall_to_convert)
                possible = GEM_TYPE_IDS.copy()
                for i, j in wall_to_convert: self.board[i][j] = random.choice(possible)
            self.gravity()
        return total_cleared

    def has_immediate_match(self, board):
        for i in range(ROWS):
            for j in range(COLS):
                val = board[i][j]
                if val in (EMPTY, WALL): continue
                if j >= 2 and val == board[i][j - 1] == board[i][j - 2]: return True
                if j >= 1 and j < COLS - 1 and val == board[i][j - 1] == board[i][j + 1]: return True
                if j < COLS - 2 and val == board[i][j + 1] == board[i][j + 2]: return True
                if i >= 2 and val == board[i - 1][j] == board[i - 2][j]: return True
                if i >= 1 and i < ROWS - 1 and val == board[i - 1][j] == board[i + 1][j]: return True
                if i < ROWS - 2 and val == board[i + 1][j] == board[i + 2][j]: return True
        return False
    
    def generate_initial_board(self):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        for i in range(ROWS - 3, ROWS):
            for j in range(COLS):
                while True:
                    gem = random.choice(GEM_TYPE_IDS)
                    if j >= 2 and gem == self.board[i][j - 1] == self.board[i][j - 2]: continue
                    if i >= 2 and gem == self.board[i - 1][j] == self.board[i - 2][j]: continue
                    self.board[i][j] = gem
                    break


    def resolve(self):
        cleared_positions = set()
        total_cleared = 0
        self.wall_cleared_count = 0

        while True:
            self.gravity()
            to_clear = set()

            # === 找出可以消除的格子 ===
            for i in range(ROWS):
                for j in range(COLS):
                    val = self.board[i][j]
                    if val in (EMPTY, WALL):
                        continue
                    hor = [(i, j)]
                    for dj in range(1, COLS - j):
                        if self.board[i][j + dj] == val:
                            hor.append((i, j + dj))
                        else:
                            break
                    if len(hor) >= 3:
                        to_clear.update(hor)

                    ver = [(i, j)]
                    for di in range(1, ROWS - i):
                        if self.board[i + di][j] == val:
                            ver.append((i + di, j))
                        else:
                            break
                    if len(ver) >= 3:
                        to_clear.update(ver)

            if not to_clear:
                break

            # === 消除寶石 ===
            self.combo += 1
            total_cleared += len(to_clear)
            cleared_positions.update(to_clear)

            for i, j in to_clear:
                self.board[i][j] = EMPTY

            self.gravity()

            # === 消除周圍的牆壁 ===
            triggered_walls = set()
            for i, j in to_clear:
                for dx, dy in [(-1, 0)]:  # 只看牆壁在上方的
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < ROWS and 0 <= nj < COLS and self.board[ni][nj] == WALL:
                        triggered_walls.add((ni, nj))

            wall_to_convert = set()
            visited = set()
            for i, j in triggered_walls:
                if (i, j) in visited:
                    continue
                stack = [(i, j)]
                group = set()
                while stack:
                    x, y = stack.pop()
                    if (x, y) in visited or self.board[x][y] != WALL:
                        continue
                    visited.add((x, y))
                    group.add((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < ROWS and 0 <= ny < COLS and self.board[nx][ny] == WALL:
                            stack.append((nx, ny))
                wall_to_convert |= group

            if wall_to_convert:
                self.wall_cleared_count = len(wall_to_convert)
                possible = GEM_TYPE_IDS.copy()
                for i, j in wall_to_convert:
                    self.board[i][j] = random.choice(possible)

            self.gravity()

        return total_cleared



    def gravity(self):
        for j in range(COLS):
            for i in range(ROWS-2, -1, -1):
                if self.board[i][j] != WALL:
                    r = i
                    while r+1 < ROWS and self.board[r+1][j] == EMPTY:
                        self.board[r+1][j], self.board[r][j] = self.board[r][j], EMPTY
                        r += 1



    def insert_wall(self):
        if np.any(self.board[0] != EMPTY):
            self.top_block_timer += 1
            return
        width = random.randint(2, 6)
        start_col = random.randint(0, COLS - width)
        self.falling_walls.append((0, start_col, width))
        for j in range(start_col, start_col + width):
            self.board[0][j] = WALL



    def drop_walls(self):
            visited = set()

            for row in range(ROWS - 1, -1, -1):
                for col in range(COLS):
                    if (row, col) in visited or self.board[row][col] != WALL:
                        continue

                    # 使用 DFS 找出一塊牆群（T/L型都可以）
                    stack = [(row, col)]
                    group = set()
                    while stack:
                        r, c = stack.pop()
                        if (r, c) in visited or self.board[r][c] != WALL:
                            continue
                        visited.add((r, c))
                        group.add((r, c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < ROWS and 0 <= nc < COLS and self.board[nr][nc] == WALL:
                                stack.append((nr, nc))

                    # 將該群組盡可能向下移動
                    can_move = True
                    while can_move:
                        # 檢查是否整塊群組都能往下掉一格
                        for r, c in group:
                            if r + 1 >= ROWS or (self.board[r + 1][c] != EMPTY and (r + 1, c) not in group):
                                can_move = False
                                break
                        if can_move:
                            # 先清空目前位置（注意反向排序）
                            for r, c in sorted(group, reverse=True):
                                self.board[r][c] = EMPTY
                            # 下移一格
                            new_group = set()
                            for r, c in group:
                                self.board[r + 1][c] = WALL
                                new_group.add((r + 1, c))
                            group = new_group

    # def render(self):
    #     for i in range(ROWS):
    #         row_str = ""
    #         for j in range(COLS):
    #             val = self.board[i][j]
    #             symbol = "." if val == EMPTY else str(val)

    #             # 標記被點擊的格子
    #             if self.last_action_coord == (i, j):
    #                 symbol = f" \033[91m{val}\033[0m " 
    #             else:
    #                 symbol = f" {symbol} "

    #             row_str += symbol
    #         print(row_str)
    #     print()

    def calculate_adjacent_match_reward(self, row, col):
        """計算與上下左右是否有同色寶石，每個方向 +0.05"""
        if row < 0 or row >= ROWS or col < 0 or col >= COLS:
            return 0.0
        gem = self.board[row][col]
        if gem in [WALL, EMPTY]:
            return 0.0
        reward = 0.0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r2, c2 = row + dr, col + dc
            if 0 <= r2 < ROWS and 0 <= c2 < COLS:
                if self.board[r2][c2] == gem:
                    reward += 0.05
        return reward

    


    def render(self, mode="human"):
        if not hasattr(self, "_tk_root"):
            self._tk_root = tk.Tk()
            self._tk_root.title("Jewel Puzzle AI Viewer")
            self._canvas = tk.Canvas(self._tk_root, width=6*50, height=9*50)
            self._canvas.pack()
            self._tk_root.update()

        self._canvas.delete("all")

        color_map = {
            0: "#000000",  # EMPTY
            1: "#9D23DE",
            2: "#52A778",
            3: "#DED59A",
            4: "#032E91",
            5: "#C82CAE",
            6: "#DD8921",
            7: "#FFFFFF",  # WALL
            8: "#FF0000",  # 可用於特殊寶石
        }

        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                val = int(self.board[row, col])
                x0, y0 = col * 50, row * 50
                x1, y1 = x0 + 50, y0 + 50
                color = color_map.get(val, "#000000")
                self._canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
                self._canvas.create_text((x0 + x1)//2, (y0 + y1)//2, text=str(val), fill="white", font=("Arial", 12))

        # Highlight selected action
        if self.last_action_coord is not None:
            row, col = self.last_action_coord
            
            x0, y0 = col * 50, row * 50
            x1, y1 = x0 + 50, y0 + 50
            self._canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=3)

        self._tk_root.update()








    def check_gameover(self):
        if self.top_block_timer >= GAMEOVER_T:
            self.top_block_timer = 0
            self.generate_initial_board()
            self.upload_timer = IDLE_UPLOAD_T
            self.combo = 0
            return True
        else:
            return False
        
    




# -----------------------------------主程式--------------------------------------

if __name__ == "__main__":
    env = JewelEnv()
    # obs = env.reset()
    # env.render()
    # game_over_count = 0
    # for _ in range(100):
        
        # action = env.action_space.sample()

        # if action == ROWS * (COLS - 1):
        #     action_type = "Upload"
        #     coord_str = "⬆️ Upload"
        # else:
        #     row = action // (COLS - 1)
        #     col = action % (COLS - 1)
        #     action_type = "Swap"
        #     coord_str = f"click ({row}, {col}) ↔ ({row}, {col+1})"
        # obs, reward, terminated, truncated, info = env.step(action)
        # done = terminated or truncated
        # print(f"Action {action}: {action_type}, {coord_str}")
        # print(f"  → Reward: {reward}, Cleared: {info['cleared']}, Combo: {info['combo']}")

        # env.render()

    #     if done:
            # print("💀 Game Over!")
            # game_over_count += 1
            # obs = env.reset()

    # print(f"總共 Game Over 次數：{game_over_count}")


    env = JewelEnv(...)
    ep_rewards = []
    env.reward_mode = "simple"
    game_over_count = 0
    for _ in range(100):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        # print("💀 Game Over!")
        game_over_count += 1
        # obs = env.reset()
        while not done:
            action = env.action_space.sample()

            if action == ROWS * (COLS - 1):
                action_type = "Upload"
                coord_str = "⬆️ Upload"
            else:
                row = action // (COLS - 1)
                col = action % (COLS - 1)
                action_type = "Swap"
                coord_str = f"click ({row}, {col}) ↔ ({row}, {col+1})"
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # print(f"Action {action}: {action_type}, {coord_str}")
            # print(f"  → Reward: {reward}, Cleared: {info['cleared']}, Combo: {info['combo']}")
            total_reward += reward
            # env.render()
            
        ep_rewards.append(total_reward)
    print("隨機策略平均分數：", np.mean(ep_rewards))
    print(f"總共 Game Over 次數：{game_over_count}")
