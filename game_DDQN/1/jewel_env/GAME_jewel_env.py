
# 用於給AI訓練的遊戲架構



import numpy as np
import random
import gymnasium as gym  # 不是 gym！
from gymnasium import spaces
import tkinter as tk
import math
import copy
import torch
ROWS, COLS = 9, 6
GEM_TYPES = 9
WALL = 7
EMPTY = 0
GAMEOVER_T = 26
wall_timer =15# 每 20 步產生一次牆
IDLE_UPLOAD_T = 13
# GAMEOVER_T = 50
# wall_timer =20
# IDLE_UPLOAD_T = 30

GEM_TYPE_IDS = [1, 2, 3, 4, 5, 6, 8]  # 寶石代碼（排除0:空, 7:牆）
# GEM_TYPE_IDS = [1, 2, 3]  # 寶石代碼（排除0:空, 7:牆）


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
        self.last_action_coord = None  # (row, col)
        self.action_space = spaces.Discrete(ROWS * (COLS - 1) + 1)  # 0~44: swap, 45: upload
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9, 9, 6), dtype=np.float32)

        self.wall_interval = wall_timer
        self.wall_timer = self.wall_interval
        self.upload_timer = IDLE_UPLOAD_T
        self.falling_walls = []
        self.wall_cleared_count = 0
        self.no_clear_steps = 0 
        self.flag_upload = False
        self.upload_no_cleared_step = 0
        self.reward_mode = reward_mode
        self.reward_sum_components = {}
        self.episode_score = 0
        
        # if self.reward_mode == 'simple':
        #     self.generate_initial_randomboard()
        # else:
        self.generate_initial_board()


    def get_state(self):
        """
        回傳足以完全還原遊戲進度的 state（deepcopy，避免參照問題）
        """
        # 必須包含所有會影響遊戲進度的變數
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
        })

    def reset_to(self, state):
        """
        根據 get_state 回存
        """
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

        
        # 回傳 observation
        return self._get_obs(), {}

    def simulate_step(self, action):
        """
        回傳(新state, reward, done, info)，但不影響原本環境
        """
        tmp_env = copy.deepcopy(self)
        obs, reward, terminated, truncated, info = tmp_env.step(action)
        done = terminated or truncated
        next_state = tmp_env.get_state()
        return next_state, reward, done, info
    
    
    def seed(self, seed=None):
        self.seed_value = seed
        random.seed(seed)
        np.random.seed(seed)

    def valid_actions(self):
        valid = []
        # 檢查每一個 swap 動作是否合法（左右兩格都不是牆才合法）
        for row in range(ROWS):
            for col in range(COLS-1):
                if self.board[row][col] != WALL and self.board[row][col+1] != WALL:
                    valid.append(row * (COLS-1) + col)
        # upload 永遠都是合法動作（你可以根據遊戲邏輯決定要不要放進來）
        valid.append(ROWS * (COLS-1))  # 這是 upload row 動作
        return valid






    def _get_obs(self):
        """
        輸出 obs shape: (9, 9, 6)，前 8 層是寶石與牆壁，第 9 層是動作位置。
        """
        gem_ids = [1, 2, 3, 4, 5, 6, 8, 7]  # 7種寶石 + 牆壁
        H, W = self.board.shape

        obs = np.zeros((len(gem_ids), H, W), dtype=np.float32)

        for i, g in enumerate(gem_ids):
            obs[i] = (self.board == g).astype(np.float32)

        obs_tensor = torch.tensor(obs, dtype=torch.float32)  # (8, 9, 6)

        # === 動作通道 ===
        action_layer = torch.zeros((1, H, W), dtype=torch.float32)
        if self.last_action_coord is not None:
            row, col = self.last_action_coord
            action_layer[0, row, col] = 1.0

        # === 拼接第 9 層 ===
        obs_tensor = torch.cat([obs_tensor, action_layer], dim=0)  # => (9, 9, 6)
        return obs_tensor.detach().cpu().numpy().astype(np.float32)






    def get_obs(self):
        return self._get_obs()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.top_block_timer = 0
        self.combo = 0
        self.total_cleared = 0
        self.no_clear_steps = 0         # 🔧 要加上
        self.upload_timer = IDLE_UPLOAD_T  # 🔧 這也建議重設
        self.wall_timer = self.wall_interval  # 🔧 這也建議重設
        self.falling_walls.clear()     # 🔧 清空牆壁
        self.wall_cleared_count = 0
        self.steps_in_episode = 0
        self.no_clear_steps = 0 
        self.same_action_count = 0
        # if self.reward_mode == 'simple':
        #     self.generate_initial_randomboard()
        # else:
        self.generate_initial_board()

        self.total_cleared_this_episode = 0
        self.flag_upload = False
        self.same_action_count = 0
        self.upload_no_cleared_step = 0
        self.reward_sum_components = {}
        self.episode_score = 0

        obs = self._get_obs()
        info = {"episode_start": True}
        return obs, info

    





    def step(self, action):
        self.steps_in_episode += 1
        cleared = 0
        done = False
        self.combo = 0
        row = 0
        col = 0
        if isinstance(action, (list, np.ndarray)):
            action = action[0]

        reward_components = {
            "same_action": 0,
            "cleared": 0,
            "combo": 0,
            "bonus": 0,
            "wall": 0,
            "penalty": 0,
            "gameover": 0,
            "duration_bonus": 0,
            "no_clear_penalty": 0,
            "first_combo": 0,
            "upload": 0
        }



        # 處理動作
        if action == ROWS * (COLS - 1):  # upload row
            self.last_action_coord = None
            self.upload_no_cleared_step += 1
            self.flag_upload = True

            if self.upload_no_cleared_step >= 4:
                reward_components["upload"] = -0.5  # 連續 upload 3 次懲罰

        else:
            row = action // (COLS - 1)
            col = action % (COLS - 1)
            if (row, col) == self.last_action_coord:
                self.same_action_count += 1
                # if self.same_action_count >= 3:
                #     reward_components["same_action"] = -1.0
                # else:
                reward_components["same_action"] = -min(0.05 * (self.same_action_count), 0.6)
            else:
                self.same_action_count = 0
            self.last_action_coord = (row, col)
            if self.board[row][col] != WALL and self.board[row][col + 1] != WALL:
                self.board[row][col], self.board[row][col + 1] = self.board[row][col + 1], self.board[row][col]


        



        if self.upload_timer <= 0 or action == (ROWS * (COLS - 1)):
            self.upload_row()
        else:
            self.upload_timer -= 1

        cleared = self.resolve()
        self.drop_walls()


        WALL_mode = False
        cleared_mode = False
        combo_mode = False
        bonus_mode =False
        no_clear_penalty_mode = False
        upload_mode = False
        gameover_mode = False


        if self.reward_mode != "simple":
            self.wall_timer -= 1
            if self.wall_timer <= 0:
                self.insert_wall()
                self.wall_timer = self.wall_interval
            self.drop_walls()

        upload = ROWS * (COLS - 1)

        if cleared > 0:
            self.no_clear_steps = 0
        else:
            self.no_clear_steps += 1
            if self.flag_upload:
                self.upload_no_cleared_step += 1
            else:
                self.upload_no_cleared_step = 0

        # ==== 這裡開始調整 reward 範圍 ====
        if self.reward_mode == "simple":
            # self.upload_timer = 0
            reward_components["cleared"] = 0.5 * cleared  # 1 次消除給 0.2~0.6
            # if cleared == 0:
                # reward_components["no_clear_penalty"] = -0.2  # 沒有消除小小懲罰
                    # 多步沒消除，懲罰
            if self.no_clear_steps > 20:
                reward_components["no_clear_penalty"] = -0.3
            # if self.no_clear_steps > 50:
            #     # reward_components["no_clear_step"] = -0.05* self.no_clear_steps
            #     done = True
            # else:
            #     done = False
            if self.top_block_timer >= GAMEOVER_T:
                reward_components["gameover"] = -1.0
                done = True
            else:
                done = False

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
            else:
                done = False




        elif self.reward_mode == "advanced":
            reward_components["cleared"] = 0.15 * cleared
            reward_components["combo"] = 0.05 * self.combo
            if cleared >= 5:
                reward_components["bonus"] = 0.2
            reward_components["wall"] = 0.1 * self.wall_cleared_count
            reward_components["penalty"] = -0.05 * (self.top_block_timer / GAMEOVER_T)


            if cleared == 0:
                reward_components["no_clear_penalty"] = -0.1


            if self.top_block_timer >= GAMEOVER_T:
                    reward_components["gameover"] = -1.0
                    done = True
            else:
                done = False


        # upload
        # if self.flag_upload:
        #     if self.upload_no_cleared_step > 5:
        #         reward_components["upload"] = -0.5
        #         self.flag_upload = False
        #         self.upload_no_cleared_step = 0
        #     elif cleared >= 3:
        #         reward_components["upload"] = 0.5
        #         self.upload_no_cleared_step = 0
        #         self.flag_upload = False
            

        # Gameover
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
            "no_clear_steps": self.no_clear_steps,
            "top_block_timer": self.top_block_timer,
            "wall_cleared_count": self.wall_cleared_count,
            "gameover": done,
            "steps_in_episode": self.steps_in_episode,
            "episode_cleared": self.total_cleared_this_episode,
            "reward_breakdown": reward_components
        }
        info.update(reward_components)
        info["reward_sum_components"] = copy.deepcopy(self.reward_sum_components)
        terminated = done
        truncated = False
        if terminated or truncated:
            info["episode"] = {
                "r": float(self.episode_score),  # 或 reward 累加值
                "l": int(self.steps_in_episode)
            }
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info


    def generate_initial_randomboard(self):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        for i in range(ROWS - 9, ROWS): 
            for j in range(COLS):
                while True:
                    gem = random.choice(GEM_TYPE_IDS)

                    # 檢查水平：左兩個
                    if j >= 2 and gem == self.board[i][j - 1] == self.board[i][j - 2]:
                        continue

                    # 檢查垂直：上兩個
                    if i >= 2 and gem == self.board[i - 1][j] == self.board[i - 2][j]:
                        continue

                    self.board[i][j] = gem
                    break



    def generate_initial_board(self):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        for i in range(ROWS - 3, ROWS):  # 只填最底下 3 行
            for j in range(COLS):
                while True:
                    gem = random.choice(GEM_TYPE_IDS)

                    # 檢查水平：左兩個
                    if j >= 2 and gem == self.board[i][j - 1] == self.board[i][j - 2]:
                        continue

                    # 檢查垂直：上兩個
                    if i >= 2 and gem == self.board[i - 1][j] == self.board[i - 2][j]:
                        continue

                    self.board[i][j] = gem
                    break



    def upload_row(self):
        possible = GEM_TYPE_IDS.copy()
        if np.any(self.board[0] != EMPTY):
            self.top_block_timer += 1
            return

        # 嘗試多次直到不會產生 auto-clear 為止
        for _ in range(20):  # 最多試 20 次，避免死迴圈
            new_row = [random.choice(possible) for _ in range(COLS)]

            # 模擬插入盤面
            simulated_board = np.copy(self.board[1:])  # 原本第1~8行
            simulated_board = np.vstack([simulated_board, np.array(new_row)])  # 加入新的一行，變成9行

            if not self.has_immediate_match(simulated_board):
                # 如果這一行插入不會造成自動消除，就正式插入
                self.board[:-1] = self.board[1:]
                self.board[-1] = new_row
                self.upload_timer = IDLE_UPLOAD_T
                return

        # 如果 20 次都無法避免，還是強制插入最後一個 new_row
        self.board[:-1] = self.board[1:]
        self.board[-1] = new_row
        self.upload_timer = IDLE_UPLOAD_T


    def has_immediate_match(self, board):
        for i in range(ROWS):
            for j in range(COLS):
                val = board[i][j]
                if val in (EMPTY, WALL):
                    continue

                # 檢查水平三連
                if j >= 2 and val == board[i][j - 1] == board[i][j - 2]:
                    return True
                if j >= 1 and j < COLS - 1 and val == board[i][j - 1] == board[i][j + 1]:
                    return True
                if j < COLS - 2 and val == board[i][j + 1] == board[i][j + 2]:
                    return True

                # 檢查垂直三連
                if i >= 2 and val == board[i - 1][j] == board[i - 2][j]:
                    return True
                if i >= 1 and i < ROWS - 1 and val == board[i - 1][j] == board[i + 1][j]:
                    return True
                if i < ROWS - 2 and val == board[i + 1][j] == board[i + 2][j]:
                    return True

        return False


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
