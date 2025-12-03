import tkinter as tk
import numpy as np
from jewel_env.GAME_jewel_env_cursor import JewelCursorEnv  # 換成你新版的環境

CELL_SIZE = 40
ROWS, COLS = 9, 6

GEM_COLORS = {
    0: "gray20", 1: "violet", 2: "green", 3: "khaki",
    4: "blue", 5: "magenta", 6: "orange", 7: "dim gray", 8: "cyan"
}

class ManualViewer:
    def __init__(self):
        self.env = JewelCursorEnv()
        self.env.reward_mode = "simple"
        self.obs, _ = self.env.reset()
        self.cursor_row = self.env.cursor_row
        self.cursor_col = self.env.cursor_col

        self.total_score = 0
        self.step_count = 0
        self.last_cleared = 0
        self.gameover_count = 0
        self.wall_cleared_count = 0
        self.total_combo = 0
        self.combo = 0
        self.episode_combo = 0
        self.reward = 0
        self.reward_components = {}

        self.root = tk.Tk()
        self.root.title("Manual Play Viewer - Keyboard Control")
        self.canvas = tk.Canvas(
            self.root,
            width=COLS * CELL_SIZE + 180,
            height=ROWS * CELL_SIZE + 50,
            bg="black"
        )
        self.canvas.pack()

        # 綁定鍵盤事件
        self.root.bind("<KeyPress>", self.on_key_press)

        self.draw_board()
        self.root.mainloop()

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(ROWS):
            for j in range(COLS):
                val = self.env.board[i][j]
                color = GEM_COLORS.get(val, "white")
                x0, y0 = j * CELL_SIZE, i * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
                self.canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2,
                                        text=str(val), fill="black", font=("Arial", 10))

        # === 游標紅框 ===
        x0 = self.cursor_col * CELL_SIZE
        y0 = self.cursor_row * CELL_SIZE
        x1 = x0 + CELL_SIZE
        y1 = y0 + CELL_SIZE
        self.canvas.create_rectangle(x0 + 2, y0 + 2, x1 - 2, y1 - 2,
                                     outline="red", width=3)

        # === 右側資訊列 ===
        offset = COLS * CELL_SIZE + 10
        y = 10
        spacing = 25

        def label(txt):
            nonlocal y
            self.canvas.create_text(offset, y, anchor="nw", text=txt, font=("Arial", 12), fill="white")
            y += spacing

        label(f"Step: {self.step_count}")
        label(f"Reward: {self.reward:.2f}")
        label(f"Combo: {self.combo}")
        label(f"Total Combo: {self.total_combo}")
        label(f"Score: {self.total_score:.2f}")
        label(f"Cleared: {self.last_cleared}")
        label(f"Game Over: {self.gameover_count}")
        label(f"Wall Cleared: {self.wall_cleared_count}")

        if hasattr(self, "reward_components"):
            label("Reward Breakdown:")
            for k, v in self.reward_components.items():
                label(f"  {k}: {v:+.3f}")

    def on_key_press(self, event):
        key = event.keysym.lower()

        action_map = {
            "up": 1,
            "down": 2,
            "left": 3,
            "right": 4,
            "return": 5,  # Enter
            "u": 6
        }

        if key not in action_map:
            return

        action = action_map[key]
        self.step(action)
        self.draw_board()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.reward = reward
        self.total_score += reward
        self.last_cleared = info['cleared']
        self.combo = info['combo']
        self.total_combo += info['combo']
        self.episode_combo += info['combo']
        self.wall_cleared_count = info['wall_cleared_count']
        self.reward_components = info.get("reward_breakdown", {})
        self.step_count += 1

        self.cursor_row = self.env.cursor_row
        self.cursor_col = self.env.cursor_col

        # ✅ 顯示每一層 OBS
        print("🔍 --- OBS 分層顯示 (C=9, H=9, W=6) ---")
        for i in range(self.obs.shape[0]):
            print(f"【Layer {i}】")
            print(self.obs[i])
            print()

        if terminated or truncated:
            self.gameover_count += 1
            self.env.reset()
            self.step_count = 0
            self.total_score = 0
            self.total_combo = 0
            self.cursor_row = self.env.cursor_row
            self.cursor_col = self.env.cursor_col


if __name__ == "__main__":
    ManualViewer()
