import tkinter as tk
import numpy as np
from jewel_env.GAME_jewel_env_blacklist import JewelEnv

CELL_SIZE = 40
ROWS, COLS = 9, 6
UPLOAD_ACTION = ROWS * (COLS - 1)

GEM_COLORS = {
    0: "gray20", 1: "violet", 2: "green", 3: "khaki",
    4: "blue", 5: "magenta", 6: "orange", 7: "dim gray", 8: "cyan"
}

class ManualViewer:
    def __init__(self):
        self.env = JewelEnv()
        self.obs = self.env.reset()
        self.selected = None  # (row, col)

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
        self.root.title("Manual Play Viewer")
        self.canvas = tk.Canvas(
            self.root,
            width=COLS * CELL_SIZE + 180,
            height=ROWS * CELL_SIZE + 50,
            bg="black"
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

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

        if self.selected:
            i, j = self.selected
            x0, y0 = j * CELL_SIZE, i * CELL_SIZE
            x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE
            self.canvas.create_rectangle(x0 + 2, y0 + 2, x1 - 2, y1 - 2,
                                         outline="red", width=3)

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
        label(f"wall_cleared_count: {self.wall_cleared_count}")


        if hasattr(self, "reward_components"):
            label("Reward Breakdown:")
            for k, v in self.reward_components.items():
                label(f"  {k}: {v:+.3f}")



        upload_width = 120
        upload_height = 30
        upload_x = (COLS * CELL_SIZE - upload_width) // 2  # 水平置中
        upload_y = ROWS * CELL_SIZE + 10                   # 棋盤正下方偏移一點

        self.canvas.create_rectangle(
            upload_x, upload_y,
            upload_x + upload_width, upload_y + upload_height,
            fill="gray60", tags="upload"
        )
        self.canvas.create_text(
            upload_x + upload_width // 2, upload_y + upload_height // 2,
            text="Upload ⬆️", font=("Arial", 12), tags="upload"
        )
        self.canvas.tag_bind("upload", "<Button-1>", self.upload)

    def on_click(self, event):
        row, col = event.y // CELL_SIZE, event.x // CELL_SIZE

        # 避免 upload 按鈕重複觸發
        upload_x = COLS * CELL_SIZE + 10
        upload_y = 10 + 25 * 4  # 第四行之後的 y 位置
        if upload_x <= event.x <= upload_x + 120 and upload_y <= event.y <= upload_y + 30:
            return  # upload 由 tag_bind 控制，這裡跳過

        if not (0 <= row < ROWS and 0 <= col < COLS - 1):
            return

        self.step(row * (COLS - 1) + col)
        self.selected = None
        self.draw_board()


    def step(self, action):
        self.env.reward_mode = "advanced"

        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.obs = obs
        self.reward = reward

        # === ⭐ 在 CMD 印出 OBS（你要的） ⭐===
        print("\n=== OBS (8×9×6) ===")
        # print(self.obs)
        # print("shape:", self.obs.shape)
        for layer_idx in range(self.obs.shape[0]):
            print(f"\n---- 第 {layer_idx+1} 層 ----")
            print(self.obs[layer_idx])
        print("======================================\n")
        self.total_score += reward
        self.last_cleared = info['cleared']
        self.combo = info['combo']
        self.total_combo += info['combo']
        self.episode_combo += info['combo']
        self.wall_cleared_count = info['wall_cleared_count']
        self.reward_components = info.get("reward_breakdown", {})
        self.step_count += 1

        if done:
            self.gameover_count += 1
            self.env.reset()


    def upload(self, event):
        self.step(UPLOAD_ACTION)
        self.draw_board()

if __name__ == "__main__":
    ManualViewer()
