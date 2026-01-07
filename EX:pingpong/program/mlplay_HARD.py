import sys
import random
import os
import pickle
import uuid

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.side = ai_name
        print(f"[{ai_name}] 初始化 Ultimate Collector (高精度物理核心)...")
        
        self.round_buffer = []
        self.final_status = "GAME_ALIVE" 
        self.current_strategy = None 
        self.prev_blocker_x = None
        
        self.log_dir = os.path.join(os.path.dirname(__file__), "data_log_HARD")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.BALL_WIDTH = 10.101
        self.BALL_HEIGHT = 10.101
        self.BOARD_WIDTH = 200
        self.PLATFORM_WIDTH = 40
        self.PLATFORM_HEIGHT = 10 
        self.BLOCKER_WIDTH = 30
        self.BLOCKER_HEIGHT = 20

    def update(self, scene_info: dict, *args, **kwargs):
        self.final_status = scene_info["status"]

        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        if not scene_info["ball_served"]:
            self.prev_blocker_x = None
            return "SERVE_TO_RIGHT"

        ball_x, ball_y = scene_info["ball"]
        speed_x, speed_y = scene_info["ball_speed"]
        blocker_x, blocker_y = scene_info["blocker"]
        
        blocker_vx = 0
        if self.prev_blocker_x is not None:
            blocker_vx = blocker_x - self.prev_blocker_x
        self.prev_blocker_x = blocker_x

        if self.side == "1P":
            platform_x, platform_y = scene_info["platform_1P"]
            opponent_x = scene_info["platform_2P"][0]
            intercept_y = platform_y - self.BALL_HEIGHT 
            is_incoming = speed_y > 0
        else:
            platform_x, platform_y = scene_info["platform_2P"]
            opponent_x = scene_info["platform_1P"][0]
            intercept_y = platform_y + self.PLATFORM_HEIGHT
            is_incoming = speed_y < 0

        command = "NONE"

        # ==========================================
        # ★★★ 預測與決策 ★★★
        # ==========================================
        if is_incoming:
            blocker_center = blocker_x + self.BLOCKER_WIDTH / 2
            
            if 60 < blocker_center < 140:
                self.current_strategy = "AVOID"
            elif self.current_strategy is None or self.current_strategy == "AVOID":
                if random.random() < 0.3:
                    self.current_strategy = "SLICE"
                else:
                    self.current_strategy = "FEED"

            # 使用升級版物理模擬
            pred_x = self.calculate_landing_x(
                ball_x, ball_y, speed_x, speed_y, intercept_y,
                blocker_x, blocker_y, blocker_vx
            )
            
            final_target = pred_x 

            if self.current_strategy == "AVOID":
                if blocker_center < 100: final_target = pred_x + 15
                else: final_target = pred_x - 15
            elif self.current_strategy == "SLICE":
                if speed_x > 0: final_target = pred_x - 10 
                else: final_target = pred_x + 10 
            else: # FEED
                opponent_center = opponent_x + 20
                if opponent_center < 100: final_target = pred_x - 5 
                else: final_target = pred_x + 5 

            # 牆壁緩衝：如果預測點太靠近牆，稍微往內修正，確保板子能接到
            if final_target < 20: final_target = 20
            elif final_target > 180: final_target = 180
            
            platform_center = platform_x + 20
            if abs(platform_center - final_target) < 3:
                command = "NONE"
                action_code = 2
            elif platform_center < final_target:
                command = "MOVE_RIGHT"
                action_code = 1
            else:
                command = "MOVE_LEFT"
                action_code = 0

            feature = [ball_x, ball_y, speed_x, speed_y, blocker_x, blocker_y, blocker_vx, platform_x]
            self.round_buffer.append([feature, action_code])

        else:
            self.current_strategy = None
            # 回防中間 (100)
            platform_center = platform_x + 20
            if platform_center < 95: command = "MOVE_RIGHT"
            elif platform_center > 105: command = "MOVE_LEFT"
            else: command = "NONE"

        return command

    def reset(self):
        i_lost = False
        if self.side == "1P" and self.final_status == "GAME_2P_WIN": i_lost = True
        elif self.side == "2P" and self.final_status == "GAME_1P_WIN": i_lost = True

        if len(self.round_buffer) > 0 and not i_lost:
            unique_filename = f"data_fullenv_{uuid.uuid4()}.pickle"
            file_path = os.path.join(self.log_dir, unique_filename)
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(self.round_buffer, f)
            except Exception as e:
                print(f"❌ Save failed: {e}")
        elif i_lost:
            print(f"🗑️ 漏接 (Loss)，丟棄本局數據。")

        self.round_buffer = [] 
        self.current_strategy = None
        self.prev_blocker_x = None
        self.final_status = "GAME_ALIVE"

    # ==========================================
    # ★★★ 高精度物理模擬 (Micro-stepping V4.0) ★★★
    # ==========================================
    def calculate_landing_x(self, x, y, vx, vy, target_y, bx, by, bvx):
        if vy == 0: return x
        
        sim_x, sim_y = x, y
        sim_vx, sim_vy = vx, vy
        
        sim_bx = bx
        sim_bvx = bvx
        has_blocker = (by != 0)
        
        is_moving_down = (target_y > y)
        RIGHT_WALL_LIMIT = self.BOARD_WIDTH - self.BALL_WIDTH
        BLOCKER_RIGHT_LIMIT = self.BOARD_WIDTH - self.BLOCKER_WIDTH 
        
        # 微步設定：每一幀拆成 10 小步來算
        # 這能大幅減少「穿透」或「牆角誤判」的問題
        MICRO_STEPS = 5
        dt = 1.0 / MICRO_STEPS
        
        frame_count = 0
        while frame_count < 2000: # 限制最大幀數
            
            # --- 判斷是否到達目標 ---
            if is_moving_down:
                if sim_y + sim_vy >= target_y:
                    # 簡單線性插值算出最後一步的精確 X
                    remaining_dist = target_y - sim_y
                    if sim_vy != 0:
                        time_needed = remaining_dist / sim_vy
                        sim_x += sim_vx * time_needed
                    break
            else:
                if sim_y + sim_vy <= target_y:
                    remaining_dist = target_y - sim_y
                    if sim_vy != 0:
                        time_needed = remaining_dist / sim_vy
                        sim_x += sim_vx * time_needed
                    break

            # --- 開始微步模擬 (Micro-stepping) ---
            for _ in range(MICRO_STEPS):
                # 1. 移動一小步
                sim_x += sim_vx * dt
                sim_y += sim_vy * dt
                
                # 2. 牆壁反彈 (修正版：鏡像彈射)
                # 這樣不會讓球「黏」在牆上，而是正確彈回
                if sim_x <= 0:
                    sim_x = -sim_x # 鏡像：把多跑出去的距離彈回來
                    sim_vx = -sim_vx
                elif sim_x >= RIGHT_WALL_LIMIT:
                    # 鏡像：多跑的距離 (sim_x - Limit) 扣回來
                    sim_x = RIGHT_WALL_LIMIT - (sim_x - RIGHT_WALL_LIMIT)
                    sim_vx = -sim_vx
                
                # 3. 障礙物模擬
                if has_blocker:
                    # 障礙物也只移動一小步
                    sim_bx += sim_bvx * dt
                    
                    # 障礙物撞牆
                    if sim_bx <= 0:
                        sim_bx = -sim_bx 
                        sim_bvx *= -1
                    elif sim_bx >= BLOCKER_RIGHT_LIMIT:
                        sim_bx = BLOCKER_RIGHT_LIMIT - (sim_bx - BLOCKER_RIGHT_LIMIT)
                        sim_bvx *= -1
                    
                    # 碰撞偵測 (AABB)
                    if (sim_x < sim_bx + self.BLOCKER_WIDTH and 
                        sim_x + self.BALL_WIDTH > sim_bx and
                        sim_y < by + self.BLOCKER_HEIGHT and 
                        sim_y + self.BALL_HEIGHT > by):
                        
                        # 計算重疊量
                        overlap_left = (sim_x + self.BALL_WIDTH) - sim_bx
                        overlap_right = (sim_bx + self.BLOCKER_WIDTH) - sim_x
                        overlap_top = (sim_y + self.BALL_HEIGHT) - by
                        overlap_bottom = (by + self.BLOCKER_HEIGHT) - sim_y
                        
                        min_overlap_x = min(overlap_left, overlap_right)
                        min_overlap_y = min(overlap_top, overlap_bottom)
                        
                        # 碰撞反應
                        if min_overlap_x < min_overlap_y:
                            sim_vx = -sim_vx # 側面反彈
                            # 推球出障礙物 (微推一點點防止黏住)
                            if overlap_left < overlap_right:
                                 sim_x = sim_bx - self.BALL_WIDTH - 0.1
                            else:
                                 sim_x = sim_bx + self.BLOCKER_WIDTH + 0.1
                        else:
                            sim_vy = -sim_vy # 上下反彈
                            if overlap_top < overlap_bottom:
                                 sim_y = by - self.BALL_HEIGHT - 0.1
                            else:
                                 sim_y = by + self.BLOCKER_HEIGHT + 0.1

            frame_count += 1
            
        return sim_x + (self.BALL_WIDTH / 2.0)