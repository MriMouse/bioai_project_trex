from random import randrange as rnd
from itertools import cycle
from random import choice, seed
from PIL import Image
import pygame
import time

# 固定随机种子常量 42
# DEFAULT_SEED = 1


# Game主类，支持AI和人类两种模式
class TRexGame:
    def __init__(self, human_mode=False, use_fixed_seed=True, random_seed=41):
        pygame.init()
        pygame.font.init()  # Initialize font module
        self.font = pygame.font.Font(None, 36)  # Default font, size 36
        self.small_font = pygame.font.Font(None, 28)
        self.speed = 4  # Initial game speed
        self.human_mode = human_mode
        self.score = 0
        self.use_fixed_seed = use_fixed_seed
        self.random_seed = random_seed
        # 设置随机种子
        if self.use_fixed_seed:
            seed(self.random_seed)
        self._init_resources()  # Initializes self.ground_img among others
        self.reset()

    def _init_resources(self):
        # Player sprites
        self.player_init = Image.open("resources.png").crop((77, 5, 163, 96)).convert("RGBA")
        self.player_init = self.player_init.resize(list(map(lambda x: x // 2, self.player_init.size)))

        self.player_frame_1 = Image.open("resources.png").crop((1679, 2, 1765, 95)).convert("RGBA")  # Standing/jumping
        self.player_frame_1 = self.player_frame_1.resize(list(map(lambda x: x // 2, self.player_frame_1.size)))

        self.player_frame_2 = Image.open("resources.png").crop((1767, 2, 1853, 95)).convert("RGBA")
        self.player_frame_2 = self.player_frame_2.resize(list(map(lambda x: x // 2, self.player_frame_2.size)))

        self.player_frame_3 = Image.open("resources.png").crop((1855, 2, 1941, 95)).convert("RGBA")  # Running
        self.player_frame_3 = self.player_frame_3.resize(list(map(lambda x: x // 2, self.player_frame_3.size)))

        self.player_frame_31 = Image.open("resources.png").crop((1943, 2, 2029, 95)).convert("RGBA")  # Running
        self.player_frame_31 = self.player_frame_31.resize(list(map(lambda x: x // 2, self.player_frame_31.size)))

        self.player_frame_4 = Image.open("resources.png").crop((2030, 2, 2117, 95)).convert("RGBA")  # Crashed
        self.player_frame_4 = self.player_frame_4.resize(list(map(lambda x: x // 2, self.player_frame_4.size)))

        self.player_frame_5 = Image.open("resources.png").crop((2207, 2, 2323, 95)).convert("RGBA")  # Crouching
        self.player_frame_5 = self.player_frame_5.resize(list(map(lambda x: x // 2, self.player_frame_5.size)))

        self.player_frame_6 = Image.open("resources.png").crop((2324, 2, 2441, 95)).convert("RGBA")  # Crouching
        self.player_frame_6 = self.player_frame_6.resize(list(map(lambda x: x // 2, self.player_frame_6.size)))

        # Environment
        self.cloud_img = Image.open("resources.png").crop((166, 2, 257, 29)).convert("RGBA")
        self.cloud_img = self.cloud_img.resize(list(map(lambda x: x // 2, self.cloud_img.size)))

        self.ground_img = Image.open("resources.png").crop((2, 102, 2401, 127)).convert("RGBA")
        self.ground_img = self.ground_img.resize(list(map(lambda x: x // 2, self.ground_img.size)))

        # Obstacles - Cacti
        self.obstacle1_img = Image.open("resources.png").crop((446, 2, 479, 71)).convert("RGBA")
        self.obstacle1_img = self.obstacle1_img.resize(list(map(lambda x: x // 2, self.obstacle1_img.size)))

        self.obstacle2_img = Image.open("resources.png").crop((446, 2, 547, 71)).convert("RGBA")
        self.obstacle2_img = self.obstacle2_img.resize(list(map(lambda x: x // 2, self.obstacle2_img.size)))

        self.obstacle3_img = Image.open("resources.png").crop((446, 2, 581, 71)).convert("RGBA")
        self.obstacle3_img = self.obstacle3_img.resize(list(map(lambda x: x // 2, self.obstacle3_img.size)))

        self.obstacle4_img = Image.open("resources.png").crop((653, 2, 701, 101)).convert("RGBA")  # Large cactus
        self.obstacle4_img = self.obstacle4_img.resize(list(map(lambda x: x // 2, self.obstacle4_img.size)))

        self.obstacle5_img = Image.open("resources.png").crop((653, 2, 749, 101)).convert("RGBA")  # Large cactus
        self.obstacle5_img = self.obstacle5_img.resize(list(map(lambda x: x // 2, self.obstacle5_img.size)))

        self.obstacle6_img = Image.open("resources.png").crop((851, 2, 950, 101)).convert("RGBA")  # Large cactus
        self.obstacle6_img = self.obstacle6_img.resize(list(map(lambda x: x // 2, self.obstacle6_img.size)))

        self.cactus_obstacles = [
            self.obstacle1_img,
            self.obstacle2_img,
            self.obstacle3_img,
            self.obstacle4_img,
            self.obstacle5_img,
            self.obstacle6_img,
        ]
        self.large_cactus_group = [self.obstacle4_img, self.obstacle5_img, self.obstacle6_img]

        # Obstacles - Birds (Pterodactyls)
        self.pterodactyl_f1_img = Image.open("resources.png").crop((262, 2, 262 + 90, 2 + 78)).convert("RGBA")
        self.pterodactyl_f1_img = self.pterodactyl_f1_img.resize(
            list(map(lambda x: x // 2, self.pterodactyl_f1_img.size))
        )
        self.pterodactyl_f2_img = Image.open("resources.png").crop((352, 2, 352 + 90, 2 + 78)).convert("RGBA")
        self.pterodactyl_f2_img = self.pterodactyl_f2_img.resize(
            list(map(lambda x: x // 2, self.pterodactyl_f2_img.size))
        )
        self.pterodactyl_sprites = [self.pterodactyl_f1_img, self.pterodactyl_f2_img]
        self.bird_altitudes = [115 - 35, 115 - 15, 115 + 10]

        self.all_obstacle_types = self.cactus_obstacles + self.pterodactyl_sprites

        self.speed_identifier = lambda x: 2 if x >= 30 else 8 if x < 8 else 5

    def _convert_pil_to_pygame(self, pil_image):
        return pygame.image.fromstring(pil_image.tobytes(), pil_image.size, "RGBA")

    def reset(self):
        # 重置随机种子，确保每次重置后生成的地图相同
        seed(self.random_seed)

        # 鸟计数器，每3个鸟强制生成一个高飞鸟
        self.bird_counter = 0

        self.cust_speed = self.speed_identifier(self.speed)
        self.running_animation = cycle(
            [self.player_frame_3] * self.cust_speed + [self.player_frame_31] * self.cust_speed
        )
        self.crouch_animation = cycle([self.player_frame_5] * self.cust_speed + [self.player_frame_6] * self.cust_speed)
        self.bird_animation = cycle([self.pterodactyl_f1_img] * 15 + [self.pterodactyl_f2_img] * 15)

        self.gameDisplay = pygame.display.set_mode((600, 200))
        pygame.display.set_caption("T-Rex Runner")
        self.clock = pygame.time.Clock()

        self.current_player_sprite = self.player_frame_1
        self.crashed = False
        self.done = False
        self.display_game_over_screen = False

        self.lock_bg_scroll = False
        self.bg_x = 0
        self.bg1_x = self.ground_img.width

        # AI模式自动开始
        if not self.human_mode:
            self.start_game_action = True
        else:
            self.start_game_action = False

        self.player_y = 110
        self.is_jumping = False
        self.is_crouching = False
        self.fast_fall = False
        self.vertical_velocity = 0

        self.clouds = []
        for _ in range(4):
            self.clouds.append([rnd(0, 600), rnd(0, 100)])

        self.active_obstacles = []
        self._spawn_initial_obstacles()

        self.score = 0
        self.game_speed = self.speed

        # 返回初始状态
        return self.get_state()

    def _check_if_obstacle_is_passable(self, obs_img, y_pos, is_bird):
        return True
        """检查障碍物是否可以被玩家通过（跳过或蹲下），鸟考虑体积（hitbox）"""
        # 玩家基本参数
        player_x = 5  # 玩家固定x位置
        player_width = self.player_frame_1.width
        player_height_standing = self.player_frame_1.height
        player_height_crouching = self.player_frame_5.height
        player_y_standing = 110
        player_y_crouching = 110 + (player_height_standing - player_height_crouching)

        # 跳跃参数
        jump_height = 70  # 跳跃高度

        # 障碍物参数
        obstacle_width = obs_img.width
        obstacle_height = obs_img.height
        obstacle_x = 0  # 障碍物x相对玩家x的偏移，实际判定时会用
        obstacle_y = y_pos

        if not is_bird:
            # 仙人掌的情况
            # 如果障碍物高度小于玩家跳跃高度，则可通过
            if obstacle_height + (115 - y_pos) < jump_height:
                return True
        else:
            # 鸟的情况，考虑hitbox重叠
            # 鸟的hitbox
            bird_rect = pygame.Rect(player_x + obstacle_x, obstacle_y, obstacle_width, obstacle_height)
            # 恐龙站立hitbox
            trex_stand_rect = pygame.Rect(player_x, player_y_standing, player_width, player_height_standing)
            # 恐龙蹲下hitbox
            trex_crouch_rect = pygame.Rect(player_x, player_y_crouching, player_width, player_height_crouching)

            # 跳跃时恐龙的hitbox（假设跳到最高点）
            trex_jump_rect = pygame.Rect(
                player_x, player_y_standing - jump_height, player_width, player_height_standing
            )

            # 鸟在高处：蹲下能通过
            if not trex_crouch_rect.colliderect(bird_rect):
                return True
            # 鸟在低处：跳跃能通过
            if not trex_jump_rect.colliderect(bird_rect):
                return True
            # 鸟在中间：站立、蹲下、跳跃都碰撞，视为不可通过
            return False

        # 默认：生成一个可通过的小仙人掌（保底解决方案）
        return False

    def _spawn_obstacle(self, x_pos):
        """
        生成障碍物，保证每个障碍物都能通过：
        """
        PRINT_BIRD = False  # 是否打印鸟的生成信息
        # if self.score >= 128:
        #     import code

        #     code.interact(local=locals())
        max_attempts = 20
        jump_height = 90
        player_x = 5
        player_width = self.player_frame_1.width
        player_height = self.player_frame_1.height
        crouch_height = player_height // 2
        player_y = 110
        max_cactus_width = 60
        for _ in range(max_attempts):
            chosen_obj_img = choice(self.all_obstacle_types)
            y_pos = 130
            is_bird = chosen_obj_img in self.pterodactyl_sprites
            if is_bird:
                # 只允许低飞（跳跃必过）或高飞（下蹲必过），且鸟宽度强行缩小
                if choice([True, False]):
                    y_pos = self.bird_altitudes[2]  # 只选最低飞
                else:
                    y_pos = self.bird_altitudes[0]  # 只选最高飞
                self.bird_counter += 1
                obs_img = self.pterodactyl_f1_img.copy()
                # 鸟宽度强行缩小
                obs_img = obs_img.crop((0, 0, min(40, obs_img.width), obs_img.height))
                obs_rect = pygame.Rect(x_pos, y_pos, obs_img.width, obs_img.height)
                crouch_rect = pygame.Rect(player_x, player_y + player_height // 2, player_width, crouch_height)
                jump_rect = pygame.Rect(player_x, player_y - jump_height, player_width, player_height)
                # 鸟宽度过大也不生成
                max_jump_time = 1.5  # 秒
                max_jump_horiz_dist = int(self.speed * 1.5 * max_jump_time * 60)
                if obs_img.width > max_jump_horiz_dist:
                    continue
                if not crouch_rect.colliderect(obs_rect) or not jump_rect.colliderect(obs_rect):
                    ret = {"x": x_pos, "y": y_pos, "img": obs_img, "is_bird": True, "passed": False}
                    if PRINT_BIRD:
                        print(f"Spawned: {ret}")
                    return ret
            else:
                if chosen_obj_img in self.large_cactus_group:
                    y_pos = 115
                obs_img = chosen_obj_img
                # 仙人掌高度不能超过最大跳跃高度，宽度不能超过最大允许宽度
                if obs_img.height + (115 - y_pos) < jump_height + 10 and obs_img.width <= max_cactus_width:
                    ret = {"x": x_pos, "y": y_pos, "img": obs_img, "is_bird": False, "passed": False}
                    if PRINT_BIRD:
                        print(f"Spawned: {ret}")
                    return ret

        # 如果多次尝试都失败，则生成一个默认的小仙人掌（肯定可通过）
        default_obstacle = self.obstacle1_img

        ret = {"x": x_pos, "y": 130, "img": default_obstacle, "is_bird": False, "passed": False}
        if PRINT_BIRD:
            print(f"Spawned: {ret}")
        return ret

    def _spawn_initial_obstacles(self):
        self.active_obstacles = []
        self.active_obstacles.append(self._spawn_obstacle(rnd(600, 800)))
        self.active_obstacles.append(self._spawn_obstacle(self.active_obstacles[0]["x"] + rnd(300, 500)))
        self.active_obstacles.append(self._spawn_obstacle(self.active_obstacles[1]["x"] + rnd(300, 500)))

    def get_state(self):
        player_x_pos = 5  # 玩家x坐标
        # 默认障碍物结构
        default_obs_val = {
            "x": 999 + player_x_pos,
            "y": 130,
            "img": self.obstacle1_img,
            "is_bird": False,
            "passed": True,
        }
        obs1 = self.active_obstacles[0] if len(self.active_obstacles) > 0 else default_obs_val
        obs2 = self.active_obstacles[1] if len(self.active_obstacles) > 1 else default_obs_val
        return {
            "player_y": self.player_y,
            "player_vertical_velocity": self.vertical_velocity,
            "is_crouching": self.is_crouching,
            "obs1_dist_x": obs1["x"] - player_x_pos,
            "obs1_y": obs1["y"],
            "obs1_width": obs1["img"].width,
            "obs1_height": obs1["img"].height,
            "obs1_is_bird": obs1["is_bird"],
            "obs2_dist_x": obs2["x"] - player_x_pos,
            "obs2_y": obs2["y"],
            "obs2_width": obs2["img"].width,
            "obs2_height": obs2["img"].height,
            "obs2_is_bird": obs2["is_bird"],
            "game_speed": self.game_speed,
            "score": self.score,
            "done": self.done,
        }

    def step(self, action):

        if self.done:
            return self.get_state(), 0, self.done

        reward = 0.1
        if action == 1:
            if not self.is_jumping and self.player_y >= 110:
                self.is_jumping = True
                self.vertical_velocity = -12
                self.is_crouching = False
                self.fast_fall = False
        elif action == 2:
            if self.is_jumping:
                self.fast_fall = True
            else:
                self.is_crouching = True
        else:
            if self.is_crouching and not self.is_jumping:
                self.is_crouching = False

        self._tick_game_logic()

        if self.done:
            reward = -1
        else:
            for obs in self.active_obstacles:
                if obs["x"] < 5 - obs["img"].width and not obs["passed"]:
                    reward += 10
                    obs["passed"] = True

        return self.get_state(), reward, self.done

    def _handle_input_human(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.crashed = True
                self.done = True

            if not self.done:
                if event.type == pygame.KEYDOWN:
                    if not self.start_game_action:
                        self.start_game_action = True

                    if event.key == pygame.K_UP or event.key == pygame.K_SPACE:
                        if not self.is_jumping and self.player_y >= 110:
                            self.is_jumping = True
                            self.vertical_velocity = -12
                            self.is_crouching = False
                            self.fast_fall = False

                    elif event.key == pygame.K_DOWN:
                        if self.is_jumping:
                            self.fast_fall = True
                        else:
                            self.is_crouching = True

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_DOWN:
                        self.is_crouching = False
                        self.fast_fall = False

            elif self.done and self.display_game_over_screen:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.reset()

    def _update_player_state(self):
        if self.done:
            self.current_player_sprite = self.player_frame_4
        elif self.is_jumping or self.player_y < 110:
            self.current_player_sprite = self.player_frame_1
        elif self.is_crouching:
            self.current_player_sprite = next(self.crouch_animation)
        else:
            self.current_player_sprite = next(self.running_animation)

        if self.is_jumping or self.player_y < 110:
            self.player_y += self.vertical_velocity
            fall_speed = 0.8  # Changed from 0.6 to 0.9
            if self.fast_fall:
                fall_speed *= 3
            self.vertical_velocity += fall_speed

            if self.player_y >= 110:
                self.player_y = 110
                self.is_jumping = False
                self.vertical_velocity = 0
                self.fast_fall = False
                keys = pygame.key.get_pressed()
                if keys[pygame.K_DOWN]:  # If still holding down after landing from fast fall
                    self.is_crouching = True
                else:
                    self.is_crouching = False

    def _update_environment_and_obstacles(self):
        if not self.start_game_action or self.done:
            return

        for cloud_pos in self.clouds:
            cloud_pos[0] -= 1
            if cloud_pos[0] <= -self.cloud_img.width:
                cloud_pos[0] = 600 + rnd(0, 100)
                cloud_pos[1] = rnd(0, 100)

        # Move ground
        self.bg_x -= self.game_speed
        self.bg1_x -= self.game_speed

        # MODIFICATION 2: Correct ground scrolling logic
        # If a ground image has moved completely off-screen to the left,
        # shift it to appear to the right of the other ground image by effectively
        # adding 2 * self.ground_img.width to its current x position.
        if self.bg_x <= -self.ground_img.width:
            self.bg_x += 2 * self.ground_img.width
        if self.bg1_x <= -self.ground_img.width:
            self.bg1_x += 2 * self.ground_img.width

        new_obstacles = []
        max_x = 0
        for obs in self.active_obstacles:
            obs["x"] -= self.game_speed
            if obs["x"] > -obs["img"].width:
                new_obstacles.append(obs)
                if obs["x"] > max_x:
                    max_x = obs["x"]
            else:
                self.score += 10  # This score increment should perhaps be based on player passing it, not it going off-screen for AI.
                # For human mode this is one way to do it. Already handled for AI in step().

            if obs["x"] < 0:
                obs["passed"] = True
        self.active_obstacles = new_obstacles

        while len(self.active_obstacles) < 3:
            min_dist = int(self.game_speed * 50)
            max_dist = int(self.game_speed * 70)
            if max_dist <= min_dist:  # Ensure randrange has a valid range
                max_dist = min_dist + 100  # Give a default larger range if speed is too low
            spawn_x = max(600, int(max_x) + rnd(min_dist, max_dist))
            self.active_obstacles.append(self._spawn_obstacle(spawn_x))

        self.score += 0.1
        self.game_speed += 0.001

    def _check_collisions(self):
        # 角色参数
        player_x = 5
        player_width = self.player_frame_1.width
        player_height = self.player_frame_1.height
        # 下蹲时碰撞箱高度减半，底边不变
        if self.is_crouching and not self.is_jumping:
            hitbox_height = player_height // 2
            hitbox_y = self.player_y + player_height // 2
        else:
            hitbox_height = player_height
            hitbox_y = self.player_y
        player_rect = pygame.Rect(player_x, int(hitbox_y), player_width, int(hitbox_height))

        for obs in self.active_obstacles:
            obs_rect = pygame.Rect(int(obs["x"]), int(obs["y"]), obs["img"].width, obs["img"].height)
            if player_rect.colliderect(obs_rect):
                self.done = True
                # self.crashed = True
                break

    def _render(self):
        self.gameDisplay.fill((255, 255, 255))

        for cloud_pos in self.clouds:
            self.gameDisplay.blit(self._convert_pil_to_pygame(self.cloud_img), cloud_pos)

        self.gameDisplay.blit(self._convert_pil_to_pygame(self.ground_img), (int(self.bg_x), 150))
        self.gameDisplay.blit(self._convert_pil_to_pygame(self.ground_img), (int(self.bg1_x), 150))

        self.gameDisplay.blit(self._convert_pil_to_pygame(self.current_player_sprite), (5, self.player_y))

        current_bird_anim_frame = self._convert_pil_to_pygame(next(self.bird_animation))
        for obs in self.active_obstacles:
            img_to_draw_pil = obs["img"]
            if obs["is_bird"]:
                img_to_draw_pygame = current_bird_anim_frame
            else:
                img_to_draw_pygame = self._convert_pil_to_pygame(img_to_draw_pil)
            self.gameDisplay.blit(img_to_draw_pygame, (obs["x"], obs["y"]))

        score_text = self.font.render(f"Score: {int(self.score)}", True, (0, 0, 0))
        self.gameDisplay.blit(score_text, (450, 10))

        if self.active_obstacles:
            obs1 = self.active_obstacles[0]
            player_x_pos = 5  # Player's fixed x position
            obs1_dist_x = obs1["x"] - player_x_pos
            obs1_height = obs1["img"].height
            obs1_type_char = "Bird" if obs1["is_bird"] else "Cactus"

            obs1_info_str = f"Obs1: D:{int(obs1_dist_x)} H:{int(obs1_height)} {obs1_type_char} Passed: {obs1["passed"]}"
            obs1_info_surface = self.small_font.render(obs1_info_str, True, (0, 0, 0))

            # Position below the score
            score_text_height = score_text.get_height()
            obs1_info_pos_y = 10 + score_text_height + 2  # 2px padding
            self.gameDisplay.blit(obs1_info_surface, (250, obs1_info_pos_y))

        if self.done and self.display_game_over_screen:
            game_over_text = self.font.render("Game Over!", True, (200, 0, 0))
            restart_text = self.small_font.render("Press R to Restart", True, (0, 0, 0))

            text_rect = game_over_text.get_rect(center=(600 // 2, 200 // 2 - 20))
            self.gameDisplay.blit(game_over_text, text_rect)

            restart_rect = restart_text.get_rect(center=(600 // 2, 200 // 2 + 20))
            self.gameDisplay.blit(restart_text, restart_rect)

        pygame.display.update()

    def _tick_game_logic(self):
        self._update_player_state()
        if not self.done:
            self._update_environment_and_obstacles()
            self._check_collisions()

        if self.done and not self.display_game_over_screen:
            self.display_game_over_screen = True

    def human_play_tick(self):
        self._handle_input_human()
        self._tick_game_logic()
        self._render()
        self.clock.tick(60)

    def is_over(self):
        return self.done

    def get_score(self):
        return int(self.score)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    game = TRexGame(human_mode=True)
    while not game.crashed:
        game.human_play_tick()

        # This part of the loop body for 'done' game is implicitly handled
        # by human_play_tick now (input for restart, logic stopping)
        # if game.done and game.display_game_over_screen:
        #     pass

    final_score = game.get_score()
    print(f"Game Ended! Final Score: {final_score}")
    game.close()
    # pygame.quit() is called in game.close().
    # The second quit() is for exiting the Python script.
    quit()
