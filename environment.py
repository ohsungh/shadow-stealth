import pygame
import math
import random
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces

# --- 환경 설정 (game.py와 동기화) ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
CELL_SIZE = 40
GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

PLAYER_SPEED = 4.0
ENEMY_SPEED = 1.8
LIGHT_EXPOSURE_LIMIT = 0.7  # 초

# --- LIDAR 및 관측 공간 설정 ---
LIDAR_RAYS = 36
LIDAR_MAX_DIST = 1000
# 관측 타입 ID
TYPE_ID = {
    "NOTHING": 0,
    "WALL": 1,
    "ENEMY": 2,
    "EXIT": 3,
    "PUDDLE": 4,
    "LIGHT": 5,
}

# --- 유틸리티 함수 (game.py에서 가져옴) ---
def get_wall_vertices(walls):
    points = set()
    for wall in walls:
        points.add(wall.topleft); points.add(wall.topright)
        points.add(wall.bottomleft); points.add(wall.bottomright)
    return list(points)

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def is_solvable(walls, player_pos, exit_rect):
    start_node = (int(player_pos[0] // CELL_SIZE), int(player_pos[1] // CELL_SIZE))
    end_node = (int(exit_rect.centerx // CELL_SIZE), int(exit_rect.centery // CELL_SIZE))
    grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for wall in walls:
        for x in range(wall.left // CELL_SIZE, wall.right // CELL_SIZE + 1):
            for y in range(wall.top // CELL_SIZE, wall.bottom // CELL_SIZE + 1):
                if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT: grid[y][x] = 1
    if not (0 <= start_node[1] < GRID_HEIGHT and 0 <= start_node[0] < GRID_WIDTH and 0 <= end_node[1] < GRID_HEIGHT and 0 <= end_node[0] < GRID_WIDTH): return False
    if grid[start_node[1]][start_node[0]] == 1 or grid[end_node[1]][end_node[0]] == 1: return False
    queue = deque([start_node]); visited = {start_node}
    while queue:
        x, y = queue.popleft()
        if (x, y) == end_node: return True
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and grid[ny][nx] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny)); queue.append((nx, ny))
    return False

def generate_level(difficulty):
    walls = []
    player_start = (80, SCREEN_HEIGHT / 2)
    exit_rect = pygame.Rect(SCREEN_WIDTH - 120, SCREEN_HEIGHT / 2 - 30, 60, 60)
    max_walls = 10 + difficulty * 2
    
    # 맵 생성 시도
    for _ in range(50): # 최대 50번 시도
        walls = []
        for _ in range(max_walls * 3):
            if len(walls) >= max_walls: break
            w = random.randint(CELL_SIZE * 2, CELL_SIZE * 6)
            h = random.randint(CELL_SIZE * 2, CELL_SIZE * 6)
            x = random.randint(0, SCREEN_WIDTH - w)
            y = random.randint(0, SCREEN_HEIGHT - h)
            new_wall = pygame.Rect(x, y, w, h)
            player_area = pygame.Rect(*player_start, 1, 1).inflate(CELL_SIZE*2, CELL_SIZE*2)
            if new_wall.colliderect(player_area) or new_wall.colliderect(exit_rect.inflate(CELL_SIZE*2, CELL_SIZE*2)): continue
            walls.append(new_wall)
        
        if is_solvable(walls, player_start, exit_rect):
            break # 해결 가능한 맵이 생성되면 중단
    else: # 50번 시도 후에도 실패하면 그냥 마지막 생성된 맵 사용
        print("Warning: Failed to generate a provably solvable map.")

    open_spaces = []
    for i in range(200):
        x, y = random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)
        is_on_wall = any(wall.collidepoint(x, y) for wall in walls)
        is_on_exit = exit_rect.collidepoint(x,y)
        if not is_on_wall and not is_on_exit:
             open_spaces.append((x, y))
    if not open_spaces: open_spaces.append((SCREEN_WIDTH/3, SCREEN_HEIGHT/3))

    num_towers = min(len(open_spaces), 1 + difficulty)
    towers = [{"pos": open_spaces.pop(random.randrange(len(open_spaces))), "angle": random.randint(0, 360), "speed": random.uniform(0.5, 1.0) * random.choice([-1, 1]), "cone_angle": random.randint(25, 40), "radius": random.randint(300, 450)} for _ in range(num_towers)]
    num_puddles = min(len(open_spaces), 5 + difficulty * 2)
    puddles = [pygame.Rect(*open_spaces.pop(random.randrange(len(open_spaces))), CELL_SIZE, CELL_SIZE) for _ in range(num_puddles)]
    num_enemies = min(len(open_spaces), difficulty)
    enemies = [{"pos": open_spaces.pop(random.randrange(len(open_spaces)))} for _ in range(num_enemies)]
    
    return {"player_start": player_start, "exit": exit_rect, "walls": walls, "puddles": puddles, "towers": towers, "enemies": enemies}

# --- 강화학습 환경 클래스 (Gymnasium 표준 준수) ---
class ShadowStealthEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super().__init__()
        pygame.init()
        
        self.difficulty = 0
        self.screen = None
        self.render_mode = render_mode

        # 행동 공간: 8방향 이동 + 정지 (총 9개)
        self.action_space = spaces.Discrete(9) 
        
        # 관측 공간: LIDAR (거리, 타입) * 광선 수 + 플레이어 상태 (빛 노출, 은신 여부)
        obs_shape = (LIDAR_RAYS * 2 + 2,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """새로운 에피소드를 위해 환경을 초기화합니다."""
        super().reset(seed=seed) # Gymnasium API 준수

        level_data = generate_level(self.difficulty)
        self.player_pos = pygame.Vector2(level_data["player_start"])
        self.player_radius = 12
        self.exit_rect = level_data["exit"]
        self.walls = level_data["walls"]
        self.puddles = level_data["puddles"]
        self.towers = level_data["towers"]
        self.enemies = [{"pos": pygame.Vector2(e["pos"]), "radius": 15} for e in level_data["enemies"]]
        self.wall_vertices = get_wall_vertices(self.walls)
        
        self.light_exposure_time = 0
        self.light_polygons = []
        self.steps = 0
        
        observation = self._get_observation()
        info = {} # info 딕셔너리
        
        return observation, info

    def step(self, action):
        """행동을 수행하고 (관측, 보상, 종료, 시간초과, 정보)를 반환합니다."""
        self.steps += 1
        
        self._move_player(action)
        self._update_game_state()
        
        reward, terminated = self._calculate_reward()
        observation = self._get_observation()
        
        truncated = False
        if self.steps > 2500: # 너무 오래 걸리면 에피소드 종료 (시간초과)
            truncated = True
        
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        """환경을 시각적으로 렌더링합니다."""
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        if self.screen is None and self.render_mode == 'human':
            pygame.display.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Shadow Stealth - AI Environment")
            self.font = pygame.font.Font(None, 36)

        canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        canvas.fill((0,0,0))

        # 조명
        for poly in self.light_polygons:
            # pygame.draw.polygon(canvas, (255, 255, 0, 50), poly) # 투명도 있는 색상은 surface에 직접 그릴 때 다르게 동작
            s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.polygon(s, (255, 255, 0, 30), poly)
            canvas.blit(s, (0,0))
        # 벽
        for wall in self.walls:
            pygame.draw.rect(canvas, (204, 34, 34), wall)
        # 물웅덩이
        for puddle in self.puddles:
            puddle_surf = pygame.Surface(puddle.size, pygame.SRCALPHA)
            puddle_surf.fill((74, 90, 121, 100))
            canvas.blit(puddle_surf, puddle.topleft)
        # 출구
        pygame.draw.rect(canvas, (162, 255, 0), self.exit_rect, border_radius=10)
        # 감시탑
        for tower in self.towers:
            pygame.draw.circle(canvas, (255, 255, 255), tower['pos'], 10)
        # 적
        for enemy in self.enemies:
            pygame.draw.circle(canvas, (255, 0, 85), enemy['pos'], enemy['radius'])
        
        # 플레이어
        pygame.draw.circle(canvas, (0, 255, 221), self.player_pos, self.player_radius)

        # LIDAR 광선 그리기 (디버깅용)
        obs = self._get_observation()
        lidar_data = obs[:LIDAR_RAYS * 2].reshape((LIDAR_RAYS, 2))
        for i in range(LIDAR_RAYS):
            angle = (2 * math.pi / LIDAR_RAYS) * i
            dist = lidar_data[i, 0] * LIDAR_MAX_DIST
            obj_type = lidar_data[i, 1]
            
            color = (100, 100, 100)
            if obj_type == TYPE_ID["WALL"]: color = (255, 0, 0)
            elif obj_type == TYPE_ID["ENEMY"]: color = (255, 100, 0)
            elif obj_type == TYPE_ID["EXIT"]: color = (0, 255, 0)
            
            start_pos = self.player_pos
            end_pos = start_pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * dist
            pygame.draw.line(canvas, color, start_pos, end_pos, 1)

        if self.render_mode == "human":
            self.screen.blit(canvas, (0,0))
            pygame.event.pump()
            pygame.display.flip()
        
        if self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def _move_player(self, action):
        """주어진 action에 따라 플레이어를 이동시킵니다."""
        direction = pygame.Vector2(0, 0)
        if action == 0: direction.y = -1
        elif action == 1: direction.y = -1; direction.x = 1
        elif action == 2: direction.x = 1
        elif action == 3: direction.y = 1; direction.x = 1
        elif action == 4: direction.y = 1
        elif action == 5: direction.y = 1; direction.x = -1
        elif action == 6: direction.x = -1
        elif action == 7: direction.y = -1; direction.x = -1

        if direction.length() != 0:
            direction.normalize_ip()
        
        # X축 이동 및 충돌
        self.player_pos.x += direction.x * PLAYER_SPEED
        player_rect = pygame.Rect(0, 0, self.player_radius * 2, self.player_radius * 2)
        player_rect.center = self.player_pos
        for wall in self.walls:
            if wall.colliderect(player_rect):
                if direction.x > 0: player_rect.right = wall.left
                if direction.x < 0: player_rect.left = wall.right
                self.player_pos.x = player_rect.centerx

        # Y축 이동 및 충돌
        self.player_pos.y += direction.y * PLAYER_SPEED
        player_rect.center = self.player_pos
        for wall in self.walls:
            if wall.colliderect(player_rect):
                if direction.y > 0: player_rect.bottom = wall.top
                if direction.y < 0: player_rect.top = wall.bottom
                self.player_pos.y = player_rect.centery

        self.player_pos.x = np.clip(self.player_pos.x, self.player_radius, SCREEN_WIDTH - self.player_radius)
        self.player_pos.y = np.clip(self.player_pos.y, self.player_radius, SCREEN_HEIGHT - self.player_radius)

    def _update_game_state(self):
        """적, 조명 등 게임의 동적 요소들을 업데이트합니다."""
        is_in_puddle = any(puddle.collidepoint(self.player_pos) for puddle in self.puddles)
        
        for tower in self.towers:
            tower['angle'] = (tower['angle'] + tower['speed']) % 360
        self.light_polygons = self._calculate_light_polygons()
        
        for enemy in self.enemies:
            direction = self.player_pos - enemy["pos"]
            if 0 < direction.length() < 300 and not is_in_puddle:
                can_see = not any(pygame.Rect(wall).clipline(enemy["pos"], self.player_pos) for wall in self.walls)
                if can_see:
                    direction.normalize_ip()
                    enemy["pos"] += direction * ENEMY_SPEED

    def _calculate_reward(self):
        """현재 게임 상태를 기반으로 보상과 종료 여부를 결정합니다."""
        terminated = False
        reward = -0.1 # 생존 패널티

        is_in_puddle = any(puddle.collidepoint(self.player_pos) for puddle in self.puddles)
        is_in_light = any(point_in_polygon(self.player_pos, poly) for poly in self.light_polygons)

        if is_in_light and not is_in_puddle:
            self.light_exposure_time += 1 / self.metadata['render_fps']
            reward -= 5.0
            if self.light_exposure_time > LIGHT_EXPOSURE_LIMIT:
                terminated = True
                reward -= 200
        else:
            self.light_exposure_time = max(0, self.light_exposure_time - 1 / self.metadata['render_fps'])

        if is_in_puddle:
            reward += 1.0

        for enemy in self.enemies:
            if self.player_pos.distance_to(enemy["pos"]) < self.player_radius + enemy["radius"]:
                terminated = True
                reward -= 200
                break
        
        if self.exit_rect.collidepoint(self.player_pos):
            terminated = True
            reward += 500
            self.difficulty += 1

        return reward, terminated

    def _get_observation(self):
        """LIDAR 스캔을 포함한 현재 관측 상태를 반환합니다."""
        lidar_data = self._scan_lidar()
        
        is_in_puddle = 1.0 if any(puddle.collidepoint(self.player_pos) for puddle in self.puddles) else 0.0
        light_exposure_ratio = np.clip(self.light_exposure_time / LIGHT_EXPOSURE_LIMIT, 0, 1)
        
        observation = np.concatenate((lidar_data.flatten(), [is_in_puddle, light_exposure_ratio]))
        return observation.astype(np.float32)

    def _scan_lidar(self):
        """플레이어 위치에서 LIDAR 광선을 쏴서 주변 환경을 스캔합니다."""
        obs = np.zeros((LIDAR_RAYS, 2), dtype=np.float32)

        for i in range(LIDAR_RAYS):
            angle = (2 * math.pi / LIDAR_RAYS) * i
            ray_dir = pygame.Vector2(math.cos(angle), math.sin(angle))
            ray_end = self.player_pos + ray_dir * LIDAR_MAX_DIST
            
            min_dist_sq = LIDAR_MAX_DIST**2
            closest_type = float(TYPE_ID["NOTHING"])

            for wall in self.walls:
                intersect = wall.clipline(self.player_pos, ray_end)
                if intersect:
                    dist_sq = self.player_pos.distance_squared_to(intersect[0])
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        closest_type = float(TYPE_ID["WALL"])
            
            for enemy in self.enemies:
                p, r = enemy["pos"], enemy["radius"]
                oc = p - self.player_pos
                l = ray_dir.dot(oc)
                if l > 0:
                    d2 = oc.dot(oc) - l*l
                    if d2 < r*r:
                        dist_sq = (l - math.sqrt(r*r - d2))**2
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            closest_type = float(TYPE_ID["ENEMY"])

            intersect = self.exit_rect.clipline(self.player_pos, ray_end)
            if intersect:
                dist_sq = self.player_pos.distance_squared_to(intersect[0])
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_type = float(TYPE_ID["EXIT"])

            for puddle in self.puddles:
                intersect = puddle.clipline(self.player_pos, ray_end)
                if intersect:
                    dist_sq = self.player_pos.distance_squared_to(intersect[0])
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        closest_type = float(TYPE_ID["PUDDLE"])

            obs[i, 0] = math.sqrt(min_dist_sq) / LIDAR_MAX_DIST
            obs[i, 1] = closest_type
            
        return obs

    def _calculate_light_polygons(self):
        """game.py와 동일한 조명 다각형 계산 로직"""
        polygons = []
        if not hasattr(self, 'wall_vertices'): return []
        for tower in self.towers:
            light_source = pygame.Vector2(tower['pos'])
            radius = tower['radius']
            tower_angle_rad = math.radians(tower['angle'])
            cone_half_angle_rad = math.radians(tower['cone_angle'] / 2)
            
            angles = [tower_angle_rad - cone_half_angle_rad, tower_angle_rad + cone_half_angle_rad]
            
            for vertex in self.wall_vertices:
                angle = math.atan2(vertex[1] - light_source.y, vertex[0] - light_source.x)
                diff = (angle - tower_angle_rad + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) < cone_half_angle_rad:
                    angles.extend([angle - 0.0001, angle, angle + 0.0001])

            angles = sorted(list(set(angles)))

            for i in range(len(angles) - 1):
                angle1, angle2 = angles[i], angles[i+1]
                
                avg_angle = (angle1 + angle2) / 2
                diff = (avg_angle - tower_angle_rad + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) > cone_half_angle_rad:
                    continue

                points = [light_source]
                for angle in [angle1, angle2]:
                    p_end = light_source + pygame.Vector2(math.cos(angle), math.sin(angle)) * radius
                    closest_intersect = p_end
                    min_dist_sq = radius**2
                    
                    for wall in self.walls:
                        inter = wall.clipline(light_source, p_end)
                        if inter:
                            dist_sq = light_source.distance_squared_to(inter[0])
                            if dist_sq < min_dist_sq:
                                min_dist_sq = dist_sq
                                closest_intersect = pygame.Vector2(inter[0])
                    points.append(closest_intersect)
                
                if len(points) > 2:
                    polygons.append(points)
        return polygons