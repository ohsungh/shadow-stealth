
import pygame
import math
import random
import numpy as np
from collections import deque

# --- 기본 설정 ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
CELL_SIZE = 40
GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

# --- 색상 팔레트 ---
COLOR_BACKGROUND = pygame.Color("black")
COLOR_PLAYER = pygame.Color("#00ffdd")
COLOR_PLAYER_GLOW = pygame.Color(0, 255, 221, 50)
COLOR_EXIT = pygame.Color("#a2ff00")
COLOR_WALL = pygame.Color("#cc2222")
COLOR_PUDDLE = pygame.Color(74, 90, 121, 100)
COLOR_PUDDLE_BORDER = pygame.Color(150, 180, 220)
COLOR_LIGHT = pygame.Color(255, 255, 0, 50)
COLOR_HAZARD = pygame.Color("#ff0055")
COLOR_TOWER = pygame.Color("white")
COLOR_TEXT = pygame.Color(230, 230, 230)

# --- 물리 및 게임 상수 ---
PLAYER_SPEED = 4.0
ENEMY_SPEED = 1.8
LIGHT_EXPOSURE_LIMIT = 0.7 # 0.7초

# --- 유틸리티 함수 ---
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

# --- 개방형 랜덤 맵 생성 함수 ---
def is_solvable(walls, player_pos, exit_rect):
    start_node = (int(player_pos[0] // CELL_SIZE), int(player_pos[1] // CELL_SIZE))
    end_node = (int(exit_rect.centerx // CELL_SIZE), int(exit_rect.centery // CELL_SIZE))
    grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for wall in walls:
        for x in range(wall.left // CELL_SIZE, wall.right // CELL_SIZE + 1):
            for y in range(wall.top // CELL_SIZE, wall.bottom // CELL_SIZE + 1):
                if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT: grid[y][x] = 1
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
    for _ in range(max_walls * 3):
        if len(walls) >= max_walls: break
        w = random.randint(CELL_SIZE * 2, CELL_SIZE * 6)
        h = random.randint(CELL_SIZE * 2, CELL_SIZE * 6)
        x = random.randint(0, SCREEN_WIDTH - w)
        y = random.randint(0, SCREEN_HEIGHT - h)
        new_wall = pygame.Rect(x, y, w, h)
        player_area = pygame.Rect(*player_start, 1, 1).inflate(CELL_SIZE*2, CELL_SIZE*2)
        if new_wall.colliderect(player_area) or new_wall.colliderect(exit_rect.inflate(CELL_SIZE*2, CELL_SIZE*2)): continue
        if is_solvable(walls + [new_wall], player_start, exit_rect): walls.append(new_wall)
    open_spaces = []
    for i in range(200):
        x, y = random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)
        if not any(wall.collidepoint(x, y) for wall in walls): open_spaces.append((x, y))
    if not open_spaces: open_spaces.append((SCREEN_WIDTH/3, SCREEN_HEIGHT/3))
    num_towers = min(len(open_spaces), 1 + difficulty)
    towers = [{"pos": open_spaces.pop(random.randrange(len(open_spaces))), "angle": random.randint(0, 360), "speed": random.uniform(0.5, 1.0) * random.choice([-1, 1]), "cone_angle": random.randint(25, 40), "radius": random.randint(300, 450)} for _ in range(num_towers)]
    num_puddles = min(len(open_spaces), 5 + difficulty * 2)
    puddles = [pygame.Rect(*open_spaces.pop(random.randrange(len(open_spaces))), CELL_SIZE, CELL_SIZE) for _ in range(num_puddles)]
    num_enemies = min(len(open_spaces), difficulty)
    enemies = [{"pos": open_spaces.pop(random.randrange(len(open_spaces)))} for _ in range(num_enemies)]
    return {"player_start": player_start, "exit": exit_rect, "walls": walls, "puddles": puddles, "towers": towers, "enemies": enemies}

# --- 게임 객체 클래스 ---
class Player:
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y); self.vel = pygame.Vector2(0, 0); self.radius = 12; self.speed = PLAYER_SPEED
    def update(self, walls):
        self.vel = pygame.Vector2(0, 0)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: self.vel.x = -1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: self.vel.x = 1
        if keys[pygame.K_UP] or keys[pygame.K_w]: self.vel.y = -1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]: self.vel.y = 1
        if self.vel.length() != 0: self.vel.normalize_ip(); self.vel *= self.speed
        self.pos.x += self.vel.x
        player_rect = self.get_rect()
        for wall in walls: 
            if wall.colliderect(player_rect):
                if self.vel.x > 0: player_rect.right = wall.left
                if self.vel.x < 0: player_rect.left = wall.right
                self.pos.x = player_rect.centerx
        self.pos.y += self.vel.y
        player_rect = self.get_rect()
        for wall in walls: 
            if wall.colliderect(player_rect):
                if self.vel.y > 0: player_rect.bottom = wall.top
                if self.vel.y < 0: player_rect.top = wall.bottom
                self.pos.y = player_rect.centery
        self.pos.x = np.clip(self.pos.x, self.radius, SCREEN_WIDTH - self.radius)
        self.pos.y = np.clip(self.pos.y, self.radius, SCREEN_HEIGHT - self.radius)
    def get_rect(self):
        return pygame.Rect(self.pos.x - self.radius, self.pos.y - self.radius, self.radius * 2, self.radius * 2)
    def draw(self, screen):
        for i in range(5, 0, -1):
            glow_radius = self.radius + i * 3
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
            screen.blit(glow_surface, (self.pos.x - glow_radius, self.pos.y - glow_radius))
        pygame.draw.circle(screen, COLOR_PLAYER, self.pos, self.radius)

class Enemy:
    def __init__(self, pos):
        self.pos = pygame.Vector2(pos); self.radius = 15; self.speed = ENEMY_SPEED
    def update(self, player_pos, walls, is_player_hidden):
        if is_player_hidden: return
        direction = player_pos - self.pos
        if 0 < direction.length() < 300:
            can_see = not any(wall.clipline(self.pos, player_pos) for wall in walls)
            if can_see: direction.normalize_ip(); self.pos += direction * self.speed
    def draw(self, screen):
        pygame.draw.circle(screen, COLOR_HAZARD, self.pos, self.radius)

# --- 메인 게임 클래스 ---
class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Shadow Stealth - Open Random")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 50)
        self.small_font = pygame.font.Font(None, 36)
        self.running = True
        self.difficulty = 0
        self.load_next_level()

    def load_next_level(self):
        level_data = generate_level(self.difficulty)
        self.player = Player(*level_data["player_start"])
        self.exit_rect = level_data["exit"]
        self.walls = level_data["walls"]
        self.puddles = level_data["puddles"]
        self.towers = level_data["towers"]
        self.enemies = [Enemy(**e) for e in level_data["enemies"]]
        self.wall_vertices = get_wall_vertices(self.walls)
        self.level_start_time = pygame.time.get_ticks()
        self.game_state = "PLAYING"
        self.light_exposure_time = 0
        self.light_polygons = []
        self.score = 1000 # <--- 시작 점수 1000으로 수정
        self.last_score_tick = pygame.time.get_ticks() # <--- 초당 점수 감소용 타이머

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                if event.type == pygame.KEYDOWN and self.game_state != "PLAYING":
                    if event.key == pygame.K_r: self.load_next_level()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()

    def update(self):
        if self.game_state != "PLAYING": return
        self.player.update(self.walls)
        for tower in self.towers: tower['angle'] = (tower['angle'] + tower['speed']) % 360
        self.light_polygons = self.calculate_light_polygons()
        is_in_puddle = any(puddle.colliderect(self.player.get_rect()) for puddle in self.puddles)
        is_in_light = any(point_in_polygon(self.player.pos, poly) for poly in self.light_polygons)
        is_hidden = is_in_puddle
        for enemy in self.enemies: enemy.update(self.player.pos, self.walls, is_hidden)
        # 점수 및 게임 오버 로직
        if pygame.time.get_ticks() - self.last_score_tick > 1000: # <--- 1초마다
            self.score -= 1 # <--- 1점씩만 감소
            self.last_score_tick = pygame.time.get_ticks()
        if is_in_light and not is_hidden:
            self.light_exposure_time += 1 / FPS
            self.score -= 2 # 빛 노출 페널티 (프레임당 2점이므로 초당 120점)
            if self.light_exposure_time > LIGHT_EXPOSURE_LIMIT: self.game_state = "GAME_OVER"
        else:
            self.light_exposure_time = max(0, self.light_exposure_time - 1 / FPS)
        for enemy in self.enemies:
            if self.player.pos.distance_to(enemy.pos) < self.player.radius + enemy.radius: self.game_state = "GAME_OVER"
        if self.player.get_rect().colliderect(self.exit_rect):
            self.difficulty += 1; self.game_state = "LEVEL_CLEAR"

    def draw(self):
        self.screen.fill(COLOR_BACKGROUND)
        for poly in self.light_polygons: pygame.draw.polygon(self.screen, COLOR_LIGHT, poly)
        for wall in self.walls: pygame.draw.rect(self.screen, COLOR_WALL, wall)
        for puddle in self.puddles:
            border_alpha = 100 + math.sin(pygame.time.get_ticks() * 0.005) * 50
            border_color = (*COLOR_PUDDLE_BORDER[:3], border_alpha)
            pygame.draw.rect(self.screen, border_color, puddle.inflate(6, 6), 2, border_radius=5)
            puddle_surf = pygame.Surface(puddle.size, pygame.SRCALPHA); puddle_surf.fill(COLOR_PUDDLE); self.screen.blit(puddle_surf, puddle.topleft)
        pygame.draw.rect(self.screen, COLOR_EXIT, self.exit_rect, border_radius=10)
        for tower in self.towers: pygame.draw.circle(self.screen, COLOR_TOWER, tower['pos'], 10)
        for enemy in self.enemies: enemy.draw(self.screen)
        self.player.draw(self.screen)
        # UI 그리기
        time_text = self.font.render(f"Time: {((pygame.time.get_ticks() - self.level_start_time) / 1000):.1f}", True, COLOR_TEXT)
        score_text = self.font.render(f"Score: {int(self.score)}", True, COLOR_TEXT) # <--- 정수로 표시
        self.screen.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 20, 20))
        self.screen.blit(score_text, (20, 20))
        if self.light_exposure_time > 0:
            bar_rect = pygame.Rect(self.player.pos.x - 25, self.player.pos.y - self.player.radius - 20, 50, 10)
            fill_ratio = self.light_exposure_time / LIGHT_EXPOSURE_LIMIT
            pygame.draw.rect(self.screen, (80,80,80), bar_rect, 2)
            pygame.draw.rect(self.screen, COLOR_HAZARD, (bar_rect.x, bar_rect.y, bar_rect.width * fill_ratio, bar_rect.height))
        if self.game_state != "PLAYING":
            self.draw_overlay()
        pygame.display.flip()

    def draw_overlay(self):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        text = "GAME OVER" if self.game_state == "GAME_OVER" else f"LEVEL {self.difficulty} CLEAR!"
        main_text = self.font.render(text, True, COLOR_TEXT)
        sub_text = self.small_font.render("Press 'R' to Continue", True, COLOR_TEXT)
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(main_text, main_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 30)))
        self.screen.blit(sub_text, sub_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 30)))

    def calculate_light_polygons(self):
        polygons = []
        for tower in self.towers:
            light_source = pygame.Vector2(tower['pos'])
            radius = tower['radius']
            tower_angle_rad = math.radians(tower['angle'])
            cone_half_angle_rad = math.radians(tower['cone_angle'] / 2)
            rays = [tower_angle_rad - cone_half_angle_rad, tower_angle_rad + cone_half_angle_rad]
            for vertex in self.wall_vertices:
                angle = math.atan2(vertex[1] - light_source.y, vertex[0] - light_source.x)
                diff = (angle - tower_angle_rad + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) < cone_half_angle_rad: rays.extend([angle - 0.0001, angle, angle + 0.0001])
            rays = sorted(list(set(rays)))
            for i in range(len(rays) - 1):
                angle1, angle2 = rays[i], rays[i+1]
                if (angle2 - angle1) >= math.pi: continue
                points = [light_source]
                for angle in [angle1, angle2]:
                    p_end = light_source + pygame.Vector2(math.cos(angle), math.sin(angle)) * radius
                    closest_intersect = p_end
                    min_dist_sq = radius**2
                    for wall in self.walls:
                        inter = wall.clipline(light_source, p_end)
                        if inter:
                            dist_sq = light_source.distance_squared_to(inter[0])
                            if dist_sq < min_dist_sq: min_dist_sq = dist_sq; closest_intersect = inter[0]
                    points.append(closest_intersect)
                polygons.append(points)
        return polygons

if __name__ == '__main__':
    game = Game()
    game.run()
