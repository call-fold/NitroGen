"""
NitroGen Platformer Test Game

A 2D platformer game designed for testing the NitroGen agent.
Features platforms, enemies, collectibles, and visual feedback.

Controls:
- Left Joystick: Move player
- Right Joystick: Aim cursor (shows direction indicator)
- A Button (South): Jump
- B Button (East): Dash
- X Button (West): Shoot projectile
- Y Button (North): Special ability (shield)
- D-Pad: Navigate menus
- Start: Pause game
- Triggers: Fine movement control

This game provides:
1. Clear visual state for model observation
2. Distinct actions for testing action prediction
3. Score-based feedback
4. Multiple enemy types with different behaviors
5. Platforming challenges
"""

import pygame
import random
import sys
import math
from pathlib import Path

# Game Constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 60

# Colors
BACKGROUND = (15, 15, 30)
BACKGROUND_GRADIENT_1 = (25, 25, 50)
BACKGROUND_GRADIENT_2 = (15, 15, 30)
PLAYER_COLOR = (0, 180, 255)
PLAYER_GLOW = (0, 100, 200)
PLAYER_OUTLINE = (255, 255, 255)
ENEMY_COLOR = (255, 80, 80)
ENEMY_GLOW = (200, 50, 50)
COLLECTIBLE_COLOR = (255, 220, 50)
PROJECTILE_COLOR = (100, 255, 100)
PLATFORM_COLOR = (80, 80, 100)
PLATFORM_TOP = (120, 120, 150)
SHIELD_COLOR = (100, 150, 255)
TEXT_COLOR = (255, 255, 255)


class Particle:
    """Visual particle effects."""
    def __init__(self, x, y, vx, vy, color, size=8, life=30, fade=True):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.size = size
        self.max_size = size
        self.life = life
        self.max_life = life
        self.fade = fade

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.2  # Gravity
        self.life -= 1
        if self.fade:
            self.size = self.max_size * (self.life / self.max_life)

    def draw(self, surface):
        if self.life > 0:
            alpha = min(255, int(255 * (self.life / self.max_life))) if self.fade else 255
            s = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
            color = (*self.color, alpha)
            pygame.draw.circle(s, color, (int(self.size), int(self.size)), int(self.size))
            surface.blit(s, (int(self.x - self.size), int(self.y - self.size)))


class Projectile:
    """Player projectile for attacking enemies."""
    def __init__(self, x, y, vx, vy, owner="player"):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = 8
        self.owner = owner
        self.life = 60

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        pygame.draw.circle(surface, PROJECTILE_COLOR, (int(self.x), int(self.y)), self.radius)
        # Glow effect
        s = pygame.Surface((self.radius * 4, self.radius * 4), pygame.SRCALPHA)
        pygame.draw.circle(s, (*PROJECTILE_COLOR, 100), (self.radius * 2, self.radius * 2), self.radius * 2)
        surface.blit(s, (int(self.x - self.radius * 2), int(self.y - self.radius * 2)))

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)


class Platform:
    """Platform the player can stand on."""
    def __init__(self, x, y, width, height, moving=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.start_x = x
        self.start_y = y
        self.moving = moving
        self.move_speed = 2
        self.move_range = 100
        self.move_direction = 1
        self.angle = 0

    def update(self):
        if self.moving:
            self.angle += 0.05
            offset_x = math.sin(self.angle) * self.move_range
            self.rect.x = int(self.start_x + offset_x)

    def draw(self, surface):
        # Draw platform body
        pygame.draw.rect(surface, PLATFORM_COLOR, self.rect, border_radius=8)
        # Draw top highlight
        pygame.draw.rect(surface, PLATFORM_TOP,
                         (self.rect.x + 4, self.rect.y, self.rect.width - 8, 8), border_radius=4)
        # Draw outline
        pygame.draw.rect(surface, (150, 150, 180), self.rect, 2, border_radius=8)


class Collectible:
    """Collectible item that gives points."""
    def __init__(self, x, y, c_type="coin"):
        self.x = x
        self.y = y
        self.type = c_type  # "coin", "gem", "heart"
        self.radius = 15
        self.angle = 0
        self.collected = False
        self.float_offset = random.random() * 6.28

        if c_type == "coin":
            self.color = (255, 215, 0)
            self.points = 10
        elif c_type == "gem":
            self.color = (255, 100, 255)
            self.points = 25
        elif c_type == "heart":
            self.color = (255, 100, 100)
            self.points = 0
        else:
            self.color = COLLECTIBLE_COLOR
            self.points = 10

    def update(self):
        self.angle += 0.08
        self.float_offset += 0.05

    def draw(self, surface):
        if self.collected:
            return

        pulse = 1 + 0.15 * math.sin(self.angle)
        float_y = math.sin(self.float_offset) * 5
        radius = int(self.radius * pulse)

        # Draw glow
        s = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, 50), (radius * 2, radius * 2), radius * 2)
        surface.blit(s, (int(self.x - radius * 2), int(self.y - radius * 2 + float_y)))

        # Draw main circle
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y + float_y)), radius)
        # Draw inner detail
        pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y + float_y)), radius // 2)

    def get_rect(self):
        float_y = math.sin(self.float_offset) * 5
        return pygame.Rect(self.x - self.radius, self.y - self.radius + float_y,
                          self.radius * 2, self.radius * 2)


class Enemy:
    """Enemy that chases the player."""
    def __init__(self, x, y, e_type="walker"):
        self.x = x
        self.y = y
        self.type = e_type
        self.width = 40
        self.height = 40
        self.vx = 0
        self.vy = 0
        self.speed = 2
        self.health = 2
        self.max_health = 2
        self.angle = 0
        self.jump_timer = 0
        self.direction = 1

        if e_type == "walker":
            self.speed = 2
            self.health = 2
            self.color = ENEMY_COLOR
        elif e_type == "flyer":
            self.speed = 3
            self.health = 1
            self.color = (255, 150, 50)
        elif e_type == "tank":
            self.speed = 1
            self.health = 5
            self.color = (150, 50, 150)
            self.width = 60
            self.height = 60

    def update(self, player, platforms):
        if self.type == "flyer":
            # Fly towards player
            dx = player.x + player.width / 2 - self.x - self.width / 2
            dy = player.y + player.height / 2 - self.y - self.height / 2
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0:
                self.vx = (dx / dist) * self.speed
                self.vy = (dy / dist) * self.speed
        else:
            # Walker and tank: move horizontally towards player, with gravity
            dx = player.x + player.width / 2 - self.x - self.width / 2
            self.direction = 1 if dx > 0 else -1
            self.vx = self.direction * self.speed
            self.vy += 0.5  # Gravity

            # Simple platform collision
            on_ground = False
            for platform in platforms:
                if platform.rect.colliderect(
                    pygame.Rect(self.x + 5, self.y + self.height + self.vy,
                               self.width - 10, self.vy + 1)
                ):
                    self.y = platform.rect.top - self.height
                    self.vy = 0
                    on_ground = True

                    # Jump over obstacles
                    self.jump_timer += 1
                    if self.jump_timer > 60 and random.random() < 0.02:
                        self.vy = -10
                        self.jump_timer = 0

            if not on_ground:
                self.jump_timer = 0

        self.x += self.vx
        self.y += self.vy

        # Keep in bounds
        self.x = max(0, min(WINDOW_WIDTH - self.width, self.x))
        self.y = max(0, min(WINDOW_HEIGHT - self.height, self.y))

        self.angle += 0.1

    def draw(self, surface):
        # Draw glow
        s = pygame.Surface((self.width * 2, self.height * 2), pygame.SRCALPHA)
        glow_color = (*self.color, 60)
        pygame.draw.rect(s, glow_color,
                        (self.width // 2, self.height // 2, self.width, self.height), border_radius=8)
        surface.blit(s, (int(self.x - self.width // 2), int(self.y - self.height // 2)))

        # Draw enemy body
        pulse = 1 + 0.05 * math.sin(self.angle)
        size_x = int(self.width * pulse)
        size_y = int(self.height * pulse)
        rect = pygame.Rect(
            self.x + (self.width - size_x) // 2,
            self.y + (self.height - size_y) // 2,
            size_x, size_y
        )
        pygame.draw.rect(surface, self.color, rect, border_radius=8)
        pygame.draw.rect(surface, (255, 255, 255), rect, 2, border_radius=8)

        # Draw eyes
        eye_offset = self.direction * (size_x // 4)
        eye_y = self.y + size_y // 3
        pygame.draw.circle(surface, (255, 255, 255),
                          (int(self.x + size_x // 2 + eye_offset - 8), int(eye_y)), 6)
        pygame.draw.circle(surface, (255, 255, 255),
                          (int(self.x + size_x // 2 + eye_offset + 8), int(eye_y)), 6)
        pygame.draw.circle(surface, (0, 0, 0),
                          (int(self.x + size_x // 2 + eye_offset - 8), int(eye_y)), 3)
        pygame.draw.circle(surface, (0, 0, 0),
                          (int(self.x + size_x // 2 + eye_offset + 8), int(eye_y)), 3)

        # Draw health bar
        if self.health < self.max_health:
            bar_width = size_x
            bar_height = 4
            bar_x = self.x + (self.width - bar_width) // 2
            bar_y = self.y - 10
            pygame.draw.rect(surface, (100, 100, 100),
                            (bar_x, bar_y, bar_width, bar_height))
            health_width = int(bar_width * (self.health / self.max_health))
            pygame.draw.rect(surface, (255, 50, 50), (bar_x, bar_y, health_width, bar_height))

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Player:
    """The player character controlled by gamepad."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 50
        self.height = 50
        self.vel_x = 0
        self.vel_y = 0
        self.speed = 6
        self.max_speed = 6
        self.jump_power = -14
        self.gravity = 0.7
        self.on_ground = False
        self.facing_right = True

        # Cooldowns and states
        self.jump_cooldown = 0
        self.dash_cooldown = 0
        self.shoot_cooldown = 0
        self.shield_cooldown = 0
        self.is_dashing = False
        self.dash_timer = 0
        self.has_shield = False
        self.shield_timer = 0

        # Aiming
        self.aim_x = 0
        self.aim_y = 0

        # Health
        self.max_health = 3
        self.health = 3
        self.invincible_timer = 0

    def update(self, left_stick_x, left_stick_y, right_stick_x, right_stick_y,
               a_pressed, b_pressed, x_pressed, y_pressed,
               left_trigger, right_trigger, platforms):
        """Update player based on gamepad input."""

        # Movement
        input_x = 0
        input_y = 0

        # Left joystick or triggers for fine control
        if abs(left_stick_x) > 0.1:
            input_x = left_stick_x
        else:
            input_x = (right_trigger - left_trigger) * 2

        if abs(left_stick_y) > 0.1:
            input_y = left_stick_y

        # Apply dash
        if self.is_dashing:
            self.dash_timer -= 1
            if self.dash_timer <= 0:
                self.is_dashing = False
                self.vel_x *= 0.1
        else:
            # Normal movement
            self.vel_x = input_x * self.speed

        # Update facing direction
        if input_x > 0.1:
            self.facing_right = True
        elif input_x < -0.1:
            self.facing_right = False

        # Store aim direction
        if abs(right_stick_x) > 0.1 or abs(right_stick_y) > 0.1:
            self.aim_x = right_stick_x
            self.aim_y = right_stick_y

        # Apply gravity
        self.vel_y += self.gravity

        # Jump (A button)
        if a_pressed and self.on_ground and self.jump_cooldown <= 0:
            self.vel_y = self.jump_power
            self.on_ground = False
            self.jump_cooldown = 10
            return "jump"

        # Dash (B button)
        if b_pressed and self.dash_cooldown <= 0 and not self.is_dashing:
            dash_dir = 1 if self.facing_right else -1
            self.vel_x = dash_dir * 20
            self.is_dashing = True
            self.dash_timer = 8
            self.dash_cooldown = 40
            return "dash"

        # Apply velocity
        if not self.is_dashing:
            self.x += self.vel_x
            self.y += self.vel_y

        # Collision detection with platforms
        self.on_ground = False
        for platform in platforms:
            # Vertical collision
            if platform.rect.colliderect(
                pygame.Rect(self.x, self.y, self.width, self.height + self.vel_y + 1)
            ):
                if self.vel_y > 0:  # Falling
                    self.y = platform.rect.top - self.height
                    self.vel_y = 0
                    self.on_ground = True
                elif self.vel_y < 0:  # Jumping up
                    self.y = platform.rect.bottom
                    self.vel_y = 0

            # Horizontal collision
            if platform.rect.colliderect(
                pygame.Rect(self.x + self.vel_x, self.y, self.width + abs(self.vel_y), self.height)
            ):
                if self.vel_x > 0:
                    self.x = platform.rect.left - self.width
                elif self.vel_x < 0:
                    self.x = platform.rect.right

        # Screen boundaries
        if self.x < 0:
            self.x = 0
            self.vel_x = 0
        if self.x > WINDOW_WIDTH - self.width:
            self.x = WINDOW_WIDTH - self.width
            self.vel_x = 0
        if self.y > WINDOW_HEIGHT - self.height:
            self.y = WINDOW_HEIGHT - self.height
            self.vel_y = 0
            self.on_ground = True
        if self.y < 0:
            self.y = 0
            self.vel_y = 0

        # Cooldowns
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
        if self.dash_cooldown > 0:
            self.dash_cooldown -= 1
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.shield_cooldown > 0:
            self.shield_cooldown -= 1

        # Shield timer
        if self.has_shield:
            self.shield_timer -= 1
            if self.shield_timer <= 0:
                self.has_shield = False

        # Invincibility timer
        if self.invincible_timer > 0:
            self.invincible_timer -= 1

        return None

    def shoot(self):
        """Shoot projectile in aim direction."""
        if self.shoot_cooldown <= 0:
            speed = 12
            vx = speed * (self.aim_x if abs(self.aim_x) > 0.1 else (1 if self.facing_right else -1))
            vy = speed * self.aim_y
            # Normalize
            mag = math.sqrt(vx * vx + vy * vy)
            if mag > 0:
                vx = (vx / mag) * speed
                vy = (vy / mag) * speed
            self.shoot_cooldown = 15
            return Projectile(self.x + self.width // 2, self.y + self.height // 2, vx, vy)
        return None

    def activate_shield(self):
        """Activate shield ability."""
        if not self.has_shield and self.shield_cooldown <= 0:
            self.has_shield = True
            self.shield_timer = 120  # 2 seconds
            self.shield_cooldown = 300  # 5 seconds
            return True
        return False

    def take_damage(self):
        """Take damage if not invincible."""
        if self.has_shield:
            self.has_shield = False
            return False
        if self.invincible_timer <= 0:
            self.health -= 1
            self.invincible_timer = 60  # 1 second
            return True
        return False

    def draw(self, surface):
        # Flicker when invincible
        if self.invincible_timer > 0 and (self.invincible_timer // 4) % 2 == 0:
            return

        # Draw glow
        s = pygame.Surface((self.width * 3, self.height * 3), pygame.SRCALPHA)
        pygame.draw.rect(s, (*PLAYER_GLOW, 80),
                        (self.width, self.height, self.width, self.height), border_radius=10)
        surface.blit(s, (int(self.x - self.width), int(self.y - self.height)))

        # Draw player body
        rect = pygame.Rect(int(self.x), int(self.y), self.width, self.height)
        pygame.draw.rect(surface, PLAYER_COLOR, rect, border_radius=10)
        pygame.draw.rect(surface, PLAYER_OUTLINE, rect, 3, border_radius=10)

        # Draw direction indicator (eyes)
        eye_offset = self.width // 4
        eye_x = self.x + self.width // 2 + (eye_offset if self.facing_right else -eye_offset)
        eye_y = self.y + 15

        pygame.draw.circle(surface, PLAYER_OUTLINE, (int(eye_x), int(eye_y)), 10)
        pygame.draw.circle(surface, (0, 0, 0), (int(eye_x), int(eye_y)), 5)

        # Draw aim indicator
        if abs(self.aim_x) > 0.1 or abs(self.aim_y) > 0.1:
            aim_len = 40
            aim_x = self.x + self.width // 2 + self.aim_x * aim_len
            aim_y = self.y + self.height // 2 + self.aim_y * aim_len
            pygame.draw.line(surface, PLAYER_OUTLINE,
                           (self.x + self.width // 2, self.y + self.height // 2),
                           (aim_x, aim_y), 3)
            pygame.draw.circle(surface, PLAYER_COLOR, (int(aim_x), int(aim_y)), 5)

        # Draw shield
        if self.has_shield:
            shield_radius = 40
            s = pygame.Surface((shield_radius * 2, shield_radius * 2), pygame.SRCALPHA)
            alpha = 150 + int(100 * math.sin(pygame.time.get_ticks() * 0.01))
            pygame.draw.circle(s, (*SHIELD_COLOR, alpha), (shield_radius, shield_radius), shield_radius, 3)
            surface.blit(s, (int(self.x + self.width // 2 - shield_radius),
                           int(self.y + self.height // 2 - shield_radius)))

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class PlatformerGame:
    """Main game class."""
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("NitroGen Platformer Test Game")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # Game objects
        self.player = Player(100, WINDOW_HEIGHT - 200)
        self.platforms = []
        self.collectibles = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        # Game state
        self.score = 0
        self.high_score = 0
        self.lives = 3
        self.game_over = False
        self.paused = False
        self.running = True
        self.spawn_timer = 0
        self.difficulty = 1.0

        # Joystick setup
        self.joystick = None
        self.joystick_detected = False
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.joystick_detected = True
            print(f"[+] Joystick detected: {self.joystick.get_name()}")
        else:
            print("[-] No joystick detected. Using keyboard fallback.")

        # Initialize level
        self.create_level()

    def create_level(self):
        """Create platforms and initial collectibles."""
        # Ground platform
        self.platforms.append(Platform(0, WINDOW_HEIGHT - 20, WINDOW_WIDTH, 20))

        # Various platforms
        platform_configs = [
            (200, 550, 200, 20),
            (500, 450, 250, 20),
            (850, 350, 200, 20),
            (100, 350, 150, 20),
            (350, 250, 200, 20),
            (700, 200, 150, 20, True),  # Moving platform
            (950, 500, 200, 20),
            (50, 150, 200, 20),
        ]

        for config in platform_configs:
            moving = len(config) > 4
            # If moving flag is in config, use it; otherwise pass it separately
            if moving:
                self.platforms.append(Platform(*config))
            else:
                self.platforms.append(Platform(*config, moving=False))

        # Initial collectibles
        for _ in range(5):
            self.spawn_collectible()

    def spawn_collectible(self):
        """Spawn a collectible at a random platform location."""
        if not self.platforms:
            return

        platform = random.choice(self.platforms)
        x = random.randint(platform.rect.left + 30, platform.rect.right - 30)
        y = platform.rect.top - 30

        c_type = random.choice(["coin"] * 6 + ["gem"] * 2 + ["heart"] * 1)
        self.collectibles.append(Collectible(x, y, c_type))

    def spawn_enemy(self):
        """Spawn an enemy at the edge of the screen."""
        e_type = random.choice(
            ["walker"] * 5 +
            ["flyer"] * 2 +
            (["tank"] * 1 if self.difficulty > 1.5 else [])
        )

        y = random.randint(100, WINDOW_HEIGHT - 100)
        if e_type == "flyer":
            y = random.randint(50, 200)
        else:
            y = WINDOW_HEIGHT - 60

        from_left = random.random() < 0.5
        x = -50 if from_left else WINDOW_WIDTH + 50

        enemy = Enemy(x, y, e_type)
        if from_left:
            enemy.direction = 1
        else:
            enemy.direction = -1

        self.enemies.append(enemy)

    def get_input(self):
        """Get input from gamepad or keyboard."""
        left_stick_x = 0
        left_stick_y = 0
        right_stick_x = 0
        right_stick_y = 0
        a_pressed = False  # Jump
        b_pressed = False  # Dash
        x_pressed = False  # Shoot
        y_pressed = False  # Shield
        left_trigger = 0
        right_trigger = 0
        start_pressed = False

        if self.joystick:
            # Joysticks (-1 to 1)
            left_stick_x = self.joystick.get_axis(0)
            left_stick_y = self.joystick.get_axis(1)
            right_stick_x = self.joystick.get_axis(2)
            right_stick_y = self.joystick.get_axis(3)

            # Triggers (0 to 1)
            left_trigger = self.joystick.get_axis(4)
            right_trigger = self.joystick.get_axis(5)

            # Buttons (Xbox: 0=A, 1=B, 2=X, 3=Y, 7=Start)
            a_pressed = self.joystick.get_button(0)
            b_pressed = self.joystick.get_button(1)
            x_pressed = self.joystick.get_button(2)
            y_pressed = self.joystick.get_button(3)
            start_pressed = self.joystick.get_button(7)
        else:
            # Keyboard fallback
            keys = pygame.key.get_pressed()

            # Movement
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                left_stick_x = -1
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                left_stick_x = 1
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                left_stick_y = -1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                left_stick_y = 1

            # Aiming with arrow keys
            if keys[pygame.K_j]:
                right_stick_x = -1
            elif keys[pygame.K_l]:
                right_stick_x = 1
            if keys[pygame.K_i]:
                right_stick_y = -1
            elif keys[pygame.K_k]:
                right_stick_y = 1

            a_pressed = keys[pygame.K_SPACE]
            b_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            x_pressed = keys[pygame.K_z] or keys[pygame.K_x]
            y_pressed = keys[pygame.K_c]

        return (left_stick_x, left_stick_y, right_stick_x, right_stick_y,
                a_pressed, b_pressed, x_pressed, y_pressed,
                left_trigger, right_trigger, start_pressed)

    def spawn_particles(self, x, y, color, count=10, speed_range=(3, 8)):
        """Spawn visual particles."""
        for _ in range(count):
            angle = random.random() * 6.28
            speed = random.uniform(*speed_range)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.particles.append(Particle(x, y, vx, vy, color))

    def check_collisions(self):
        """Check all game collisions."""
        # Player - Collectibles
        player_rect = self.player.get_rect()
        for collectible in self.collectibles[:]:
            if not collectible.collected and player_rect.colliderect(collectible.get_rect()):
                collectible.collected = True
                self.collectibles.remove(collectible)

                if collectible.type == "heart":
                    if self.player.health < self.player.max_health:
                        self.player.health += 1
                else:
                    self.score += collectible.points

                self.spawn_particles(collectible.x, collectible.y, collectible.color, 8)

                # Spawn new collectible
                if random.random() < 0.7:
                    self.spawn_collectible()

        # Projectiles - Enemies
        for projectile in self.projectiles[:]:
            proj_rect = projectile.get_rect()
            for enemy in self.enemies[:]:
                if proj_rect.colliderect(enemy.get_rect()):
                    enemy.health -= 1
                    self.spawn_particles(proj_rect.centerx, proj_rect.centery,
                                       PROJECTILE_COLOR, 5)
                    if projectile in self.projectiles:
                        self.projectiles.remove(projectile)

                    if enemy.health <= 0:
                        self.enemies.remove(enemy)
                        self.score += 50
                        self.spawn_particles(enemy.x + enemy.width // 2,
                                           enemy.y + enemy.height // 2,
                                           enemy.color, 15, speed_range=(5, 12))
                        self.difficulty += 0.05
                    break

        # Player - Enemies
        for enemy in self.enemies[:]:
            if player_rect.colliderect(enemy.get_rect()):
                if self.player.take_damage():
                    self.spawn_particles(self.player.x + self.player.width // 2,
                                       self.player.y + self.player.height // 2,
                                       (255, 100, 100), 10)
                    if self.player.health <= 0:
                        self.lives -= 1
                        self.reset_player()

    def reset_player(self):
        """Reset player after losing a life."""
        if self.lives <= 0:
            self.game_over = True
            if self.score > self.high_score:
                self.high_score = self.score
        else:
            self.player.x = 100
            self.player.y = WINDOW_HEIGHT - 200
            self.player.vel_x = 0
            self.player.vel_y = 0
            self.player.health = self.player.max_health

    def update(self):
        """Update game state."""
        if self.game_over or self.paused:
            return

        # Get input
        (left_stick_x, left_stick_y, right_stick_x, right_stick_y,
         a_pressed, b_pressed, x_pressed, y_pressed,
         left_trigger, right_trigger, start_pressed) = self.get_input()

        # Check pause
        if start_pressed:
            self.paused = True
            return

        # Update player
        action = self.player.update(left_stick_x, left_stick_y, right_stick_x, right_stick_y,
                                    a_pressed, b_pressed, x_pressed, y_pressed,
                                    left_trigger, right_trigger, self.platforms)

        if action == "jump":
            self.spawn_particles(self.player.x + self.player.width // 2,
                               self.player.y + self.player.height,
                               (200, 200, 255), 5)

        if action == "dash":
            self.spawn_particles(self.player.x + self.player.width // 2,
                               self.player.y + self.player.height // 2,
                               PLAYER_COLOR, 8)

        # Shoot
        if x_pressed:
            proj = self.player.shoot()
            if proj:
                self.projectiles.append(proj)

        # Shield
        if y_pressed:
            self.player.activate_shield()

        # Update platforms
        for platform in self.platforms:
            platform.update()

        # Update collectibles
        for collectible in self.collectibles:
            collectible.update()

        # Spawn enemies
        self.spawn_timer += 1
        spawn_rate = max(60, int(180 / self.difficulty))
        if self.spawn_timer >= spawn_rate and len(self.enemies) < 10:
            self.spawn_enemy()
            self.spawn_timer = 0

        # Update enemies
        for enemy in self.enemies:
            enemy.update(self.player, self.platforms)

        # Update projectiles
        self.projectiles = [p for p in self.projectiles if p.update()]

        # Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

        # Check collisions
        self.check_collisions()

    def draw_background(self):
        """Draw animated background."""
        # Create gradient
        for y in range(WINDOW_HEIGHT):
            ratio = y / WINDOW_HEIGHT
            r = int(BACKGROUND_GRADIENT_1[0] * (1 - ratio) + BACKGROUND_GRADIENT_2[0] * ratio)
            g = int(BACKGROUND_GRADIENT_1[1] * (1 - ratio) + BACKGROUND_GRADIENT_2[1] * ratio)
            b = int(BACKGROUND_GRADIENT_1[2] * (1 - ratio) + BACKGROUND_GRADIENT_2[2] * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (WINDOW_WIDTH, y))

        # Draw subtle grid pattern
        grid_size = 100
        offset = int(pygame.time.get_ticks() * 0.02) % grid_size
        for x in range(-grid_size, WINDOW_WIDTH + grid_size, grid_size):
            pygame.draw.line(self.screen, (30, 30, 60),
                           (x + offset, 0), (x + offset, WINDOW_HEIGHT), 1)

    def draw_hud(self):
        """Draw heads-up display."""
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, TEXT_COLOR)
        self.screen.blit(score_text, (20, 20))

        # High score
        if self.high_score > 0:
            hs_text = self.font_small.render(f"High: {self.high_score}", True, (150, 150, 150))
            self.screen.blit(hs_text, (20, 55))

        # Lives
        lives_text = self.font.render(f"Lives: {self.lives}", True, TEXT_COLOR)
        self.screen.blit(lives_text, (WINDOW_WIDTH - 150, 20))

        # Health bar
        bar_width = 150
        bar_height = 20
        bar_x = WINDOW_WIDTH - 170
        bar_y = 55
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        health_width = int(bar_width * (self.player.health / self.player.max_health))
        health_color = (50, 200, 50) if self.player.health > 2 else (200, 200, 50) if self.player.health > 1 else (200, 50, 50)
        pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, health_width, bar_height), border_radius=5)

        # Difficulty
        diff_text = self.font_small.render(f"Difficulty: {self.difficulty:.1f}x", True, (150, 150, 150))
        diff_rect = diff_text.get_rect(center=(WINDOW_WIDTH // 2, 25))
        self.screen.blit(diff_text, diff_rect)

        # Input display
        controls = [
            ("L-Stick: Move", (100, 100, 100)),
            ("R-Stick: Aim", (100, 100, 100)),
            ("A: Jump", (50, 200, 150)),
            ("B: Dash", (200, 150, 50)),
            ("X: Shoot", (100, 255, 100)),
            ("Y: Shield", (100, 150, 255)),
        ]
        y_offset = WINDOW_HEIGHT - 25
        x_offset = 20
        for text, color in controls:
            t = self.font_small.render(text, True, color)
            self.screen.blit(t, (x_offset, y_offset))
            x_offset += t.get_width() + 20

    def draw_game_over(self):
        """Draw game over screen."""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        go_text = self.font_large.render("GAME OVER", True, (255, 100, 100))
        go_rect = go_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 60))
        self.screen.blit(go_text, go_rect)

        score_text = self.font.render(f"Final Score: {self.score}", True, TEXT_COLOR)
        score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        self.screen.blit(score_text, score_rect)

        hs_text = self.font.render(f"High Score: {self.high_score}", True, (255, 215, 0))
        hs_rect = hs_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 40))
        self.screen.blit(hs_text, hs_rect)

        restart_text = self.font_small.render("Press SPACE or A to restart", True, (150, 150, 150))
        restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 90))
        self.screen.blit(restart_text, restart_rect)

    def draw_pause(self):
        """Draw pause screen."""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.screen.blit(overlay, (0, 0))

        pause_text = self.font_large.render("PAUSED", True, TEXT_COLOR)
        pause_rect = pause_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        self.screen.blit(pause_text, pause_rect)

        hint_text = self.font_small.render("Press START to resume", True, (150, 150, 150))
        hint_rect = hint_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
        self.screen.blit(hint_text, hint_rect)

    def draw(self):
        """Draw everything."""
        self.draw_background()

        # Draw platforms
        for platform in self.platforms:
            platform.draw(self.screen)

        # Draw collectibles
        for collectible in self.collectibles:
            collectible.draw(self.screen)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw projectiles
        for proj in self.projectiles:
            proj.draw(self.screen)

        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(self.screen)

        # Draw player
        self.player.draw(self.screen)

        # Draw HUD
        self.draw_hud()

        # Game over or pause overlay
        if self.game_over:
            self.draw_game_over()
        elif self.paused:
            self.draw_pause()

        pygame.display.flip()

    def reset_game(self):
        """Reset the game to initial state."""
        self.player = Player(100, WINDOW_HEIGHT - 200)
        self.enemies.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.collectibles.clear()
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.paused = False
        self.difficulty = 1.0
        self.spawn_timer = 0

        self.platforms.clear()
        self.create_level()

        for _ in range(5):
            self.spawn_collectible()

    def run(self):
        """Main game loop."""
        while self.running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif self.game_over and event.key == pygame.K_SPACE:
                        self.reset_game()

            # Handle game over restart
            if self.game_over:
                _, _, _, _, a_pressed, _, _, _, _, _, start_pressed = self.get_input()
                if a_pressed:
                    self.reset_game()

            # Handle pause toggle
            if self.paused:
                _, _, _, _, _, _, _, _, _, _, start_pressed = self.get_input()
                if start_pressed:
                    self.paused = False

            # Update and draw
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


def main():
    """Run the game."""
    print("=" * 70)
    print("      NitroGen Platformer Test Game")
    print("=" * 70)
    print("A 2D platformer for testing NitroGen agent capabilities.")
    print("")
    print("GAMEPAD CONTROLS:")
    print("  Left Joystick  : Move player")
    print("  Right Joystick : Aim cursor")
    print("  A Button       : Jump")
    print("  B Button       : Dash")
    print("  X Button       : Shoot projectile")
    print("  Y Button       : Shield ability")
    print("  Start Button   : Pause game")
    print("  Triggers       : Fine movement control")
    print("")
    print("KEYBOARD FALLBACK:")
    print("  WASD / Arrows : Move")
    print("  IJKL          : Aim")
    print("  Space         : Jump")
    print("  Shift         : Dash")
    print("  Z / X         : Shoot")
    print("  C             : Shield")
    print("  ESC           : Quit")
    print("=" * 70)

    game = PlatformerGame()
    game.run()


if __name__ == "__main__":
    main()
