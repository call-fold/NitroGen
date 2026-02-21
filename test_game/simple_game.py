"""
Simple test game for NitroGen project.

This is a basic Pygame-based game that can be controlled via gamepad.
It provides visual feedback for testing the NitroGen agent's screen capture
and action prediction capabilities.

Controls:
- Left Joystick: Move player
- A Button (South): Jump
- B Button (East): Dash
- X Button (West): Attack (spawn particle)
"""

import pygame
import random
import sys
from pathlib import Path

# Game Constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 60

# Colors
BACKGROUND = (20, 20, 40)
PLAYER_COLOR = (50, 200, 150)
PLAYER_OUTLINE = (255, 255, 255)
COLLECTIBLE_COLOR = (255, 200, 50)
PARTICLE_COLOR = (200, 100, 255)
TEXT_COLOR = (255, 255, 255)


class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 60
        self.height = 60
        self.vel_x = 0
        self.vel_y = 0
        self.speed = 8
        self.jump_power = -15
        self.gravity = 0.8
        self.on_ground = False
        self.jump_cooldown = 0
        self.dash_cooldown = 0
        self.attack_cooldown = 0
        self.facing_right = True

    def update(self, left_stick_x, left_stick_y, a_pressed, b_pressed, x_pressed):
        """Update player state based on gamepad input."""

        # Movement with left joystick (normalized -1 to 1)
        self.vel_x = left_stick_x * self.speed
        if abs(left_stick_x) > 0.1:
            self.facing_right = left_stick_x > 0

        # Apply gravity
        self.vel_y += self.gravity

        # Jump (A button)
        if a_pressed and self.on_ground and self.jump_cooldown <= 0:
            self.vel_y = self.jump_power
            self.on_ground = False
            self.jump_cooldown = 15

        # Dash (B button)
        if b_pressed and self.dash_cooldown <= 0:
            dash_direction = 1 if self.facing_right else -1
            self.vel_x = dash_direction * 30
            self.dash_cooldown = 30

        # Apply velocity
        self.x += self.vel_x
        self.y += self.vel_y

        # Boundary collision
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
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

    def draw(self, surface):
        """Draw the player."""
        rect = pygame.Rect(int(self.x), int(self.y), self.width, self.height)

        # Draw player body
        pygame.draw.rect(surface, PLAYER_COLOR, rect)
        pygame.draw.rect(surface, PLAYER_OUTLINE, rect, 3)

        # Draw direction indicator
        eye_offset = self.width // 4
        eye_x = self.x + self.width // 2 + (eye_offset if self.facing_right else -eye_offset)
        pygame.draw.circle(surface, PLAYER_OUTLINE, (int(eye_x), int(self.y + 20)), 8)
        pygame.draw.circle(surface, (0, 0, 0), (int(eye_x), int(self.y + 20)), 4)


class Collectible:
    def __init__(self):
        self.respawn()

    def respawn(self):
        """Spawn at random position."""
        self.x = random.randint(50, WINDOW_WIDTH - 50)
        self.y = random.randint(100, WINDOW_HEIGHT - 100)
        self.radius = 20
        self.angle = 0

    def update(self):
        """Animate collectible."""
        self.angle += 0.1

    def draw(self, surface):
        """Draw the collectible with pulsing effect."""
        pulse = 1 + 0.2 * (1 + self.angle * 0.5) % 1
        radius = int(self.radius * pulse)
        pygame.draw.circle(surface, COLLECTIBLE_COLOR, (int(self.x), int(self.y)), radius)
        pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y)), radius // 2)


class Particle:
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.vel_x = direction * random.randint(5, 15)
        self.vel_y = random.randint(-5, 5)
        self.life = 30
        self.size = random.randint(5, 15)

    def update(self):
        self.x += self.vel_x
        self.y += self.vel_y
        self.vel_y += 0.3
        self.life -= 1

    def draw(self, surface):
        alpha = int((self.life / 30) * 255)
        color = (*PARTICLE_COLOR, alpha)
        s = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, color, (self.size, self.size), self.size)
        surface.blit(s, (int(self.x) - self.size, int(self.y) - self.size))


class SimpleGame:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("NitroGen Test Game - Simple")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # Game objects
        self.player = Player(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 150)
        self.collectible = Collectible()
        self.particles = []

        # Game state
        self.score = 0
        self.running = True

        # Joystick setup
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Joystick detected: {self.joystick.get_name()}")
        else:
            print("No joystick detected. Using keyboard fallback.")
            print("Keyboard: WASD/Arrows to move, Space to jump, Shift to dash")

    def get_input(self):
        """Get input from gamepad or keyboard."""
        left_stick_x = 0
        left_stick_y = 0
        a_pressed = False  # Jump
        b_pressed = False  # Dash
        x_pressed = False  # Attack

        if self.joystick:
            # Get joystick axis (-1 to 1)
            left_stick_x = self.joystick.get_axis(0)
            left_stick_y = self.joystick.get_axis(1)

            # Get buttons (0 = A, 1 = B, 2 = X on Xbox)
            a_pressed = self.joystick.get_button(0)
            b_pressed = self.joystick.get_button(1)
            x_pressed = self.joystick.get_button(2)
        else:
            # Keyboard fallback
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                left_stick_x = -1
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                left_stick_x = 1
            a_pressed = keys[pygame.K_SPACE]
            b_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            x_pressed = keys[pygame.K_z]

        return left_stick_x, left_stick_y, a_pressed, b_pressed, x_pressed

    def check_collisions(self):
        """Check player-collectible collision."""
        player_center_x = self.player.x + self.player.width / 2
        player_center_y = self.player.y + self.player.height / 2
        distance = ((player_center_x - self.collectible.x) ** 2 +
                   (player_center_y - self.collectible.y) ** 2) ** 0.5

        if distance < self.player.width / 2 + self.collectible.radius:
            self.score += 10
            self.collectible.respawn()
            return True
        return False

    def spawn_attack_particles(self):
        """Spawn particles when attacking."""
        direction = 1 if self.player.facing_right else -1
        start_x = self.player.x + (self.player.width if direction > 0 else 0)
        start_y = self.player.y + self.player.height / 2

        for _ in range(5):
            self.particles.append(Particle(start_x, start_y, direction))

    def draw_hud(self):
        """Draw the heads-up display."""
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, TEXT_COLOR)
        self.screen.blit(score_text, (20, 20))

        # Controls hint
        hint_text = self.font.render("Left Stick: Move | A: Jump | B: Dash | X: Attack",
                                      True, (150, 150, 150))
        hint_rect = hint_text.get_rect(center=(WINDOW_WIDTH // 2, 30))
        self.screen.blit(hint_text, hint_rect)

        # Cooldown indicators
        y_pos = WINDOW_HEIGHT - 40
        if self.player.dash_cooldown > 0:
            dash_text = self.font.render(f"Dash CD: {self.player.dash_cooldown}", True, (200, 100, 100))
            self.screen.blit(dash_text, (20, y_pos))

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

            # Get input
            left_stick_x, left_stick_y, a_pressed, b_pressed, x_pressed = self.get_input()

            # Update game state
            self.player.update(left_stick_x, left_stick_y, a_pressed, b_pressed, x_pressed)

            # Attack particles
            if x_pressed and self.player.attack_cooldown <= 0:
                self.spawn_attack_particles()
                self.player.attack_cooldown = 10

            self.collectible.update()
            self.check_collisions()

            # Update particles
            self.particles = [p for p in self.particles if p.life > 0]
            for p in self.particles:
                p.update()

            # Draw
            self.screen.fill(BACKGROUND)

            # Draw platforms (visual only)
            pygame.draw.rect(self.screen, (50, 50, 80), (0, WINDOW_HEIGHT - 20, WINDOW_WIDTH, 20))

            self.collectible.draw(self.screen)
            for p in self.particles:
                p.draw(self.screen)
            self.player.draw(self.screen)
            self.draw_hud()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


def main():
    """Run the game."""
    print("=" * 60)
    print("NitroGen Test Game - Simple")
    print("=" * 60)
    print("A simple test game for the NitroGen agent framework.")
    print("")
    print("Controls:")
    print("  - Left Joystick: Move player")
    print("  - A Button (South): Jump")
    print("  - B Button (East): Dash")
    print("  - X Button (West): Attack")
    print("  - ESC: Quit")
    print("=" * 60)

    game = SimpleGame()
    game.run()


if __name__ == "__main__":
    main()
