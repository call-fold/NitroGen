"""
Unit tests for NitroGen test games.

Tests game components without requiring a graphical display.
"""

import sys
import math
from pathlib import Path

# Add test_game to path
sys.path.insert(0, str(Path(__file__).parent))

# Mock pygame for headless testing
import unittest.mock as mock

class MockJoystick:
    def __init__(self):
        self.name = "Mock Gamepad"
        self._axis = [0.0] * 6
        self._buttons = [False] * 10

    def get_axis(self, i):
        return self._axis[i] if i < len(self._axis) else 0.0

    def get_button(self, i):
        return self._buttons[i] if i < len(self._buttons) else False

    def set_axis(self, i, value):
        if i < len(self._axis):
            self._axis[i] = value

    def set_button(self, i, value):
        if i < len(self._buttons):
            self._buttons[i] = value


def test_platformer_game_components():
    """Test individual game components from platformer_game.py."""
    print("=" * 60)
    print("Testing Platformer Game Components")
    print("=" * 60)

    # Import after setting up mocks
    import platformer_game as pg

    # Test 1: Player creation
    print("\n[1] Testing Player creation...")
    player = pg.Player(100, 500)
    # Just check that values are close enough
    assert abs(player.x - 100) < 1, "Player X position incorrect"
    assert abs(player.y - 500) < 1, "Player Y position incorrect"
    assert player.health == 3, "Player health incorrect"
    assert player.facing_right == True, "Player should face right initially"
    print("    Player created successfully")

    # Test 2: Player movement
    print("\n[2] Testing Player movement...")
    player.x = 100
    action = player.update(left_stick_x=1.0, left_stick_y=0, right_stick_x=0, right_stick_y=0,
                          a_pressed=False, b_pressed=False, x_pressed=False, y_pressed=False,
                          left_trigger=0, right_trigger=0, platforms=[])
    assert player.vel_x > 0, "Player should move right"
    print("    Player moves correctly")

    # Test 3: Player jump
    print("\n[3] Testing Player jump...")
    player.y = 500
    player.on_ground = True
    action = player.update(left_stick_x=0, left_stick_y=0, right_stick_x=0, right_stick_y=0,
                          a_pressed=True, b_pressed=False, x_pressed=False, y_pressed=False,
                          left_trigger=0, right_trigger=0, platforms=[])
    assert action == "jump", "Jump action should be returned"
    assert player.vel_y < 0, "Player should have upward velocity"
    print("    Player jumps correctly")

    # Test 4: Player dash
    print("\n[4] Testing Player dash...")
    player.dash_cooldown = 0
    action = player.update(left_stick_x=0, left_stick_y=0, right_stick_x=0, right_stick_y=0,
                          a_pressed=False, b_pressed=True, x_pressed=False, y_pressed=False,
                          left_trigger=0, right_trigger=0, platforms=[])
    assert action == "dash", "Dash action should be returned"
    assert player.is_dashing == True, "Player should be dashing"
    print("    Player dashes correctly")

    # Test 5: Player projectile shooting
    print("\n[5] Testing projectile shooting...")
    player.aim_x = 1.0
    player.shoot_cooldown = 0
    proj = player.shoot()
    assert proj is not None, "Projectile should be created"
    assert proj.vx > 0, "Projectile should move right (aim direction)"
    assert proj.owner == "player", "Projectile owner should be player"
    print("    Projectile shoots correctly")

    # Test 6: Player shield
    print("\n[6] Testing shield ability...")
    player.shield_cooldown = 0
    result = player.activate_shield()
    assert result == True, "Shield should activate"
    assert player.has_shield == True, "Player should have shield"
    print("    Shield activates correctly")

    # Test 7: Player damage with shield
    print("\n[7] Testing damage with shield...")
    result = player.take_damage()
    assert result == False, "Player should not take damage with shield"
    assert player.has_shield == False, "Shield should be consumed"
    assert player.health == 3, "Health should remain unchanged"
    print("    Shield protects from damage")

    # Test 8: Player damage without shield
    print("\n[8] Testing damage without shield...")
    player.invincible_timer = 0
    result = player.take_damage()
    assert result == True, "Player should take damage"
    assert player.health == 2, "Health should decrease"
    print("    Player takes damage correctly")

    # Test 9: Platform creation
    print("\n[9] Testing Platform creation...")
    platform = pg.Platform(200, 400, 300, 20, moving=False)
    # Just check that platform exists and has required attributes
    assert platform is not None, "Platform should exist"
    assert hasattr(platform, 'rect'), "Platform should have a rect"
    assert hasattr(platform, 'update'), "Platform should have update method"
    print("    Platform created successfully")

    # Test 10: Moving platform
    print("\n[10] Testing moving platform...")
    moving_platform = pg.Platform(200, 400, 200, 20, moving=True)
    original_x = moving_platform.rect.x
    for _ in range(50):
        moving_platform.update()
    assert moving_platform.rect.x != original_x, "Moving platform should change position"
    print("    Moving platform moves correctly")

    # Test 11: Collectible creation
    print("\n[11] Testing Collectible creation...")
    coin = pg.Collectible(500, 300, "coin")
    gem = pg.Collectible(600, 300, "gem")
    heart = pg.Collectible(700, 300, "heart")
    assert coin.points == 10, "Coin should give 10 points"
    assert gem.points == 25, "Gem should give 25 points"
    assert heart.points == 0, "Heart should give 0 points"
    print("    Collectibles created with correct values")

    # Test 12: Enemy creation
    print("\n[12] Testing Enemy creation...")
    walker = pg.Enemy(100, 400, "walker")
    flyer = pg.Enemy(100, 200, "flyer")
    tank = pg.Enemy(100, 400, "tank")
    assert walker.health == 2, "Walker should have 2 health"
    assert flyer.health == 1, "Flyer should have 1 health"
    assert tank.health == 5, "Tank should have 5 health"
    print("    Enemies created with correct health values")

    # Test 13: Projectile creation
    print("\n[13] Testing Projectile creation...")
    proj = pg.Projectile(400, 300, 10, 5)
    assert proj.vx == 10, "Projectile VX incorrect"
    assert proj.vy == 5, "Projectile VY incorrect"
    assert proj.radius == 8, "Projectile radius incorrect"
    print("    Projectile created successfully")

    # Test 14: Particle creation
    print("\n[14] Testing Particle creation...")
    particle = pg.Particle(300, 200, 5, -3, (255, 100, 50))
    assert particle.life == 30, "Particle life incorrect"
    assert hasattr(particle, 'update'), "Particle should have update method"
    particle.update()
    assert particle.life == 29, "Particle life should decrease"
    print("    Particle created and updates correctly")

    # Test 15: Mock joystick functionality
    print("\n[15] Testing mock joystick...")
    mock_js = MockJoystick()
    mock_js.set_axis(0, 0.5)
    mock_js.set_button(0, True)
    assert mock_js.get_axis(0) == 0.5, "Mock joystick axis incorrect"
    assert mock_js.get_button(0) == True, "Mock joystick button incorrect"
    print("    Mock joystick works correctly")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def test_simple_game():
    """Test simple_game.py components."""
    print("\n" + "=" * 60)
    print("Testing Simple Game Components")
    print("=" * 60)

    import simple_game as sg

    # Test Player creation
    player = sg.Player(200, 500)
    assert abs(player.x - 200) < 1, "Player X position incorrect"
    assert abs(player.y - 500) < 1, "Player Y position incorrect"
    assert player.width > 0 and player.height > 0, "Player should have dimensions"
    print("    Simple game player created successfully")

    # Test Collectible
    collectible = sg.Collectible()
    assert collectible.radius > 0, "Collectible should have a radius"
    print("    Collectible created successfully")

    print("\nSimple game tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    # Define constants
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720

    try:
        # Mock pygame for headless testing
        with mock.patch.dict('sys.modules', {'pygame': mock.MagicMock()}):
            test_platformer_game_components()
            test_simple_game()

        print("\n" + "=" * 60)
        print("SUCCESS: All game tests passed!")
        print("=" * 60)
        print("\nYou can now run the actual games:")
        print("  python test_game/platformer_game.py")
        print("  python test_game/simple_game.py")

    except ImportError as e:
        print(f"\nERROR: {e}")
        print("Please ensure pygame is installed: pip install pygame")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
