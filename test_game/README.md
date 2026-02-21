# Test Games for NitroGen

Simple test games to verify NitroGen agent functionality.

## Simple Game

A basic Pygame-based game with:
- A controllable player character (green square)
- Collectible items (yellow circles)
- Particle effects when attacking
- Score tracking

### Controls

| Input | Action |
|-------|--------|
| Left Joystick | Move player |
| A Button (South) | Jump |
| B Button (East) | Dash |
| X Button (West) | Attack |
| ESC | Quit |

### Keyboard Fallback (if no gamepad connected)

| Key | Action |
|-----|--------|
| WASD / Arrow Keys | Move |
| Space | Jump |
| Shift | Dash |
| Z | Attack |

## Platformer Game ⭐ Recommended

A more complete 2D platformer designed specifically for NitroGen testing:

### Features
- Multiple enemy types (Walker, Flyer, Tank)
- Moving platforms
- Various collectibles (Coins, Gems, Hearts)
- Player abilities (Jump, Dash, Shoot, Shield)
- Progressive difficulty
- Rich visual effects (particles, glows, animations)

### Controls

| Gamepad Input | Action |
|---------------|--------|
| Left Joystick | Move player |
| Right Joystick | Aim cursor |
| A Button (South) | Jump |
| B Button (East) | Dash |
| X Button (West) | Shoot projectile |
| Y Button (North) | Shield ability |
| Start Button | Pause game |
| Triggers | Fine movement control |

| Keyboard Key | Action |
|--------------|--------|
| WASD / Arrows | Move |
| IJKL | Aim |
| Space | Jump |
| Shift | Dash |
| Z / X | Shoot |
| C | Shield |
| ESC | Quit |

## Running the Test Games

```bash
# Simple Game
python test_game/simple_game.py

# Platformer Game (Recommended)
python test_game/platformer_game.py

# Or as modules
python -m test_game.simple_game
python -m test_game.platformer_game
```

## Testing with NitroGen

### Method 1: Manual Testing (Recommended for Development)

1. Start the inference server:
   ```bash
   python scripts/serve.py <path_to_ng.pt> --port 5555 --cfg 1.0
   ```

2. Start the test game:
   ```bash
   python test_game/platformer_game.py
   ```

3. Play with a real gamepad or keyboard to collect gameplay footage for training.

### Method 2: Automatic Agent Control

To test NitroGen's automatic gameplay, you need to package the game as an executable:

```bash
# Install PyInstaller
pip install pyinstaller

# Package the platformer game
pyinstaller --onefile --windowed --name "NitroGenTestGame" test_game/platformer_game.py

# Run the packaged game
dist/NitroGenTestGame.exe

# In another terminal, start the inference server
python scripts/serve.py <path_to_ng.pt> --port 5555 --cfg 1.0

# In another terminal, run NitroGen to play the game
python scripts/play.py --process "NitroGenTestGame.exe" --port 5555
```

## Game Window Titles

- Simple Game: "NitroGen Test Game - Simple"
- Platformer Game: "NitroGen Platformer Test Game"

## Test Scenarios

| Scenario | Tests | Description |
|----------|-------|-------------|
| Basic Movement | Action prediction | Control character left/right |
| Platform Jumping | Timing judgment | Jump at correct timing to reach platforms |
| Collect Coins | Goal-directed behavior | Move towards coin targets |
| Avoid Enemies | Hazard awareness | Recognize and avoid red enemies |
| Defeat Enemies | Combat decision | Aim and shoot at enemies |
| Use Shield | Strategic planning | Activate shield when in danger |
| Dash Across | Dynamic response | Use dash to quickly cross danger zones |

## Troubleshooting

### Game window not captured by NitroGen
- Ensure game window is in foreground
- Check if process name in `play.py` matches exactly
- Use `psutil` or Task Manager to find the correct process name

### Gamepad not responding
- Ensure gamepad is properly connected
- Try reconnecting the gamepad
- Check if game detects the gamepad (look for console output)

### Game running slowly
- Reduce FPS setting in the game file
- Lower number of particles
- Close other resource-intensive applications

### Multiple Python processes
When running games via Python, `play.py` may capture the wrong process. Solution:
- Package game as EXE (recommended)
- Or run game with a unique script name and specify process name carefully
