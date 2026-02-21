# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NitroGen is an open foundation model for generalist gaming agents (NVIDIA research). It's a 500M parameter DiT (Diffusion Transformer) that takes pixel input and predicts gamepad actions via flow matching. Trained through behavior cloning on internet gameplay videos.

**Important constraint**: This is a Windows-only agent runtime. The inference server can run on Linux, but game interaction (`play.py`) requires Windows with virtual gamepad support.

## Setup & Commands

```bash
# Install (editable mode)
pip install -e .

# Install server-only deps (for Linux inference serving)
pip install -e ".[serve]"

# Install play-only deps (Windows)
pip install -e ".[play]"

# Start inference server (can be on Linux or Windows)
python scripts/serve.py <path_to_ng.pt> --port 5555 --cfg 1.0

# Run agent on a game (Windows only, server must be running)
python scripts/play.py --process "game.exe" --port 5555
```

There are no tests or linting configured in this project.

## Architecture

The system uses a **client-server architecture** with ZeroMQ for communication:

### Server Side (can run on Linux)
- `scripts/serve.py` — ZeroMQ REP server that loads model and handles predict/reset/info requests via pickle-serialized messages
- `nitrogen/inference_session.py` — Manages model loading, tokenization, frame buffering, and inference. Key class: `InferenceSession.from_ckpt()` loads checkpoint and prompts user to select a game from the tokenizer's game mapping
- `nitrogen/flow_matching_transformer/nitrogen.py` — The NitroGen model: SigLIP vision encoder → self-attention VL mixer → DiT diffusion model for action prediction
- `nitrogen/flow_matching_transformer/modules.py` — DiT and SelfAttentionTransformer building blocks
- `nitrogen/mm_tokenizers.py` — `NitrogenTokenizer` encodes frames/game-ID into model input and decodes model output back to joystick + button actions
- `nitrogen/cfg.py` — Pydantic config classes: `CkptConfig` (top-level), `ModalityConfig`, `NitroGen_Config`

### Client Side (Windows only)
- `scripts/play.py` — Main game loop: captures screen → sends to server → receives actions → sends gamepad inputs. Saves debug videos and action logs to `out/`
- `nitrogen/game_env.py` — `GamepadEnv` (Gymnasium env): screen capture via dxcam, virtual gamepad via vgamepad, game speed control via xspeedhack
- `nitrogen/inference_client.py` — `ModelClient`: ZeroMQ REQ client that sends images and receives predicted actions
- `nitrogen/inference_viz.py` — Debug visualization overlays and `VideoRecorder` (uses PyAV/ffmpeg)

### Data Flow
1. `GamepadEnv` captures game screen via dxcam
2. Frame resized to 256×256 and sent to inference server
3. Server processes through SigLIP → VL mixer → DiT flow matching
4. Returns joystick positions (j_left, j_right) and button probabilities
5. `play.py` converts predictions to gamepad inputs (button threshold: 0.5)
6. Actions repeated `action_downsample_ratio` times in the environment

### Key Constants
- `nitrogen/shared.py` — `BUTTON_ACTION_TOKENS` (21 gamepad buttons), `PATH_REPO`
- Images are processed at 256×256 resolution
- Model uses SigLIP (`google/siglip-large-patch16-256`) as vision encoder
- Default server port: 5555
