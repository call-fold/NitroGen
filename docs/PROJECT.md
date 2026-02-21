# NitroGen 项目文档

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 项目结构](#2-项目结构)
- [3. 架构设计](#3-架构设计)
- [4. 安装与配置](#4-安装与配置)
- [5. 使用指南](#5-使用指南)
- [6. 模块详解](#6-模块详解)
- [7. 关键常量与配置](#7-关键常量与配置)
- [8. 数据流详解](#8-数据流详解)
- [9. 许可证](#9-许可证)

---

## 1. 项目概述

NitroGen 是 NVIDIA 研究院开发的**通用游戏智能体开放基础模型**。它是一个 500M 参数的 DiT（Diffusion Transformer），接收像素输入并通过 flow matching 预测手柄动作。模型通过在互联网游戏视频上进行行为克隆（Behavior Cloning）训练而成。

### 核心特性

- **模型规模**：500M 参数
- **视觉编码器**：SigLIP (`google/siglip-large-patch16-256`)，输入分辨率 256x256
- **动作预测**：通过 Flow Matching 生成手柄摇杆位置 + 按键概率
- **多游戏支持**：通过 Game ID 条件化实现多游戏切换
- **Classifier-Free Guidance (CFG)**：支持条件/无条件引导采样

### 当前限制

- 仅能观察最后一帧画面（无长期规划能力）
- 不能端到端通关游戏
- 不能自我改进
- 对完全未见过的游戏效果有限
- 是一个快速反应的 System-1 感知模型

---

## 2. 项目结构

```
NitroGen/
├── nitrogen/                          # 核心 Python 包
│   ├── __init__.py
│   ├── shared.py                      # 全局常量 (按键 token 列表, 仓库路径)
│   ├── cfg.py                         # Pydantic 配置类 (CkptConfig, ModalityConfig)
│   ├── mm_tokenizers.py               # 多模态分词器 (NitrogenTokenizer)
│   ├── inference_session.py           # 推理会话管理 (模型加载、帧缓冲、推理)
│   ├── inference_client.py            # ZeroMQ 客户端 (ModelClient)
│   ├── inference_viz.py               # 调试可视化 + 视频录制 (VideoRecorder)
│   ├── game_env.py                    # 游戏环境 (GamepadEnv, GamepadEmulator)
│   └── flow_matching_transformer/     # 模型核心
│       ├── __init__.py
│       ├── nitrogen.py                # NitroGen 主模型类
│       └── modules.py                 # DiT / SelfAttentionTransformer 模块
├── scripts/
│   ├── serve.py                       # 推理服务器脚本
│   └── play.py                        # 游戏客户端脚本
├── pyproject.toml                     # 项目依赖与构建配置
├── LICENSE                            # NVIDIA 非商业许可证
└── README.md                          # 项目简介
```

---

## 3. 架构设计

NitroGen 采用**客户端-服务端架构**，通过 ZeroMQ 进行进程间通信。

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Windows 客户端 (play.py)                     │
│                                                                  │
│  ┌──────────┐    ┌───────────┐    ┌────────────┐                │
│  │ GamepadEnv│───→│preprocess │───→│ModelClient │                │
│  │ (dxcam   │    │ (256x256) │    │ (ZMQ REQ)  │                │
│  │  截屏)    │    └───────────┘    └──────┬─────┘                │
│  │          │                            │ tcp://host:5555       │
│  │  vgamepad│←── 动作转换 ←── 预测结果 ←──┘                      │
│  └──────────┘                                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                         ZeroMQ (TCP)
                              │
┌─────────────────────────────────────────────────────────────────┐
│                推理服务器 (serve.py, Linux/Windows)               │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              InferenceSession                         │       │
│  │                                                       │       │
│  │  ┌─────────┐   ┌──────────┐   ┌──────────────────┐  │       │
│  │  │ SigLIP  │──→│ VL Mixer │──→│ DiT Flow Matching│  │       │
│  │  │ Vision  │   │ (Self-   │   │ (Cross-Attention │  │       │
│  │  │ Encoder │   │ Attention│   │  + Euler 积分)    │  │       │
│  │  └─────────┘   │ Transf.) │   └────────┬─────────┘  │       │
│  │                └──────────┘            │             │       │
│  │  ┌──────────────┐              ┌──────┴───────┐     │       │
│  │  │ Tokenizer    │              │ Action Decode │     │       │
│  │  │ (encode/     │              │ → j_left      │     │       │
│  │  │  decode)     │              │ → j_right     │     │       │
│  │  └──────────────┘              │ → buttons     │     │       │
│  │                                └──────────────┘     │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### 模型内部结构

```
输入帧 (256x256 RGB)
    │
    ▼
SigLIP Vision Encoder (768-dim)
    │
    ▼
VL Self-Attention Transformer (混合视觉 + Game ID token)
    │
    ▼  (作为 cross-attention 的 encoder_hidden_states)
DiT (Diffusion Transformer)
    │  ← 噪声动作 + 时间步编码
    │  ← Euler ODE 积分 (N 步去噪)
    ▼
Action Decoder (MLP)
    │
    ▼
[j_left(2), j_right(2), buttons(21)]  共 25 维动作
```

---

## 4. 安装与配置

### 环境要求

- Python >= 3.10（推荐 3.12）
- 推理服务器：Linux 或 Windows，需要 NVIDIA GPU + CUDA
- 游戏客户端：仅 Windows 11

### 完整安装

```bash
git clone https://github.com/MineDojo/NitroGen.git
cd NitroGen
pip install -e .
```

### 仅安装服务端依赖（Linux 推理）

```bash
pip install -e ".[serve]"
```

核心依赖：`torch`, `transformers`, `diffusers`, `einops`, `pydantic`, `pyzmq`, `polars`

### 仅安装客户端依赖（Windows 游戏交互）

```bash
pip install -e ".[play]"
```

核心依赖：`dxcam`, `vgamepad`, `pywinctl`, `xspeedhack`, `pywin32`, `opencv-python`, `av`

### 下载模型权重

```bash
hf download nvidia/NitroGen ng.pt
```

---

## 5. 使用指南

### 启动推理服务器

```bash
python scripts/serve.py <checkpoint_path> [选项]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ckpt` | str | (必填) | 模型权重路径（如 `ng.pt`） |
| `--port` | int | 5555 | ZeroMQ 服务端口 |
| `--old-layout` | flag | False | 使用旧版动作布局 `[buttons, j_left, j_right]` |
| `--cfg` | float | 1.0 | Classifier-Free Guidance 强度（1.0 = 无 CFG） |
| `--ctx` | int | 1 | 上下文帧数 |

启动后，服务器会列出可用游戏列表，需要手动输入 Game ID 选择目标游戏。

### 运行游戏代理

```bash
python scripts/play.py [选项]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--process` | str | `celeste.exe` | 目标游戏进程名 |
| `--allow-menu` | flag | False | 允许菜单操作（GUIDE/START/BACK） |
| `--port` | int | 5555 | 推理服务器端口 |

客户端会自动：
1. 连接推理服务器并获取会话信息
2. 查找游戏窗口并初始化屏幕捕获
3. 创建虚拟手柄
4. 进入主循环：截屏 → 推理 → 执行动作
5. 录制调试视频和干净视频到 `out/<checkpoint_name>/` 目录
6. 记录动作日志到 JSONL 文件

### ZeroMQ 通信协议

服务端支持三种请求类型（pickle 序列化）：

| 请求类型 | 请求格式 | 响应格式 |
|----------|----------|----------|
| `predict` | `{"type": "predict", "image": np.ndarray}` | `{"status": "ok", "pred": {"j_left": ..., "j_right": ..., "buttons": ...}}` |
| `reset` | `{"type": "reset"}` | `{"status": "ok"}` |
| `info` | `{"type": "info"}` | `{"status": "ok", "info": {...}}` |

---

## 6. 模块详解

### 6.1 `nitrogen/shared.py`

全局共享常量。

- **`BUTTON_ACTION_TOKENS`**：21 个手柄按键名列表，定义了按键动作的标准顺序
  ```
  ['BACK', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP', 'EAST',
   'GUIDE', 'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER', 'NORTH',
   'RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT', 'RIGHT_SHOULDER',
   'RIGHT_THUMB', 'RIGHT_TRIGGER', 'RIGHT_UP', 'SOUTH', 'START', 'WEST']
  ```
- **`PATH_REPO`**：仓库根目录的绝对路径

### 6.2 `nitrogen/cfg.py`

基于 Pydantic 的配置类体系。

#### `ModalityConfig`

控制帧和动作的采样方式。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `frame_per_sample` | int | 1 | 每个样本的上下文帧数 |
| `frame_spacing` | int\|None | None | 帧间跳过数（默认等于 `action_per_chunk`） |
| `action_per_chunk` | int | 8 | 每个动作块的动作数 |
| `action_shift` | int | 1 | 帧与动作块之间的偏移 |
| `action_interleaving` | bool | False | 是否将动作块与上下文帧交错 |
| `token_set` | str | "new" | Token 集合版本 |

#### `CkptConfig`

顶级检查点配置，包含：
- `experiment_name: str` — 实验名称
- `model_cfg: NitroGen_Config` — 模型配置
- `tokenizer_cfg: NitrogenTokenizerConfig` — 分词器配置
- `modality_cfg: ModalityConfig` — 模态配置

### 6.3 `nitrogen/mm_tokenizers.py`

多模态分词器，将帧和动作编码为模型输入格式。

#### Token 类型 ID

| 常量 | 值 | 用途 |
|------|---|------|
| `_PAD_TOKEN` | 0 | 填充 |
| `_IMG_TOKEN` | 1 | 图像 token |
| `_LANG_TOKEN` | 2 | 语言 token（未使用） |
| `_PROPRIO_TOKEN` | 3 | 本体感知 token（未使用） |
| `_ACT_TOKEN` | 4 | 动作 token |
| `_IMG_SEP_TOKEN` | 5 | 图像分隔符 |
| `_GAME_ID_TOKEN` | 6 | 游戏 ID token |

#### `NitrogenTokenizerConfig`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_visual_tokens_per_frame` | int | 256 | 每帧视觉 token 数 |
| `max_action_dim` | int | 25 | 最大动作维度 |
| `max_sequence_length` | int | 300 | VL 序列最大长度 |
| `action_horizon` | int | 16 | 动作预测步数 |
| `old_layout` | bool | False | 旧版动作布局 |
| `game_mapping_cfg` | GameMappingConfig\|None | None | 游戏映射配置 |

#### `NitrogenTokenizer`

**核心方法**：

- **`encode(data: dict) -> dict`**：将帧、游戏信息编码为模型输入
  - 输入 key：`frames`, `dropped_frames`, `game`（推理时）；训练时额外需要 `buttons`, `j_left`, `j_right`
  - 输出 key：`vl_token_ids`, `sa_token_ids`, `vl_attn_mask`, `images`, `game_ids` 等
- **`decode(data: dict) -> dict`**：将模型输出的 `action_tensor` 解码为 `j_left`, `j_right`, `buttons`
- **`pack_actions(buttons, j_left, j_right)`**：将分离的动作打包为单一张量，摇杆归一化到 [0,1]
- **`unpack_actions(actions)`**：反向操作，摇杆反归一化到 [-1,1]，按键按 0.5 阈值二值化

### 6.4 `nitrogen/inference_session.py`

管理模型加载和推理状态。

#### `load_model(checkpoint_path: str)`

从 `.pt` 检查点文件加载模型，返回 `(model, tokenizer, img_proc, ckpt_config, game_mapping, action_downsample_ratio)`。

#### `InferenceSession`

**构造方法**：

- **`from_ckpt(checkpoint_path, old_layout=False, cfg_scale=1.0, context_length=None)`**：从检查点创建会话。会交互式提示用户选择游戏。

**核心方法**：

- **`predict(obs) -> dict`**：接收 PIL Image，返回 `{"j_left": ndarray, "j_right": ndarray, "buttons": ndarray}`
  - 内部维护 `obs_buffer`（帧历史）和 `action_buffer`（动作历史）
  - 根据 `cfg_scale` 决定使用 `get_action` 或 `get_action_with_cfg`
- **`reset()`**：清空帧和动作缓冲区
- **`info() -> dict`**：返回会话元信息（检查点路径、游戏名、CFG 强度等）

### 6.5 `nitrogen/inference_client.py`

ZeroMQ REQ 客户端封装。

#### `ModelClient`

```python
client = ModelClient(host="localhost", port=5555)
```

| 方法 | 说明 |
|------|------|
| `predict(image: np.ndarray) -> dict` | 发送 RGB 图像，接收动作预测 |
| `reset()` | 重置服务端会话 |
| `info() -> dict` | 获取会话信息 |
| `close()` | 关闭连接 |

支持 `with` 上下文管理器。接收超时为 30 秒。

### 6.6 `nitrogen/inference_viz.py`

调试可视化和视频录制工具。

#### `create_viz(frame, i, j_left, j_right, buttons, token_set)`

在游戏帧右侧绘制摇杆位置和按键状态的可视化面板。返回拼接后的 numpy 数组。

#### `VideoRecorder`

基于 PyAV (ffmpeg) 的视频录制器。

```python
with VideoRecorder("output.mp4", fps=30, crf=28, preset="fast") as recorder:
    recorder.add_frame(rgb_numpy_array)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_file` | str | — | 输出文件路径 |
| `fps` | int | 30 | 帧率 |
| `crf` | int | 28 | 压缩质量 (0-51, 越高文件越小) |
| `preset` | str | "fast" | 编码速度预设 |

### 6.7 `nitrogen/game_env.py`

Windows 游戏环境封装，基于 Gymnasium。

#### `GamepadEmulator`

虚拟手柄封装，支持 Xbox 和 PS4 控制器类型。

| 方法 | 说明 |
|------|------|
| `step(action: dict)` | 执行一组动作（按键 + 摇杆 + 扳机） |
| `press_button(button)` / `release_button(button)` | 按下/释放按键 |
| `set_trigger(trigger, value)` | 设置扳机值 (0-1) |
| `set_joystick(joystick, value)` | 设置摇杆轴位置 |
| `wakeup(duration=0.1)` | 唤醒控制器 |
| `reset()` | 重置手柄状态 |

#### `GamepadEnv(Env)`

Gymnasium 环境，封装了屏幕捕获、手柄控制和游戏速度控制。

**构造参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `game` | str | — | 游戏进程名 (如 `"celeste.exe"`) |
| `image_height` | int | 1440 | 观测空间高度 |
| `image_width` | int | 2560 | 观测空间宽度 |
| `controller_type` | str | "xbox" | 控制器类型 (`"xbox"` / `"ps4"`) |
| `game_speed` | float | 1.0 | 游戏速度倍数 |
| `env_fps` | int | 10 | 环境帧率 |
| `async_mode` | bool | True | 是否在每步暂停/恢复游戏 |
| `screenshot_backend` | str | "dxcam" | 截屏后端 (`"dxcam"` / `"pyautogui"`) |

**核心方法**：

| 方法 | 说明 |
|------|------|
| `step(action, step_duration=None)` | 执行动作，返回 `(obs, reward, terminated, truncated, info)` |
| `reset(seed=None, options=None)` | 重置环境 |
| `render()` | 截屏并返回 PIL Image |
| `pause()` / `unpause()` | 通过 xspeedhack 暂停/恢复游戏 |

**截屏后端**：
- `DxcamScreenshotBackend`：基于 DirectX 的高性能截屏（推荐）
- `PyautoguiScreenshotBackend`：基于 pyautogui 的备选截屏

#### `get_process_info(process_name)`

根据进程名获取 PID、窗口标题和架构（x86/x64），用于定位游戏窗口和初始化速度控制。

### 6.8 `nitrogen/flow_matching_transformer/nitrogen.py`

NitroGen 模型主类。

#### `NitroGen_Config`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_size` | int | 1024 | 隐藏层维度 |
| `vision_encoder_name` | str | `"google/siglip-large-patch16-256"` | 视觉编码器 |
| `vision_hidden_size` | int | 768 | SigLIP 输出维度 |
| `action_dim` | int | None | 动作维度 |
| `action_horizon` | int | None | 动作预测步数 |
| `num_inference_timesteps` | int | None | 推理去噪步数 |
| `num_timestep_buckets` | int | 1000 | 时间步离散化桶数 |
| `noise_beta_alpha` | float | 1.5 | 噪声 Beta 分布 alpha |
| `noise_beta_beta` | float | 1.0 | 噪声 Beta 分布 beta |
| `noise_s` | float | 0.999 | Flow matching 噪声参数 s |
| `diffusion_model_cfg` | DiTConfig | — | DiT 配置 |
| `vl_self_attention_cfg` | SelfAttentionTransformerConfig | — | VL 混合器配置 |

#### `NitroGen(nn.Module)`

**子模块**：
- `vision_encoder`：SigLIP 视觉编码器
- `vl_self_attention_model`：视觉-语言自注意力混合器
- `model`：DiT 扩散模型
- `action_encoder`：`MultiEmbodimentActionEncoder`，将噪声动作 + 时间步编码为隐状态
- `action_decoder`：`CategorySpecificMLP`，将隐状态解码为动作
- `game_embedding`：游戏 ID 嵌入表（可选）

**核心方法**：

- **`forward(data) -> {"loss": Tensor}`**：训练前向传播
  1. 编码图像 → SigLIP 特征
  2. 对真实动作加噪：`noisy = (1-t)*noise + t*action`
  3. 通过 action_encoder + DiT 预测速度场
  4. 计算 MSE loss：`pred_velocity` vs `true_velocity`

- **`get_action(data) -> {"action_tensor": Tensor}`**：推理（无 CFG）
  1. 从标准正态分布采样噪声动作
  2. Euler ODE 积分 N 步：`x(t+dt) = x(t) + dt * velocity`

- **`get_action_with_cfg(data_cond, data_uncond, cfg_scale) -> {"action_tensor": Tensor}`**：带 CFG 的推理
  - `velocity = v_cond + cfg_scale * (v_cond - v_uncond)`

#### 辅助类

- **`SinusoidalPositionalEncoding`**：正弦位置编码
- **`CategorySpecificLinear` / `CategorySpecificMLP`**：按 embodiment 类别选择不同权重的线性层/MLP
- **`MultiEmbodimentActionEncoder`**：动作编码器，融合动作嵌入 + 时间步正弦编码

### 6.9 `nitrogen/flow_matching_transformer/modules.py`

Transformer 基础模块，基于 HuggingFace `diffusers` 库构建。

#### `DiTConfig`

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `num_attention_heads` | 8 | 注意力头数 |
| `attention_head_dim` | 64 | 每头维度 |
| `num_layers` | 12 | Transformer 层数 |
| `dropout` | 0.1 | Dropout 率 |
| `norm_type` | "ada_norm" | 归一化类型（自适应层归一化） |
| `cross_attention_dim` | None | 交叉注意力维度 |
| `positional_embeddings` | "sinusoidal" | 位置编码类型 |
| `interleave_self_attention` | False | 是否交替自注意力和交叉注意力层 |

#### `DiT(ModelMixin)`

Diffusion Transformer，`inner_dim = num_attention_heads * attention_head_dim`。

- 包含 `TimestepEncoder`（时间步编码）
- N 层 `BasicTransformerBlock`
- 输出层：AdaLN → 投影

**前向传播**：接收 `hidden_states`（动作 token）、`encoder_hidden_states`（VL token）、`timestep`。

#### `SelfAttentionTransformer(ModelMixin)`

纯自注意力 Transformer，用作 VL 混合器。结构类似 DiT 但无交叉注意力和时间步条件。

#### `BasicTransformerBlock`

单个 Transformer 块：
1. AdaLayerNorm（或标准 LayerNorm）
2. Cross-Attention（可选，通过 `cross_attention_dim` 控制）
3. LayerNorm + FeedForward (GEGLU)

#### `AdaLayerNorm`

自适应层归一化，接收时间步嵌入 `temb`，输出 `norm(x) * (1 + scale) + shift`。

### 6.10 `scripts/serve.py`

推理服务器入口脚本。

**流程**：
1. 解析命令行参数
2. 调用 `InferenceSession.from_ckpt()` 加载模型
3. 创建 ZeroMQ REP socket，绑定到指定端口
4. 主循环中 poll 请求，根据 `type` 字段分发到 `predict` / `reset` / `info` 处理器
5. 使用 pickle 序列化/反序列化通信数据

### 6.11 `scripts/play.py`

游戏客户端入口脚本。

**流程**：
1. 连接推理服务器，获取 `action_downsample_ratio`
2. 初始化 `GamepadEnv`（查找游戏窗口、创建虚拟手柄、初始化截屏）
3. 特殊游戏初始化（如 `isaac-ng.exe`、`Cuphead.exe` 需要额外按键）
4. 主循环：
   - 截屏 → 缩放到 256x256 → 发送到服务器
   - 接收预测：`j_left`, `j_right`, `buttons`
   - 将预测转换为手柄动作（摇杆值映射到 [-32767, 32767]，按键按 0.5 阈值二值化）
   - 安全处理：默认禁止 GUIDE/START/BACK（防止误触菜单）
   - 每个动作重复 `action_downsample_ratio` 次
   - 录制调试视频（含可视化）和干净视频（1080p）
   - 动作日志写入 JSONL 文件

**输出文件**（保存到 `out/<checkpoint_name>/`）：
- `XXXX_DEBUG.mp4`：带动作可视化的调试视频
- `XXXX_CLEAN.mp4`：干净的 1080p 游戏录像
- `XXXX_ACTIONS.json`：JSONL 格式的逐帧动作日志

---

## 7. 关键常量与配置

### 按键 Token 映射

模型输出的 `buttons` 维度为 21，按 `BUTTON_ACTION_TOKENS` 的顺序排列：

| 索引 | Token 名 | 对应 Xbox 按键 |
|------|----------|----------------|
| 0 | BACK | 返回键 |
| 1 | DPAD_DOWN | 十字键下 |
| 2 | DPAD_LEFT | 十字键左 |
| 3 | DPAD_RIGHT | 十字键右 |
| 4 | DPAD_UP | 十字键上 |
| 5 | EAST | B 键 |
| 6 | GUIDE | Xbox 键 |
| 7 | LEFT_SHOULDER | LB |
| 8 | LEFT_THUMB | 左摇杆按下 |
| 9 | LEFT_TRIGGER | LT |
| 10 | NORTH | Y 键 |
| 11 | RIGHT_BOTTOM | — |
| 12 | RIGHT_LEFT | — |
| 13 | RIGHT_RIGHT | — |
| 14 | RIGHT_SHOULDER | RB |
| 15 | RIGHT_THUMB | 右摇杆按下 |
| 16 | RIGHT_TRIGGER | RT |
| 17 | RIGHT_UP | — |
| 18 | SOUTH | A 键 |
| 19 | START | 开始键 |
| 20 | WEST | X 键 |

### 模型默认配置

| 配置项 | 值 |
|--------|-----|
| 输入分辨率 | 256 x 256 |
| 视觉编码器 | `google/siglip-large-patch16-256` |
| 每帧视觉 token 数 | 256 |
| 动作维度 | 25 (4 摇杆 + 21 按键) |
| 动作布局（新版） | `[j_left(2), j_right(2), buttons(21)]` |
| 动作布局（旧版） | `[buttons(21), j_left(2), j_right(2)]` |
| 按键激活阈值 | 0.5 |
| 摇杆归一化范围 | 模型内部 [0,1]，输出 [-1,1] |
| 默认服务端口 | 5555 |
| 客户端超时 | 30000 ms |
| 时间步离散化桶数 | 1000 |

---

## 8. 数据流详解

### 完整推理流程（逐步）

```
1. 屏幕捕获
   GamepadEnv.render() → dxcam 截屏 → PIL Image (游戏原始分辨率)

2. 图像预处理 (play.py)
   PIL Image → cv2.resize(256, 256) → PIL Image (RGB)

3. 客户端发送 (ModelClient.predict)
   PIL Image → numpy array → pickle 序列化 → ZeroMQ 发送

4. 服务端接收 (serve.py)
   ZeroMQ 接收 → pickle 反序列化 → numpy array

5. 图像处理 (InferenceSession.predict)
   numpy array → AutoImageProcessor (SigLIP 预处理) → pixel_values tensor
   → 添加到 obs_buffer (deque, maxlen=context_length)

6. 帧缓冲组装
   obs_buffer 中所有帧拼接 → frames tensor [context_length, C, H, W]
   不足部分用零填充，标记为 dropped_frames

7. 分词 (NitrogenTokenizer.encode)
   frames + game_name → vl_token_ids, sa_token_ids, vl_attn_mask, game_ids
   VL token 序列: [GAME_ID] [IMG x 256] ... (左侧零填充到 max_sequence_length)
   SA token 序列: [ACT x action_horizon]

8. 视觉编码 (NitroGen.encode_images)
   pixel_values → SigLIP vision_model → [B, num_frames, 256, 768]

9. VL 嵌入组装 (NitroGen.prepare_input_embs)
   - 视觉特征填入 _IMG_TOKEN 位置
   - Game ID 嵌入填入 _GAME_ID_TOKEN 位置
   → vl_embs [B, max_seq_len, 768]

10. VL 混合 (SelfAttentionTransformer)
    vl_embs → N 层自注意力 → vl_embs (融合视觉 + 游戏条件)

11. Flow Matching 去噪循环 (get_action / get_action_with_cfg)
    初始化: x_0 ~ N(0, I), shape [B, action_horizon, action_dim]
    循环 N 步 (i = 0, 1, ..., N-1):
      a. t = i / N, 离散化 t_disc = int(t * 1000)
      b. action_encoder(x_t, t_disc) → action_features [B, action_horizon, hidden_size]
      c. prepare_input_embs → sa_embs
      d. DiT(sa_embs, vl_embs, t_disc) → model_output
      e. action_decoder(model_output) → pred_velocity
      f. x_{t+1} = x_t + dt * pred_velocity   (Euler 步)
    最终输出: x_N [B, action_horizon, action_dim]

12. 动作解码 (NitrogenTokenizer.decode)
    action_tensor → unpack_actions →
      j_left  [B, action_horizon, 2]  范围 [-1, 1]
      j_right [B, action_horizon, 2]  范围 [-1, 1]
      buttons [B, action_horizon, 21] 二值 {0, 1}

13. 响应传回客户端
    numpy arrays → pickle → ZeroMQ → 客户端接收

14. 动作映射 (play.py)
    j_left/j_right: float [-1,1] → int [-32767, 32767] (摇杆)
    buttons: 按 BUTTON_ACTION_TOKENS 顺序映射到手柄按键
    TRIGGER 类: value * 255 → int (0-255)
    其他按键: > 0.5 → 1, 否则 → 0

15. 手柄执行 (GamepadEmulator.step)
    action dict → vgamepad API → 虚拟 Xbox/PS4 手柄输入
    通过 xspeedhack 控制游戏时间流逝

16. 重复
    每个预测包含 action_horizon 步动作，
    每步动作重复 action_downsample_ratio 次，
    执行完毕后截取新帧，回到步骤 1
```

---

## 9. 许可证

本项目使用 **NVIDIA 非商业许可证**。

### 要点

- **允许**：使用、复制、准备衍生作品、公开展示、分发（非商业用途）
- **限制**：仅限**非商业研究目的**，明确排除军事、监控、核技术服务和生物识别处理用途
- **再分发**：必须保留本许可证副本和所有版权/商标声明
- **衍生作品**：必须保留 3.3 节的使用限制
- **专利**：对 Licensor 提起专利诉讼将导致许可终止
- **商标**：不授予使用 NVIDIA 名称/标志的权利

完整许可证文本见项目根目录 `LICENSE` 文件。

> **声明**：此项目严格用于研究目的，不是 NVIDIA 官方产品。
