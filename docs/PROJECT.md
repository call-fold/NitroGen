# NitroGen 项目文档

## 1. 项目概述

NitroGen 是 NVIDIA 研究团队开发的通用游戏智能体开放基础模型。它是一个 **500M 参数的 DiT（Diffusion Transformer）**，通过行为克隆（Behavior Cloning）在互联网游戏视频数据集上训练而成。模型接收像素输入，通过 Flow Matching 预测手柄动作。

**核心特性：**
- 多游戏支持：单一模型可操作多款游戏
- 像素输入 → 手柄动作输出（摇杆 + 按键）
- 基于 SigLIP 视觉编码器 + DiT 扩散模型架构
- 支持 Classifier-Free Guidance (CFG) 增强推理

**当前限制：**
- 仅能看到最后一帧画面，无长期规划能力
- 不能端到端通关游戏
- 不能自我改进
- 对完全未见过的游戏效果有限
- 游戏交互仅支持 Windows 平台

---

## 2. 项目结构

```
NitroGen/
├── nitrogen/                            # 核心 Python 包
│   ├── __init__.py
│   ├── shared.py                        # 全局常量（按键列表、仓库路径）
│   ├── cfg.py                           # Pydantic 配置类
│   ├── mm_tokenizers.py                 # 多模态分词器（编码/解码动作与帧）
│   ├── inference_session.py             # 推理会话管理（模型加载、预测）
│   ├── inference_client.py              # ZeroMQ 客户端
│   ├── inference_viz.py                 # 调试可视化与视频录制
│   ├── game_env.py                      # Gymnasium 游戏环境（Windows 专用）
│   └── flow_matching_transformer/       # 模型核心
│       ├── nitrogen.py                  # NitroGen 模型（SigLIP + VL Mixer + DiT）
│       └── modules.py                   # DiT、Transformer 构建模块
├── scripts/
│   ├── serve.py                         # ZeroMQ 推理服务器
│   └── play.py                          # 游戏主循环（Windows 专用）
├── assets/
│   └── github_banner.gif
├── pyproject.toml                       # 项目配置与依赖
├── LICENSE                              # NVIDIA 非商业许可证
└── README.md
```

---

## 3. 架构设计

### 3.1 客户端-服务端架构

系统采用 ZeroMQ 进行通信，允许推理服务器与游戏客户端分离部署。

```
┌─────────────────────────────────┐     ZeroMQ (TCP)     ┌──────────────────────────────────┐
│       客户端 (Windows)           │◄───────────────────►│       服务端 (Linux/Windows)       │
│                                 │   pickle 序列化       │                                  │
│  play.py                        │                      │  serve.py                         │
│  ├─ GamepadEnv (屏幕捕获+手柄)   │   predict/reset/info │  ├─ InferenceSession              │
│  ├─ ModelClient (ZMQ REQ)       │──────────────────────│  │  ├─ NitroGen 模型 (CUDA)        │
│  └─ VideoRecorder (调试录像)     │                      │  │  ├─ NitrogenTokenizer           │
│                                 │                      │  │  └─ SigLIP 图像处理器            │
│  依赖: dxcam, vgamepad,         │                      │  └─ ZMQ REP Socket                │
│        xspeedhack, pywin32      │                      │                                  │
└─────────────────────────────────┘                      └──────────────────────────────────┘
```

### 3.2 模型架构

```
输入帧 (256x256 RGB)
        │
        ▼
┌───────────────────┐
│  SigLIP 视觉编码器  │  google/siglip-large-patch16-256
│  (768 维输出)       │  每帧产生 256 个视觉 token
└───────┬───────────┘
        │ + Game ID Embedding (可选)
        ▼
┌───────────────────┐
│  VL Self-Attention │  SelfAttentionTransformer
│  Mixer             │  视觉-语言特征混合
└───────┬───────────┘
        │ (encoder_hidden_states)
        ▼
┌───────────────────┐     ┌─────────────────────────┐
│  DiT 扩散模型      │◄────│  Action Encoder          │
│  (Cross-Attention) │     │  (噪声动作 + 时间步嵌入)   │
└───────┬───────────┘     └─────────────────────────┘
        │
        ▼
┌───────────────────┐
│  Action Decoder    │  CategorySpecificMLP
│  → j_left, j_right│  输出: 摇杆位置 + 按键概率
│  → buttons (21个)  │
└───────────────────┘
```

### 3.3 Flow Matching 推理流程

推理使用 Euler 积分从噪声中逐步去噪得到动作：

```
x(0) = 随机噪声
for i in [0, 1, ..., N-1]:
    t = i / N
    velocity = model(x(t), t, 视觉特征)
    x(t + dt) = x(t) + dt * velocity
最终动作 = x(1)
```

---

## 4. 安装与配置

### 4.1 环境要求

- Python >= 3.10
- CUDA GPU（推理服务器）
- Windows 11（游戏客户端，已在 Python >= 3.12 测试）

### 4.2 安装方式

```bash
# 完整安装（服务端 + 客户端所有依赖）
git clone https://github.com/MineDojo/NitroGen.git
cd NitroGen
pip install -e .

# 仅安装服务端依赖（可在 Linux 上运行）
pip install -e ".[serve]"

# 仅安装客户端依赖（Windows）
pip install -e ".[play]"
```

### 4.3 下载模型权重

```bash
hf download nvidia/NitroGen ng.pt
```

### 4.4 依赖说明

| 分类 | 依赖 | 用途 |
|------|------|------|
| 共享 | numpy, pyzmq | 数据处理、通信 |
| 服务端 | torch, transformers, diffusers, einops, pydantic, polars, pyyaml | 模型推理 |
| 客户端 | pillow, opencv-python, pyautogui, gymnasium, psutil, av | 图像处理、环境 |
| Windows 专用 | dxcam, pywinctl, vgamepad, pywin32, xspeedhack | 屏幕捕获、虚拟手柄、加速 |

---

## 5. 使用指南

### 5.1 启动推理服务器

```bash
python scripts/serve.py <checkpoint_path> [选项]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ckpt` | (必填) | 模型权重文件路径 |
| `--port` | 5555 | 服务端口 |
| `--old-layout` | False | 使用旧动作布局 `[buttons, j_left, j_right]` |
| `--cfg` | 1.0 | Classifier-Free Guidance 缩放系数（1.0 = 不使用 CFG） |
| `--ctx` | 1 | 上下文帧数 |

启动后服务器会提示选择游戏 ID（从检查点中的 game_mapping 列表选择）。

### 5.2 运行游戏智能体

```bash
python scripts/play.py [选项]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--process` | celeste.exe | 目标游戏进程名（需精确匹配） |
| `--allow-menu` | False | 是否允许菜单操作（START/BACK/GUIDE） |
| `--port` | 5555 | 推理服务器端口 |

**查找游戏进程名**：在 Windows 任务管理器中右键点击游戏进程 → 属性 → 常规选项卡中的 `.exe` 文件名。

### 5.3 输出文件

运行后在 `out/<checkpoint_name>/` 目录下生成：
- `XXXX_DEBUG.mp4` — 带摇杆/按键可视化叠加的调试视频
- `XXXX_CLEAN.mp4` — 纯游戏画面视频（1080p）
- `XXXX_ACTIONS.json` — JSONL 格式的动作日志

---

## 6. 模块详解

### 6.1 `nitrogen/shared.py` — 全局常量

```python
BUTTON_ACTION_TOKENS = [
    'BACK', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP',
    'EAST', 'GUIDE', 'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER',
    'NORTH', 'RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT',
    'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_TRIGGER', 'RIGHT_UP',
    'SOUTH', 'START', 'WEST'
]  # 21 个手柄按键名

PATH_REPO = Path(__file__).parent.parent.resolve()  # 仓库根目录
```

### 6.2 `nitrogen/cfg.py` — 配置类

**`ModalityConfig`** — 数据模态配置
| 字段 | 默认值 | 说明 |
|------|--------|------|
| `frame_per_sample` | 1 | 每个样本的上下文帧数 |
| `frame_spacing` | None | 帧间隔（None 时等于 action_per_chunk） |
| `action_per_chunk` | 8 | 每个动作块的动作数 |
| `action_shift` | 1 | 帧与动作块之间的偏移量 |
| `action_interleaving` | False | 是否交错动作块与上下文帧 |
| `token_set` | "new" | Token 集合类型 |

**`CkptConfig`** — 检查点顶层配置
| 字段 | 说明 |
|------|------|
| `experiment_name` | 实验名称 |
| `model_cfg` | NitroGen_Config 模型配置 |
| `tokenizer_cfg` | NitrogenTokenizerConfig 分词器配置 |
| `modality_cfg` | ModalityConfig 模态配置 |

### 6.3 `nitrogen/mm_tokenizers.py` — 多模态分词器

**Token 类型常量：**
| 常量 | 值 | 含义 |
|------|---|------|
| `_PAD_TOKEN` | 0 | 填充 |
| `_IMG_TOKEN` | 1 | 图像 |
| `_LANG_TOKEN` | 2 | 语言 |
| `_PROPRIO_TOKEN` | 3 | 本体感觉 |
| `_ACT_TOKEN` | 4 | 动作 |
| `_IMG_SEP_TOKEN` | 5 | 图像分隔符 |
| `_GAME_ID_TOKEN` | 6 | 游戏 ID |

**`NitrogenTokenizerConfig`** — 分词器配置
| 字段 | 默认值 | 说明 |
|------|--------|------|
| `num_visual_tokens_per_frame` | 256 | 每帧视觉 token 数 |
| `max_action_dim` | 25 | 最大动作维度 |
| `max_sequence_length` | 300 | 最大序列长度 |
| `action_horizon` | 16 | 动作预测步长 |
| `old_layout` | False | 旧动作布局（buttons 在前）vs 新布局（joystick 在前） |
| `game_mapping_cfg` | None | 游戏映射配置 |

**`NitrogenTokenizer`** — 核心分词器类

| 方法 | 说明 |
|------|------|
| `encode(data)` | 将帧和动作编码为模型输入格式，构建 token ID 和注意力掩码 |
| `decode(data)` | 将模型输出的动作张量解码为 j_left、j_right、buttons |
| `pack_actions(buttons, j_left, j_right)` | 将分离的动作分量打包为单一张量，摇杆归一化到 [0,1] |
| `unpack_actions(actions)` | 逆操作：拆分张量，摇杆反归一化到 [-1,1]，按键阈值化 |
| `train()` / `eval()` | 切换训练/评估模式 |

**动作布局：**
- 新布局（默认）：`[j_left(2), j_right(2), buttons(21)]` = 25 维
- 旧布局：`[buttons(21), j_left(2), j_right(2)]` = 25 维

### 6.4 `nitrogen/inference_session.py` — 推理会话

**`load_model(checkpoint_path)`** — 从检查点加载模型
1. 加载检查点文件（`torch.load`）
2. 解析 CkptConfig
3. 初始化 SigLIP 图像处理器
4. 创建 NitrogenTokenizer
5. 创建 NitroGen 模型并加载权重
6. 移至 CUDA

**`InferenceSession`** — 推理会话管理类

| 属性 | 说明 |
|------|------|
| `obs_buffer` | 观察帧缓冲（deque，大小 = context_length） |
| `action_buffer` | 动作历史缓冲（deque） |
| `cfg_scale` | CFG 缩放系数 |
| `max_buffer_size` | 最大缓冲帧数 |

| 方法 | 说明 |
|------|------|
| `from_ckpt(path, old_layout, cfg_scale, context_length)` | 类方法，从检查点创建会话，交互式选择游戏 |
| `predict(obs)` | 主推理入口：图像预处理 → 模型推理 → 返回动作字典 |
| `reset()` | 清空所有缓冲区 |
| `info()` | 返回会话配置信息 |

**推理流程（`_predict_flowmatching`）：**
1. 填充帧缓冲（不足部分标记为 dropped）
2. 构建条件输入（含历史帧）和无条件输入（仅当前帧）
3. 编码为张量并移至 CUDA
4. 若 `cfg_scale == 1.0`，调用 `model.get_action()`
5. 否则调用 `model.get_action_with_cfg()` 进行 CFG 推理
6. 解码动作并返回

### 6.5 `nitrogen/inference_client.py` — 推理客户端

**`ModelClient`** — ZeroMQ REQ 客户端

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `host` | "localhost" | 服务器地址 |
| `port` | 5555 | 服务器端口 |
| `timeout_ms` | 30000 | 接收超时（毫秒） |

| 方法 | 说明 |
|------|------|
| `predict(image)` | 发送 RGB 图像（numpy），接收动作预测 |
| `reset()` | 重置服务端会话 |
| `info()` | 获取会话信息 |
| `close()` | 关闭连接 |

通信协议使用 pickle 序列化，请求格式：
```python
{"type": "predict", "image": np.ndarray}  # 预测
{"type": "reset"}                          # 重置
{"type": "info"}                           # 查询
```

### 6.6 `nitrogen/inference_viz.py` — 可视化与录制

**`create_viz(frame, i, j_left, j_right, buttons, token_set)`**
- 将游戏帧与动作可视化拼接在一起
- 左侧：游戏画面
- 右侧：摇杆位置图 + 按键状态网格 + 按键图例

**`VideoRecorder`** — 基于 PyAV 的视频录制器
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `output_file` | (必填) | 输出文件路径 |
| `fps` | 30 | 帧率 |
| `crf` | 28 | 恒定质量因子（0-51，越高文件越小） |
| `preset` | "fast" | 编码预设 |

支持上下文管理器（`with` 语句）。

### 6.7 `nitrogen/game_env.py` — 游戏环境

**`get_process_info(process_name)`** — 获取 Windows 进程信息
- 返回 PID、窗口名称、架构（x86/x64）
- 自动过滤代理/辅助窗口

**`GamepadEmulator`** — 虚拟手柄控制
- 支持 Xbox 360 和 PS4 手柄模拟
- `XBOX_MAPPING` / `PS4_MAPPING`：统一按键名到平台特定名称的映射

**`GamepadEnv`** — Gymnasium 游戏环境（核心类）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `game` | (必填) | 游戏进程名 |
| `image_height` | 1440 | 观察图像高度 |
| `image_width` | 2560 | 观察图像宽度 |
| `controller_type` | "xbox" | 手柄类型 |
| `game_speed` | 1.0 | 游戏速度倍数 |
| `env_fps` | 10 | 环境帧率 |
| `async_mode` | True | 异步模式（步进时暂停/恢复游戏） |
| `screenshot_backend` | "dxcam" | 截图后端 |

| 方法 | 说明 |
|------|------|
| `step(action, step_duration)` | 执行动作，返回 (obs, reward, terminated, truncated, info) |
| `reset()` | 唤醒手柄并等待 |
| `render()` | 截取并调整游戏画面大小 |
| `pause()` / `unpause()` | 通过 xspeedhack 暂停/恢复游戏 |

**截图后端：**
- `DxcamScreenshotBackend` — 使用 dxcam 进行 GPU 加速截图（推荐）
- `PyautoguiScreenshotBackend` — 使用 pyautogui 截图（备选）

### 6.8 `nitrogen/flow_matching_transformer/nitrogen.py` — NitroGen 模型

**`NitroGen_Config`** — 模型配置

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 1024 | 隐藏层维度 |
| `vision_hidden_size` | 768 | SigLIP 输出维度 |
| `vision_encoder_name` | google/siglip-large-patch16-256 | 视觉编码器 |
| `action_dim` | None | 动作维度（从检查点加载） |
| `action_horizon` | None | 动作预测步长 |
| `num_inference_timesteps` | None | 推理去噪步数 |
| `num_timestep_buckets` | 1000 | 时间步离散化桶数 |
| `noise_beta_alpha` | 1.5 | Beta 分布 alpha |
| `noise_beta_beta` | 1.0 | Beta 分布 beta |
| `max_num_embodiments` | 1 | 最大机器人类型数 |
| `diffusion_model_cfg` | (必填) | DiT 配置 |
| `vl_self_attention_cfg` | (必填) | VL 自注意力配置 |
| `tune_*` | True | 各组件是否可训练 |

**辅助模块：**
- `SinusoidalPositionalEncoding` — 正弦位置编码
- `CategorySpecificLinear` — 按类别索引的线性层（支持多机器人）
- `CategorySpecificMLP` — 按类别索引的 MLP
- `MultiEmbodimentActionEncoder` — 动作编码器，融合动作嵌入与时间步嵌入

**`NitroGen`** — 主模型类

| 方法 | 说明 |
|------|------|
| `encode_images(images)` | 通过 SigLIP 提取视觉特征，形状 `(B, F, N, D)` |
| `prepare_input_embs(...)` | 根据 token ID 组装 VL 嵌入和 SA 嵌入 |
| `forward(data)` | 训练前向传播：添加噪声 → 预测速度 → MSE 损失 |
| `get_action(data)` | 推理：Euler 积分去噪，返回动作张量 |
| `get_action_with_cfg(data_cond, data_uncond, cfg_scale)` | CFG 推理：`v = v_cond + scale * (v_cond - v_uncond)` |

### 6.9 `nitrogen/flow_matching_transformer/modules.py` — Transformer 模块

**`DiTConfig`** — DiT 配置

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `num_attention_heads` | 8 | 注意力头数 |
| `attention_head_dim` | 64 | 每头维度 |
| `num_layers` | 12 | Transformer 层数 |
| `dropout` | 0.1 | Dropout 率 |
| `norm_type` | "ada_norm" | 归一化类型（自适应层归一化） |
| `cross_attention_dim` | None | 交叉注意力维度 |
| `positional_embeddings` | "sinusoidal" | 位置编码类型 |

**`DiT`** — Diffusion Transformer
- 接收动作隐状态（SA 嵌入）和视觉特征（VL 嵌入）
- 通过 `TimestepEncoder` 编码时间步
- 交替使用交叉注意力和自注意力（可选）
- 输出经 AdaLayerNorm 调制后投影

**`SelfAttentionTransformer`** — VL 特征混合器
- 纯自注意力 Transformer
- 用于混合视觉 token 和游戏 ID token

**`BasicTransformerBlock`** — 基础 Transformer 块
- 自注意力/交叉注意力 + FeedForward
- 支持 AdaLayerNorm（时间步条件化）和标准 LayerNorm

### 6.10 `scripts/serve.py` — 推理服务器

ZeroMQ REP 服务器，处理三种请求：

| 请求类型 | 说明 | 响应 |
|----------|------|------|
| `predict` | 传入图像，返回预测动作 | `{status, pred: {j_left, j_right, buttons}}` |
| `reset` | 重置会话缓冲 | `{status: "ok"}` |
| `info` | 查询会话信息 | `{status, info: {...}}` |

使用 100ms 轮询超时以支持 Ctrl+C 中断。

### 6.11 `scripts/play.py` — 游戏主循环

**主循环流程：**
1. 连接推理服务器，获取配置信息
2. 创建 `GamepadEnv`（60 FPS、异步模式）
3. 特定游戏初始化（isaac-ng.exe、Cuphead.exe 需要菜单操作）
4. 循环执行：
   - 截取游戏画面 → 缩放至 256x256
   - 发送至服务器预测
   - 接收 j_left、j_right、buttons
   - 转换为手柄动作（摇杆 × 32767，按键阈值 0.5）
   - 按 `action_downsample_ratio` 重复执行动作
   - 录制调试视频和干净视频
   - 记录动作日志（JSONL）

**安全机制：**
- 默认禁止菜单按键（START、BACK、GUIDE），通过 `--allow-menu` 开启

---

## 7. 关键常量与配置

### 7.1 手柄按键（21 个）

```
BACK, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT, DPAD_UP,
EAST (B), GUIDE, LEFT_SHOULDER (LB), LEFT_THUMB (LS), LEFT_TRIGGER (LT),
NORTH (Y), RIGHT_BOTTOM, RIGHT_LEFT, RIGHT_RIGHT,
RIGHT_SHOULDER (RB), RIGHT_THUMB (RS), RIGHT_TRIGGER (RT), RIGHT_UP,
SOUTH (A), START, WEST (X)
```

### 7.2 模型默认参数

| 参数 | 值 |
|------|---|
| 图像分辨率 | 256 × 256 |
| 视觉编码器 | SigLIP (google/siglip-large-patch16-256) |
| 每帧视觉 token 数 | 256 |
| 动作维度 | 25 (4 摇杆 + 21 按键) |
| 动作预测步长 | 16 |
| 时间步桶数 | 1000 |
| 隐藏层维度 | 1024 |
| 视觉隐藏维度 | 768 |
| 按键激活阈值 | 0.5 |
| ZeroMQ 默认端口 | 5555 |
| 客户端超时 | 30 秒 |

### 7.3 摇杆值范围

| 阶段 | 范围 | 说明 |
|------|------|------|
| 模型内部 | [0, 1] | 归一化表示 |
| 模型输出 | [-1, 1] | 反归一化后 |
| 虚拟手柄 | [-32768, 32767] | 发送到 vgamepad |

---

## 8. 数据流详解

```
游戏画面 (2560×1440)
    │
    │ dxcam GPU 截图
    ▼
PIL Image (2560×1440)
    │
    │ cv2.resize → 256×256
    ▼
PIL Image (256×256 RGB)
    │
    │ pickle 序列化 → ZeroMQ 发送
    ▼
═══════════════ 服务端 ═══════════════
    │
    │ SigLIP AutoImageProcessor
    ▼
pixel_values (1, C, 256, 256)
    │
    │ 加入 obs_buffer (deque)
    │ 拼接历史帧
    ▼
pixel_values (ctx_len, C, 256, 256)
    │
    │ NitrogenTokenizer.encode()
    │ 构建 vl_token_ids, sa_token_ids, attention_mask
    ▼
tokenized_data (dict of tensors)
    │
    │ SigLIP 视觉编码器
    ▼
visual_features (B, F, 256, 768)
    │
    │ VL Self-Attention Mixer
    ▼
vl_embs (B, T, 768)
    │
    │ + 噪声动作 → Action Encoder → sa_embs
    │ DiT (Euler 积分 × N 步)
    ▼
action_tensor (B, 16, 25)
    │
    │ NitrogenTokenizer.decode()
    │ unpack: [-4:-2]=j_left, [-2:]=j_right, [:-4]=buttons
    │ 摇杆: *2-1 → [-1,1]  按键: >0.5 → 0/1
    ▼
{j_left: (16,2), j_right: (16,2), buttons: (16,21)}
    │
    │ pickle 序列化 → ZeroMQ 返回
    ▼
═══════════════ 客户端 ═══════════════
    │
    │ 摇杆: × 32767 → int
    │ 按键: > 0.5 → 0/1
    │ TRIGGER: × 255 → int
    ▼
GamepadEmulator.step(action_dict)
    │
    │ vgamepad → 虚拟 Xbox 手柄
    ▼
游戏接收手柄输入
```

每次推理预测 **16 步动作**，每步动作按 `action_downsample_ratio` 倍重复执行。

---

## 9. 许可证

本项目使用 **NVIDIA License**（非商业用途）。

**允许：**
- 非商业研究目的的使用、复制、创建衍生作品、分发
- 需保留此许可证和版权声明

**禁止：**
- 商业用途
- 军事、监控、核技术或生物识别处理
- 发起专利诉讼（将导致许可终止）

**免责声明：** 本项目严格用于研究目的，不是 NVIDIA 官方产品。

---

## 10. 相关链接

- 网站：https://nitrogen.minedojo.org/
- 模型：https://huggingface.co/nvidia/NitroGen
- 数据集：https://huggingface.co/datasets/nvidia/NitroGen
- 论文：https://arxiv.org/abs/2601.02427
