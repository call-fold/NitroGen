# NitroGen 测试游戏 - 快速入门指南

## 概述

这个目录包含两个用于测试 NitroGen AI 游戏代理的小游戏：
- **Simple Game** (`simple_game.py`) - 基础测试游戏
- **Platformer Game** (`platformer_game.py`) - 完整的 2D 平台跳跃游戏 ⭐ 推荐

## 快速开始

### 1. 安装依赖

```bash
pip install pygame
```

### 2. 运行游戏（手动模式）

#### 方式 A：直接运行 Python 脚本
```bash
# 运行平台游戏（推荐）
python test_game/platformer_game.py

# 或运行简单游戏
python test_game/simple_game.py
```

#### 方式 B：打包为 EXE 后运行
```bash
# 安装打包工具
pip install pyinstaller

# 打包游戏
python test_game/build_game.py --game platformer

# 运行打包后的游戏
dist/NitroGenPlatformerGame.exe
```

### 3. 与 NitroGen 一起使用

#### 场景 1：手动收集训练数据

1. **启动推理服务器**
   ```bash
   python scripts/serve.py <path_to_ng.pt> --port 5555 --cfg 1.0
   ```

2. **启动游戏**
   ```bash
   python test_game/platformer_game.py
   ```

3. **使用真实手柄或键盘玩游戏**

#### 场景 2：自动 AI 游玩

1. **打包游戏为 EXE**
   ```bash
   python test_game/build_game.py --game platformer
   ```

2. **启动游戏**
   ```bash
   dist/NitroGenPlatformerGame.exe
   ```

3. **启动推理服务器**
   ```bash
   python scripts/serve.py <path_to_ng.pt> --port 5555 --cfg 1.0
   ```

4. **让 AI 玩游戏**
   ```bash
   python scripts/play.py --process "NitroGenPlatformerGame.exe" --port 5555
   ```

## 控制说明

### Platformer Game（推荐）

| 游戏手柄 | 功能 |
|----------|------|
| 左摇杆 | 移动角色 |
| 右摇杆 | 瞄准方向 |
| A 按钮 | 跳跃 |
| B 按钮 | 冲刺 |
| X 按钮 | 射击 |
| Y 按钮 | 护盾 |
| Start | 暂停 |
| 扳机 | 精细控制 |

| 键盘 | 功能 |
|------|------|
| WASD/方向键 | 移动 |
| IJKL | 瞄准 |
| 空格 | 跳跃 |
| Shift | 冲刺 |
| Z/X | 射击 |
| C | 护盾 |
| ESC | 退出 |

### Simple Game

| 游戏手柄 | 功能 |
|----------|------|
| 左摇杆 | 移动 |
| A 按钮 | 跳跃 |
| B 按钮 | 冲刺 |
| X 按钮 | 攻击 |

| 键盘 | 功能 |
|------|------|
| WASD/方向键 | 移动 |
| 空格 | 跳跃 |
| Shift | 冲刺 |
| Z | 攻击 |

## 测试组件

运行单元测试验证游戏组件：
```bash
python test_game/test_game_modules.py
```

## 游戏特性对比

| 特性 | Simple Game | Platformer Game |
|------|-------------|-----------------|
| 玩家移动 | ✓ | ✓ |
| 跳跃 | ✓ | ✓ |
| 收集物品 | ✓ | ✓ |
| 粒子效果 | ✓ | ✓ |
| 多种敌人 | ✗ | ✓ |
| 射击系统 | ✗ | ✓ |
| 护盾能力 | ✗ | ✓ |
| 移动平台 | ✗ | ✓ |
| 难度递增 | ✗ | ✓ |
| 生命系统 | ✗ | ✓ |

## 常见问题

**Q: 游戏无法被 NitroGen 捕获？**
A: 确保游戏已打包为 EXE，且进程名与 `play.py` 中的参数匹配。

**Q: 手柄无响应？**
A: 确保手柄已连接，游戏启动时会显示检测到的手柄名称。

**Q: 游戏运行卡顿？**
A: 关闭其他应用程序，或降低游戏 FPS。

**Q: 如何更改游戏难度？**
A: 编辑 `platformer_game.py` 中的 `difficulty` 相关参数。

## 文件说明

```
test_game/
├── __init__.py              # 包初始化文件
├── simple_game.py           # 简单测试游戏
├── platformer_game.py       # 平台跳跃游戏（推荐）
├── test_game_modules.py     # 单元测试
├── build_game.py            # 打包脚本
├── README.md                # 详细文档
└── QUICKSTART.md            # 本文件（快速入门）
```

## 下一步

1. 运行游戏，熟悉游戏玩法
2. 修改游戏参数，测试不同的场景
3. 收集游戏数据用于训练
4. 使用 NitroGen AI 自动玩游戏

## 许可证

与 NitroGen 项目相同的许可证。
