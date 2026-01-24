#!/usr/bin/env python3
"""
贪吃蛇游戏 - Snake Game
使用方向键控制蛇的移动，吃食物得分！
按 Q 退出游戏
"""

import curses
import random
import time

def main(stdscr):
    # 初始化
    curses.curs_set(0)  # 隐藏光标
    stdscr.nodelay(1)   # 非阻塞输入
    stdscr.timeout(100) # 刷新速度（毫秒）

    # 获取屏幕大小
    sh, sw = stdscr.getmaxyx()

    # 创建游戏窗口
    box_height = min(20, sh - 2)
    box_width = min(40, sw - 2)
    win = curses.newwin(box_height, box_width, (sh - box_height) // 2, (sw - box_width) // 2)
    win.keypad(1)
    win.timeout(100)

    # 初始化蛇的位置（在中间）
    snake_y = box_height // 2
    snake_x = box_width // 4

    # 蛇的身体
    snake = [
        [snake_y, snake_x],
        [snake_y, snake_x - 1],
        [snake_y, snake_x - 2]
    ]

    # 初始化食物位置
    food = [box_height // 2, box_width // 2]
    win.addch(food[0], food[1], '🍎'[0] if False else '*', curses.A_BOLD)

    # 初始方向（向右）
    key = curses.KEY_RIGHT

    # 分数
    score = 0

    # 游戏主循环
    while True:
        # 绘制边框和分数
        win.clear()
        win.border(0)

        # 显示分数
        score_text = f" 分数: {score} | Q退出 "
        win.addstr(0, 2, score_text)

        # 绘制食物
        try:
            win.addch(food[0], food[1], '*', curses.A_BOLD | curses.color_pair(0))
        except:
            pass

        # 绘制蛇
        for i, segment in enumerate(snake):
            try:
                if i == 0:
                    win.addch(segment[0], segment[1], '@')  # 蛇头
                else:
                    win.addch(segment[0], segment[1], 'o')  # 蛇身
            except:
                pass

        win.refresh()

        # 获取用户输入
        next_key = win.getch()

        # 处理退出
        if next_key == ord('q') or next_key == ord('Q'):
            break

        # 防止反向移动
        if next_key in [curses.KEY_LEFT, curses.KEY_RIGHT, curses.KEY_UP, curses.KEY_DOWN]:
            if next_key == curses.KEY_LEFT and key != curses.KEY_RIGHT:
                key = next_key
            elif next_key == curses.KEY_RIGHT and key != curses.KEY_LEFT:
                key = next_key
            elif next_key == curses.KEY_UP and key != curses.KEY_DOWN:
                key = next_key
            elif next_key == curses.KEY_DOWN and key != curses.KEY_UP:
                key = next_key

        # 计算新的蛇头位置
        new_head = [snake[0][0], snake[0][1]]

        if key == curses.KEY_DOWN:
            new_head[0] += 1
        if key == curses.KEY_UP:
            new_head[0] -= 1
        if key == curses.KEY_LEFT:
            new_head[1] -= 1
        if key == curses.KEY_RIGHT:
            new_head[1] += 1

        # 插入新蛇头
        snake.insert(0, new_head)

        # 检查是否吃到食物
        if snake[0] == food:
            score += 10
            # 生成新食物
            while True:
                food = [
                    random.randint(2, box_height - 2),
                    random.randint(2, box_width - 2)
                ]
                if food not in snake:
                    break
            # 加速
            new_timeout = max(50, 100 - score // 2)
            win.timeout(new_timeout)
        else:
            # 移除蛇尾
            snake.pop()

        # 检查游戏结束条件
        # 撞墙
        if (snake[0][0] <= 0 or snake[0][0] >= box_height - 1 or
            snake[0][1] <= 0 or snake[0][1] >= box_width - 1):
            break

        # 撞到自己
        if snake[0] in snake[1:]:
            break

    # 游戏结束画面
    win.clear()
    win.border(0)

    game_over_msg = "游戏结束!"
    final_score_msg = f"最终分数: {score}"
    restart_msg = "按任意键退出"

    win.addstr(box_height // 2 - 1, (box_width - len(game_over_msg)) // 2, game_over_msg)
    win.addstr(box_height // 2, (box_width - len(final_score_msg) - 2) // 2, final_score_msg)
    win.addstr(box_height // 2 + 2, (box_width - len(restart_msg)) // 2, restart_msg)

    win.nodelay(0)
    win.getch()

if __name__ == "__main__":
    print("=" * 40)
    print("     🐍 贪吃蛇游戏 Snake Game 🐍")
    print("=" * 40)
    print()
    print("游戏说明:")
    print("  ↑ ↓ ← →  方向键控制蛇的移动")
    print("  *        食物（吃掉得10分）")
    print("  @        蛇头")
    print("  o        蛇身")
    print("  Q        退出游戏")
    print()
    print("按 Enter 开始游戏...")
    input()

    try:
        curses.wrapper(main)
        print("\n感谢游玩！再见！👋")
    except KeyboardInterrupt:
        print("\n游戏已退出")
