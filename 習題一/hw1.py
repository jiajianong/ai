#用chatgpt生成並修改

import random
from collections import defaultdict

# 課程資料
courses = [
    {'teacher': ' ', 'name': ' ', 'hours': -1},
    {'teacher': '甲', 'name': '機率', 'hours': 2},
    {'teacher': '甲', 'name': '線代', 'hours': 3},
    {'teacher': '甲', 'name': '離散', 'hours': 3},
    {'teacher': '乙', 'name': '視窗', 'hours': 3},
    {'teacher': '乙', 'name': '科學', 'hours': 3},
    {'teacher': '乙', 'name': '系統', 'hours': 3},
    {'teacher': '乙', 'name': '計概', 'hours': 3},
    {'teacher': '丙', 'name': '軟工', 'hours': 3},
    {'teacher': '丙', 'name': '行動', 'hours': 3},
    {'teacher': '丙', 'name': '網路', 'hours': 3},
    {'teacher': '丁', 'name': '媒體', 'hours': 3},
    {'teacher': '丁', 'name': '工數', 'hours': 3},
    {'teacher': '丁', 'name': '動畫', 'hours': 3},
    {'teacher': '丁', 'name': '電子', 'hours': 4},
    {'teacher': '丁', 'name': '嵌入', 'hours': 3},
    {'teacher': '戊', 'name': '網站', 'hours': 3},
    {'teacher': '戊', 'name': '網頁', 'hours': 3},
    {'teacher': '戊', 'name': '演算', 'hours': 3},
    {'teacher': '戊', 'name': '結構', 'hours': 3},
    {'teacher': '戊', 'name': '智慧', 'hours': 3}
]

# 老師清單
teachers = ['甲', '乙', '丙', '丁', '戊']

# 教室清單
rooms = ['A', 'B']

# 時間段清單
slots = [
    'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17',
    'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27',
    'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37',
    'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47',
    'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57',
    'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17',
    'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27',
    'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37',
    'B41', 'B42', 'B43', 'B44', 'B45', 'B46', 'B47',
    'B51', 'B52', 'B53', 'B54', 'B55', 'B56', 'B57',
]

# 初始化一個隨機排課表
def initial_schedule():
    schedule = defaultdict(list)
    for course in courses[1:]:
        slot = random.choice(slots)
        room = random.choice(rooms)
        schedule[(slot, room)].append(course)
    return schedule

# 成本函數（這是一個假設的範例，需根據實際需求進行修改）
def calculate_cost(schedule):
    cost = 0
    # 例如：懲罰同一老師有重疊的課程
    for slot_room, courses_in_slot in schedule.items():
        teachers_in_slot = defaultdict(int)
        for course in courses_in_slot:
            teachers_in_slot[course['teacher']] += course['hours']
        for teacher, hours in teachers_in_slot.items():
            if hours > 3:
                cost += 1
    return -cost  # 因為要最大化成本函數，所以取負號

# 爬山演算法
def hill_climbing():
    current_schedule = initial_schedule()
    current_cost = calculate_cost(current_schedule)
    
    while True:
        neighbors = []
        for slot_room, courses_in_slot in current_schedule.items():
            for course in courses_in_slot:
                for slot in slots:
                    if slot != slot_room[0]:  # 只考慮移到不同的時間段
                        for room in rooms:
                            neighbor_schedule = defaultdict(list)
                            for sr, courses_in_sr in current_schedule.items():
                                if sr == slot_room:
                                    neighbor_schedule[(slot, room)].append(course)
                                else:
                                    neighbor_schedule[sr] = courses_in_sr
                            neighbors.append((neighbor_schedule, calculate_cost(neighbor_schedule)))
        
        # 找出最佳（最高成本）的鄰居
        best_neighbor, best_neighbor_cost = max(neighbors, key=lambda x: x[1])
        
        if best_neighbor_cost <= current_cost:
            break
        else:
            current_schedule = best_neighbor
            current_cost = best_neighbor_cost
    
    return current_schedule

# 執行爬山演算法
final_schedule = hill_climbing()

# 輸出最終排課表
for slot_room, courses_in_slot in final_schedule.items():
    slot, room = slot_room
    print(f"{slot}, {room}: {[course['name'] for course in courses_in_slot]}")
