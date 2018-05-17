import cv2
import numpy as np
import queue as Q
import copy


class Points:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Strategy:
    def __init__(self, a, other, w, h):
        self.map_step = np.zeros([w, h])
        if a == 1:
            self.priority = other.priority
            self.path = []
            for i in other.path:
                point_buf = Points(i.x, i.y)
                self.path.append(point_buf)
            self.now_point = other.now_point
            self.step_loss = other.step_loss
            self.map_step = other.map_step

    def __cmp__(self, other):
        return (self.priority > other.priority) - (self.priority < other.priority)


class SearchMap:

    Map_Size = 20
    sqrt_2 = 1.414

    def __init__(self, w, h):
        self.w = w
        self.h = h
        # initial visual
        step_w = SearchMap.Map_Size
        step_h = SearchMap.Map_Size
        self.visual_image = np.ones([w * step_w, h * step_h, 3], dtype="double")

        # initial the Map terrain
        map_terrain = np.zeros([w, h])
        # set target and start points

        self.start_point = Points(0, 0)
        self.start_point.x = 10
        self.start_point.y = 4
        self.target_point = Points(0, 0)
        self.target_point.x = 0
        self.target_point.y = 35

        '''
        self.start_point = Points(0, 0)
        self.start_point.x = 3
        self.start_point.y = 0
        self.target_point = Points(0, 0)
        self.target_point.x = 4
        self.target_point.y = 11
        '''
        # default map
        self.map_terrain = self.default_map(map_terrain)
        #self.map_terrain = self.default_map2(map_terrain)
        self.update_map()
        #self.set_target(0, 35)
        self.set_target(self.target_point.x, self.target_point.y)
        self.visual_image = self.set_step(self.visual_image, self.start_point.x, self.start_point.y)
        self.update_grid()
        self.display_update()
        # set strategy queue
        self.strategyQ = Q.PriorityQueue()
        self.num = 0

    def display_update(self):
        cv2.namedWindow("Map", 0)
        cv2.imshow("Map", self.visual_image)
        cv2.waitKey(0)

    def update_map(self):
        for i in range(self.w):
            start_x = i * SearchMap.Map_Size
            end_x = start_x + SearchMap.Map_Size
            for j in range(self.h):
                start_y = j * SearchMap.Map_Size
                end_y = start_y + SearchMap.Map_Size
                block_type = self.map_terrain[i, j]
                if block_type == 1000:
                    self.visual_image[start_x:end_x, start_y:end_y, :] = 0.4
                elif block_type == 4:
                    self.visual_image[start_x:end_x, start_y:end_y, 0] = 18/255.0
                    self.visual_image[start_x:end_x, start_y:end_y, 1] = 153/255.0
                    self.visual_image[start_x:end_x, start_y:end_y, 2] = 1.0
                elif block_type == 2:
                    self.visual_image[start_x:end_x, start_y:end_y, 0] = 1.0
                    self.visual_image[start_x:end_x, start_y:end_y, 1] = 105/255.0
                    self.visual_image[start_x:end_x, start_y:end_y, 2] = 65/225.0

    def update_grid(self):
        for index_w in range(1, self.w):
            row = index_w * SearchMap.Map_Size
            self.visual_image[row, :, :] = 0
        for index_h in range(1, self.h):
            col = index_h * SearchMap.Map_Size
            self.visual_image[:, col, :] = 0

    def set_target(self, x, y):
        start_x = x * SearchMap.Map_Size
        end_x = start_x + SearchMap.Map_Size
        start_y = y * SearchMap.Map_Size
        end_y = start_y + SearchMap.Map_Size
        self.visual_image[start_x:end_x, int((start_y + end_y) / 2), :] = 0
        self.visual_image[int((start_x + end_x) / 2), start_y:end_y, :] = 0

    def set_step(self, img, x, y):
        start_x = x * SearchMap.Map_Size + int(SearchMap.Map_Size / 4)
        end_x = start_x + int(SearchMap.Map_Size / 4 * 2)
        start_y = y * SearchMap.Map_Size + int(SearchMap.Map_Size / 4)
        end_y = start_y + int(SearchMap.Map_Size / 4 * 2)
        img[start_x:end_x, start_y:end_y, 2] = 1.0
        img[start_x:end_x, start_y:end_y, 0:2] = 0
        return img

    def A_strategy(self):
        # first strategy
        S1 = Strategy(0, 0, self.w, self.h)
        S1.path = []
        S1.path.append(self.start_point)
        S1.now_point = Points(self.start_point.x, self.start_point.y)
        S1.step_loss = 0
        S1.priority = self.predict_distance(self.start_point, self.target_point)
        S1.map_step[self.start_point.x, self.start_point.y] = 1
        self.strategyQ.put((float(S1.priority), self.num, S1))
        self.num = self.num - 1
        strategy_now_tup = self.strategyQ.get()
        strategy_now = Strategy(1, strategy_now_tup[2], self.w, self.h)
        now_point = Points(strategy_now.now_point.x, strategy_now.now_point.y)
        while(self.equal_points(now_point, self.target_point) == 0):
            self.search_neighbor(strategy_now)
            strategy_now_tup = self.strategyQ.get()
            strategy_now = Strategy(1, strategy_now_tup[2], self.w, self.h)
            now_point = Points(strategy_now.now_point.x, strategy_now.now_point.y)
        self.show_path_final(strategy_now)
        print(strategy_now.priority)
        return strategy_now

    def show_path(self, strategy):
        strategy_buf = Strategy(1, strategy, self.w, self.h)
        path = strategy_buf.path
        image_buf = copy.copy(self.visual_image)
        for i in path:
            point_buf = copy.copy(i)
            image_buf = self.set_step(image_buf, point_buf.x, point_buf.y)
        cv2.namedWindow("Map", 0)
        cv2.imshow("Map", image_buf)
        cv2.waitKey(1)
            # print(point_buf.x, point_buf.y)
        # print(" ")

    def show_path_final(self, strategy):
        strategy_buf = Strategy(1, strategy, self.w, self.h)
        path = strategy_buf.path
        image_buf = copy.copy(self.visual_image)
        for i in path:
            point_buf = copy.copy(i)
            image_buf = self.set_step(image_buf, point_buf.x, point_buf.y)
            cv2.namedWindow("Map", 0)
            cv2.imshow("Map", image_buf)
            cv2.waitKey(1)
        cv2.waitKey(0)

    def search_neighbor(self, strategy):
        point_now = Points(strategy.now_point.x, strategy.now_point.y)
        step_loss = strategy.step_loss
        # left
        if point_now.x > 0:
            point_left = Points(point_now.x, point_now.y)
            point_left.x = point_left.x - 1  # 左边点位置
            loss = self.map_terrain[point_left.x, point_left.y]  # 当前点的地形代价
            judge = strategy.map_step[point_left.x, point_left.y]  # judge用来判断当前点是否已经在策略中走过了
            if (loss < 1000) and (judge == 0):  # 如果既不是岩石，也没有探索过则拓展策略，得到新的策略
                self.add_new_strategy(strategy, 1 + step_loss, loss, point_left)
        # right
        if point_now.x < self.w - 1:
            point_right = Points(point_now.x, point_now.y)
            point_right.x = point_right.x + 1
            loss = self.map_terrain[point_right.x, point_right.y]
            judge = strategy.map_step[point_right.x, point_right.y]
            if (loss < 1000) and (judge == 0):
                self.add_new_strategy(strategy, 1 + step_loss, loss, point_right)
        # up
        if point_now.y > 0:
            point_up = Points(point_now.x, point_now.y)
            point_up.y = point_up.y - 1
            loss = self.map_terrain[point_up.x, point_up.y]
            judge = strategy.map_step[point_up.x, point_up.y]
            if (loss < 1000) and (judge == 0):
                self.add_new_strategy(strategy, 1 + step_loss, loss, point_up)
        # down
        if point_now.y < self.h - 1:
            point_down = Points(point_now.x, point_now.y)
            point_down.y = point_down.y + 1
            loss = self.map_terrain[point_down.x, point_down.y]
            judge = strategy.map_step[point_down.x, point_down.y]
            if (loss < 1000) and (judge == 0):
                self.add_new_strategy(strategy, 1 + step_loss, loss, point_down)
        # left-up
        if (point_now.x > 0) and (point_now.y > 0):
            point_left_up = Points(point_now.x, point_now.y)
            point_left_up.x = point_left_up.x - 1
            point_left_up.y = point_left_up.y - 1
            loss = self.map_terrain[point_left_up.x, point_left_up.y]
            judge = strategy.map_step[point_left_up.x, point_left_up.y]
            if (loss < 1000) and (judge == 0):
                self.add_new_strategy(strategy, SearchMap.sqrt_2 + step_loss, loss, point_left_up)
        # left-down
        if (point_now.x > 0) and (point_now.y < self.h - 1):
            point_left_down = Points(point_now.x, point_now.y)
            point_left_down.x = point_left_down.x - 1
            point_left_down.y = point_left_down.y + 1
            loss = self.map_terrain[point_left_down.x, point_left_down.y]
            judge = strategy.map_step[point_left_down.x, point_left_down.y]
            if (loss < 1000) and (judge == 0):
                self.add_new_strategy(strategy, SearchMap.sqrt_2 + step_loss, loss, point_left_down)
        # right-up
        if (point_now.x < self.w - 1) and (point_now.y > 0):
            point_right_up = Points(point_now.x, point_now.y)
            point_right_up.x = point_right_up.x + 1
            point_right_up.y = point_right_up.y - 1
            loss = self.map_terrain[point_right_up.x, point_right_up.y]
            judge = strategy.map_step[point_right_up.x, point_right_up.y]
            if (loss < 1000) and (judge == 0):
                self.add_new_strategy(strategy, SearchMap.sqrt_2 + step_loss, loss, point_right_up)
        # right down
        if (point_now.x < self.w - 1) and (point_now.y < self.h - 1):
            point_right_down = Points(point_now.x, point_now.y)
            point_right_down.x = point_right_down.x + 1
            point_right_down.y = point_right_down.y + 1
            loss = self.map_terrain[point_right_down.x, point_right_down.y]
            judge = strategy.map_step[point_right_down.x, point_right_down.y]
            if (loss < 1000) and (judge == 0):
                self.add_new_strategy(strategy, SearchMap.sqrt_2 + step_loss, loss, point_right_down)

    def add_new_strategy(self, strategy, step_loss, loss, point_now):
        strategy_new = Strategy(1, strategy, self.w, self.h)
        strategy_new.path.append(point_now)
        strategy_new.map_step[point_now.x, point_now.y] = 1
        strategy_new.now_point = Points(point_now.x, point_now.y)
        strategy_new.step_loss = step_loss + loss
        strategy_new.priority = step_loss + loss + self.predict_distance(point_now, self.target_point)
        self.strategyQ.put((float(strategy_new.priority), self.num, strategy_new))
        self.num = self.num - 1
        self.show_path(strategy_new)

    def predict_distance(self, a, target):
        x_dis = abs(a.x - target.x)
        y_dis = abs(a.y - target.y)
        min_dis = min(x_dis,  y_dis)
        return min_dis * SearchMap.sqrt_2 + (x_dis - min_dis) + (y_dis - min_dis)

    def equal_points(self, a, b):
        if (a.x == b.x) and (a.y == b.y):
            return 1
        return 0

    def default_map(self, map_t):
        map_t[0, 3] = 1000
        map_t[2, 0:6] = 1000
        map_t[0:2, 7] = 1000
        map_t[2, 7:11] = 1000
        map_t[3, 8] = 1000
        map_t[0:8, 12] = 1000
        map_t[5, 7:9] = 1000
        map_t[6, 2:8] = 1000
        map_t[6:12, 5] = 1000
        map_t[7, 2] = 1000
        map_t[7, 7] = 1000
        map_t[11, 2:6] = 1000
        map_t[10, 2] = 1000
        map_t[11:17, 3] = 1000
        map_t[18:20, 3] = 1000
        map_t[17:20, 7] = 1000
        map_t[15, 3:9] = 1000
        map_t[10:16, 8] = 1000
        map_t[9:11, 7] = 1000
        map_t[13, 9] = 1000
        map_t[13, 11] = 1000
        map_t[12:20, 12] = 1000
        map_t[10:13, 19:22] = 1000
        map_t[15:17, 24:26] = 1000
        map_t[10, 28] = 1000
        map_t[11, 31] = 1000
        map_t[13, 31] = 1000
        map_t[7, 36] = 1000
        map_t[9, 36] = 1000
        map_t[0, 24:40] = 4
        map_t[1, 25:40] = 4
        map_t[2, 26:40] = 4
        map_t[3, 26:37] = 4
        map_t[4, 26:36] = 4
        map_t[5:7, 27:33] = 4
        map_t[7, 29:33] = 4
        map_t[1, 34] = 2
        map_t[2, 33] = 2
        map_t[3, 32] = 2
        map_t[4:11, 33] = 2
        map_t[5:10, 34] = 2
        map_t[7:9, 35] = 2
        map_t[8:12, 32] = 2
        map_t[10, 35:37] = 2
        map_t[11, 34:36] = 2
        map_t[12:15, 34] = 2
        map_t[12:17, 33] = 2
        map_t[13:18, 32] = 2
        map_t[15:19, 31] = 2
        map_t[17:20, 30] = 2
        map_t[18:20, 29] = 2
        map_t[19, 28] = 2
        return map_t

    def default_map2(self, map_t):
        map_t[0:2, 3] = 1000
        map_t[2:5, 4] = 1000
        map_t[4:7, 5] = 1000

        return map_t





