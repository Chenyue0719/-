from typing import List
from collections import namedtuple
import time
import math

# 坐标对象
class Point(namedtuple("Point", "x y")):
    def __repr__(self) -> str:
        return f'Point{tuple(self)!r}'

    # 比较x坐标 0-中间 1-左边 2-右边
    def is_x(self, p: int):
        type = 0;
        if self.x > p:
            type = 2;
        if self.x < p:
            type = 1;
        return type;

    # 比较y坐标
    def is_y(self, p: int):
        type = 0;
        if self.y > int:
            type = 2;
        if self.y < int:
            type = 1;
        return type;

# 坐标范围对象
class Rectangle(namedtuple("Rectangle", "lower upper")):
    def __repr__(self) -> str:
        return f'Rectangle{tuple(self)!r}'

    # 判断坐标在不在该坐标在
    def is_contains(self, p: Point) -> bool:
        return self.lower.x <= p.x <= self.upper.x and self.lower.y <= p.y <= self.upper.y

    # 比较x坐标 0-中间 1-左边 2-右边
    def is_x(self, p: Point):
        type = 0;
        if self.lower.x > p.x:
            type = 2;
        if self.upper.x < p.x:
            type = 1;
        return type;

    # 比较y坐标
    def is_y(self, p: Point):
        type = 0;
        if self.lower.y > p.y:
            type = 2;
        if self.upper.y < p.y:
            type = 1;
        return type;

# 树节点对象
class Node(namedtuple("Node", "location left right split")):
    """
    location: Point
    left: Node
    right: Node
    axis: int
    """

    def __repr__(self):
        return f'{tuple(self)!r}'

# kd树对接
class KDTree:
    """k-d tree"""

    def __init__(self):
        self._root = None
        self._n = 0

    # 第二题
    def insert(self, p: List[Point]):
        """insert a list of points"""
        self._root = createNode(p, 0);
        self._n = len(p);
        pass

    # 第二题
    def range(self, rectangle: Rectangle) -> List[Point]:
        """range query"""
        list1 = list();
        listAddPoint(self._root, list1, 1, rectangle);
        return list1;

    def searchNodeList(self, p:Point):
        list1 = list();
        listAddNode(self._root, list1, 1, p);
        return list(reversed(list1));

def listAddNode(node:Node , list1, type:int, p:Point):
    lOfR = 0;
    if type % 2 > 0:
        lOfR = p.is_x(node.location.x);
    else:
        lOfR = p.is_y(node.location.y);

    list1.append(node);
    type += 1;
    if (lOfR == 1 or lOfR == 0) and node.left is not None:
        listAddPoint(node.left, list1, type, p);
    if (lOfR == 2 or lOfR == 0) and node.right is not None:
        listAddPoint(node.right, list1, type, p);


# 往列表中增加符合范围的坐标
def listAddPoint(node:Node , list1, type:int, rectangle: Rectangle):
    lOfR = 0;
    if type % 2 > 0:
        lOfR = rectangle.is_x(node.location);
    else:
        lOfR = rectangle.is_y(node.location);
    if rectangle.is_contains(node.location):
        list1.append(node.location);
    type += 1;
    if (lOfR == 1 or lOfR == 0) and node.left is not None:
        listAddPoint(node.left, list1, type, rectangle);
    if (lOfR == 2 or lOfR == 0) and node.right is not None:
        listAddPoint(node.right, list1, type, rectangle);

# 创建所有节点关系树
def createNode(p, type):
    lenNum = len(p);
    if lenNum < 1:
        return None;
    # p = sortList(p, type);
    # p = quickSort(p,0, lenNum-1 ,type);
    p = sorted(p, key=(lambda x: x[type%2]))
    mid = findMid(p);

    # left = list();
    # right = list();
    # for index,it in enumerate(p):
    #     if index > mid:
    #         right.append(it);
    #     if index < mid:
    #         left.append(it);
    type += 1;
    return Node(
        p[mid],
        createNode(p[:mid], type),
        createNode(p[mid+1:], type),
        (type - 1)%2
    );

# 找到坐标列表中位数
def findMid(p: List[Point]):
    return len(p)//2;

# 坐标列表冒泡排序
def sortList(p: List[Point], type):
    lenNum = len(p);
    if type % 2 > 0:
        for i in range(lenNum - 1):
            for j in range(lenNum - 1 - i):
                if p[j].x > p[j + 1].x:
                    p[j], p[j + 1] = p[j + 1], p[j]
    else:
        for i in range(lenNum - 1):
            for j in range(lenNum - 1 - i):
                if p[j].y > p[j + 1].y:
                    p[j], p[j + 1] = p[j + 1], p[j]
    return p

# 计算点之间的距离
def GetDistance(nearest, target):
    d = 0
    for i in range(len(nearest) - 2):  # the last two attribute are classfication and ki
        d += (nearest[i] - target[i]) ** 2
    return math.sqrt(d)

# 第5题 knn查询
def findNearest(tree, target):
    search_path = []  # 搜索路径
    depth = 0  # 二叉树深度
    minxDistance = 0  # 最近邻点与目标点距离
    nearList = []  # 搜索路径上的点以及与目标点的距离
    nearest = None
    if tree is None:
        minxDistance = float("inf")
        nearest = None
        return nearList

    kd_root = tree
    #         k = len(kd_root.location) # assumes all points have the same dimension
    nearest = tree.location  # init nearest point
    while (kd_root):
        search_path.append(kd_root)
        #             axis = depth % k
        if target[kd_root.split] <= kd_root.location[kd_root.split]:
            kd_root = kd_root.left
        else:
            kd_root = kd_root.right
    #             depth+=1

    nearest = search_path.pop()  # 移除一个元素并返回该元素值
    minxDistance = GetDistance(nearest.location, target)  # 回溯初始最近邻
    # --生成搜索路径--
    nearList.append([nearest.location, minxDistance])
    pathNode = []
    while (search_path):
        pBack = search_path.pop()
        #             depth=depth-1
        #             axis=axis = depth % k
        axis = pBack.split
        if (abs(pBack.location[axis] - target[axis]) < minxDistance):
            if GetDistance(pBack.location, target) < GetDistance(nearest.location, target):
                nearest = pBack
                minxDistance = GetDistance(pBack.location, target)
                nearList.append([nearest.location, minxDistance])

            if target[axis] <= pBack.location[axis]:
                kd_root = pBack.right_child  # 如果target位于左子空间，就应进入右子空间
            else:
                kd_root = pBack.left_child
            if kd_root:
                search_path.append(kd_root)
                if GetDistance(kd_root.location, target) < GetDistance(nearest.location, target):
                    #                         depth+=1
                    nearest = kd_root
                    minxDistance = GetDistance(kd_root.location, target)
                    nearList.append([nearest.location, minxDistance])

    nearList = sorted(nearList, key=lambda node: node[-1])
    return nearList;



def range_test():
    points = [Point(7, 2), Point(5, 4), Point(9, 6), Point(4, 7), Point(8, 1), Point(2, 3)]
    kd = KDTree()
    kd.insert(points)
    result = kd.range(Rectangle(Point(0, 0), Point(6, 6)))
    assert sorted(result) == sorted([Point(2, 3), Point(5, 4)])


def performance_test():
    points = [Point(x, y) for x in range(1000) for y in range(1000)]
    print(len(points))

    lower = Point(500, 500)
    upper = Point(504, 504)
    rectangle = Rectangle(lower, upper)
    #  naive method
    start = int(round(time.time() * 1000))
    result1 = [p for p in points if rectangle.is_contains(p)]
    end = int(round(time.time() * 1000))
    print(f'Naive method: {end - start}ms')

    kd = KDTree()
    kd.insert(points)
    # k-d tree
    start = int(round(time.time() * 1000))
    result2 = kd.range(rectangle)
    end = int(round(time.time() * 1000))
    print(f'K-D tree: {end - start}ms')

    assert sorted(result1) == sorted(result2)

def find_test():
    points = [Point(x, y) for x in range(201) for y in range(201)];
    find = Point(200,300);
    kd = KDTree()
    kd.insert(points)
    list1  = findNearest(kd._root,find);
    print(getMin(list1))


# 获取距离最小的坐标
def getMin(nearList):
    min = None;
    point = None
    for it in nearList:
        if min is None:
            min = it[1];
            point = it[0];
        else:
            if it[1] < min:
                min = it[1];
                point = it[0];
    return point





if __name__ == '__main__':
    print("查找测试------")
    find_test()
    print("------")
    print("范围查询测试------")
    range_test()
    print("------")
    print("效率比对测试------")
    performance_test()
    print("------")