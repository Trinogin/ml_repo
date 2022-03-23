import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class CustomPolygon:
    def __init__(self, points, name):
        self.name = name
        self.normalized_points = CustomPolygon.normalize_to_origin(points)
        self.points = self.normalized_points.copy()
        self.center = [0, 0]
        self.alpha = 0
        self.sh_polygon = Polygon(self.normalized_points)
        self.area = self.sh_polygon.area

    def is_inside(self, point):
        p1 = Point(point[0], point[1])
        return self.sh_polygon.contains(p1)

    def is_figure_inside(self, points):
        return Polygon(points).within(self.sh_polygon)

    def is_overlap(self, another):
        return self.sh_polygon.intersects(another.sh_polygon)

    def rotate_points(self, alpha):
        c_x, c_y = self.center[0], self.center[1]
        for i in range(len(self.points)):
            point = self.points[i]
            x, y = point[0] - c_x, point[1] - c_y
            cos = np.cos(alpha)
            sin = np.sin(alpha)
            self.points[i] = [c_x + x * cos - y * sin, c_y + x * sin + y * cos]

    def rotate(self, alpha):
        self.rotate_points(alpha)
        self.alpha = self.alpha + alpha
        self.sh_polygon = Polygon(self.points)

    def set_center(self, new_center):
        for i in range(len(self.normalized_points)):
            p = self.normalized_points[i]
            self.points[i] = [p[0] + new_center[0], p[1] + new_center[1]]
        self.center = new_center
        self.rotate_points(self.alpha)
        self.sh_polygon = Polygon(self.points)

    @staticmethod
    def normalize_to_origin(points):
        center = CustomPolygon.get_center(points)
        normalized_points = []
        for p in points:
            normalized_points.append([p[0] - center[0], p[1] - center[1]])
        return normalized_points

    @staticmethod
    def get_center(points):
        x_arr = [item[0] for item in points]
        y_arr = [item[1] for item in points]
        return [(min(x_arr) + max(x_arr)) / 2, (min(y_arr) + max(y_arr)) / 2]

    def draw(self, ax):
        x_arr = [item[0] for item in self.points]
        y_arr = [item[1] for item in self.points]
        ax.plot(x_arr + [x_arr[0]], y_arr + [y_arr[0]])

    def find_intersections(self, ready_objects):
        intersections = []
        for ready_obj in ready_objects:
            if self is ready_obj:
                continue
            if self.is_overlap(ready_obj):
                intersections.append(ready_obj)
        return intersections
