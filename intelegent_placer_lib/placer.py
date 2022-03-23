from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from intelegent_placer_lib.customPolygon import CustomPolygon
import intelegent_placer_lib.objectSearcher as obj_s

save_dir = "Res"
task_counter = 0
 
 #todo:

def plot_configuration(polygon, objects, filename="", filename_postfix=""):
    fig, ax = plt.subplots()
    ax.plot()

    x_arr = [item[0] for item in polygon]
    y_arr = [item[1] for item in polygon]
    ax.plot(x_arr + [x_arr[0]], y_arr + [y_arr[0]])

    for i in range(len(objects)):
        obj = objects[i]
        obj.draw(ax)

    plt.gca().set_aspect("equal")
    try:
        if filename != "":
            plt.savefig(filename + str(filename_postfix))
        else:
            plt.savefig(save_dir + "\solution_" + str(task_counter) + str(filename_postfix))
    except FileNotFoundError as e:
        print(f"can`t save fig: {e}'")

    fig.show()
    plt.close(fig)


def find_borders(polygon):
    x_arr = [item[0] for item in polygon.points]
    y_arr = [item[1] for item in polygon.points]
    return [min(x_arr), max(x_arr), min(y_arr), max(y_arr)]


def drop_point(borders):
    x1, x2, y1, y2 = borders[0], borders[1], borders[2], borders[3]
    x = np.random.randint(x1, x2)
    y = np.random.randint(y1, y2)
    return [x, y]


def try_to_reconfigure(polygon, ready_objects, obj_to_move, new_object, K=50):
    old_center = obj_to_move.center
    move_radius = np.sqrt(obj_to_move.area / np.pi)
    all_objects = []
    for o in ready_objects:
        all_objects.append(o)
    all_objects.append(new_object)

    for i in range(K):
        x = np.random.randint(old_center[0] - move_radius / 2, old_center[0] + move_radius / 2)
        y = np.random.randint(old_center[1] - move_radius / 2, old_center[1] + move_radius / 2)
        obj_to_move.set_center([x, y])
        if not polygon.is_figure_inside(obj_to_move.points):
            continue
        if len(obj_to_move.find_intersections(all_objects)) == 0:
            return True
    obj_to_move.set_center(old_center)
    return False


def try_to_put_object(obj, polygon, ready_objects, M, verbose=False):
    borders = find_borders(polygon)

    alpha_step = 30
    for alpha in range(0, 180, alpha_step):

        # пытаемся M раз положить предмет
        for j in range(M):
            # бросаем случайную точку
            center_point = drop_point(borders)

            # проверяем, что точка внутри многоугольника, если нет, то continue
            if not polygon.is_inside(center_point):
                continue
            obj.set_center(center_point)
            obj.rotate(alpha_step)

            # проверяем, что фигура находится внутри м.у
            if not polygon.is_figure_inside(obj.points):
                continue
            # проверяем, что предмет не будет пересекаться с уже уложенными
            intersections = obj.find_intersections(ready_objects)

            if j > M / 2 and len(intersections) == 1:
                if try_to_reconfigure(polygon, ready_objects, intersections[0], obj):
                    assert len(obj.find_intersections(ready_objects)) == 0
                    ready_objects.append(obj)
                    if verbose:
                        print([r_o.name for r_o in ready_objects])
                    return True

            if len(intersections) == 0:
                ready_objects.append(obj)
                return True
    return False


def placer(polygon_points, objects_names, objects, N=500, M=100):
    global task_counter
    task_counter = task_counter + 1
    plot_configuration(polygon_points, [], filename=(save_dir + f"\polygon_{task_counter}"))
    figures = []
    for i in range(len(objects)):
        points = objects[i]
        name = objects_names[i]
        figures.append(CustomPolygon(points, name))
    polygon = CustomPolygon(polygon_points, "многоугольник")

    area_sum = sum([figure.area for figure in figures])
    if area_sum > polygon.area:
        return False

    for i in range(N):
        figures = []
        for i in range(len(objects)):
            points = objects[i]
            name = objects_names[i]
            figures.append(CustomPolygon(points, name))
        figures.sort(key=lambda f: f.area, reverse=True)
        ready_objects = []

        for obj in figures:
            if not try_to_put_object(obj, polygon, ready_objects, M):
                break

        if len(ready_objects) == len(figures):
            plot_configuration(polygon.points, ready_objects, filename_postfix="True")
            return True
    plot_configuration(polygon.points, ready_objects, filename_postfix="False")
    return False


def can_pack(polygon_points, objects_names, objects):
    return placer(polygon_points, objects_names, objects )


def check_image(path, polygon=None, verbose=False):

    objects_names, objects_convex_hull = obj_s.find_objects_on_img(path)
    if verbose:
        print(f"detected objects: {objects_names}")

    result = can_pack(polygon, objects_names, objects_convex_hull)
    return result


if __name__ == '__main__':
    polygon_points = [[40, 0], [30, 90], [90, 125], [155, 140], [120, -40]]
    for i in range(1, 2):
        path = "PlacerDataset/" + str(i) + ".jpg"
        print(check_image(path, polygon_points))
