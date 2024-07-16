import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
image_path = 'uml.png'
y_tolerance = 15
def save_list_to_txt(data, filename):
    with open(filename, mode='w') as file:
        for row in data:
            line = '\t'.join(map(str, row))
            file.write(line + '\n')


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype(np.float32), (p2 - p1).astype(np.float32)
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

rectangles = find_squares(cv2.imread(image_path))

def merge_rectangles(squares, proximity_threshold=10, epsilon=1e-6):
    def are_rectangles_close(rect1, rect2, threshold):
        edges1 = [rect1[i:i + 2] for i in range(4)] + [rect1[0:1]]
        edges2 = [rect2[i:i + 2] for i in range(4)] + [rect2[0:1]]

        for edge1 in edges1:
            for edge2 in edges2:
                dist = min(np.linalg.norm(p1 - p2) for p1 in edge1 for p2 in edge2)
                if dist <= threshold + epsilon:
                    return True
        return False

    def merge_two_rectangles(rect1, rect2):
        all_points = np.vstack((rect1, rect2))
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)

        return np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ])

    merged = sorted(squares, key=lambda r: np.prod(np.ptp(r, axis=0)), reverse=True)
    i = 0
    while i < len(merged):
        j = i + 1
        merged_this_iteration = False
        while j < len(merged):
            if are_rectangles_close(merged[i], merged[j], proximity_threshold):
                merged[i] = merge_two_rectangles(merged[i], merged[j])
                merged.pop(j)
                merged_this_iteration = True
            else:
                j += 1
        if not merged_this_iteration:
            i += 1

    return merged

def draw_rectangles(image_path, rectangles):
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    cropped_dir = "cropped_images"
    os.makedirs(cropped_dir, exist_ok=True)

    for i, rect in enumerate(rectangles):
        pts = rect.astype(np.int32)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        if not (x_min == 0 and y_min == 0 and x_max == image_width and y_max == image_height):
            padding = 20
            y_min_crop = max(0, y_min - padding)
            y_max_crop = min(image_height, y_max + padding)
            x_min_crop = max(0, x_min - padding)
            x_max_crop = min(image_width, x_max + padding)

            cropped_image = image[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
            cropped_image_path = os.path.join(cropped_dir, f"cropped_{i}.png")
            cv2.imwrite(cropped_image_path, cropped_image)
            print(f"Cropped image saved to {cropped_image_path}")

            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

merged_rectangles = merge_rectangles(rectangles)
print(merged_rectangles)

def is_point_inside_rect(point, rect):
    x, y = point
    x_min, y_min = np.min(rect, axis=0)
    x_max, y_max = np.max(rect, axis=0)
    return x_min <= x <= x_max and y_min <= y <= y_max

def count_rectangles_inside(outer_rect, all_rectangles):
    count = 0
    for inner_rect in all_rectangles:
        if np.array_equal(outer_rect, inner_rect):
            continue
        if all(is_point_inside_rect(point, outer_rect) for point in inner_rect):
            count += 1
    return count

def remove_outer_rectangles_with_multiple_inner(rectangles):
    to_remove = set()
    num_rectangles = len(rectangles)

    for i in range(num_rectangles):
        inner_count = count_rectangles_inside(rectangles[i], rectangles)
        if inner_count > 1:
            to_remove.add(i)

    filtered_rectangles = [rect for idx, rect in enumerate(rectangles) if idx not in to_remove]
    return filtered_rectangles

filtered_rectangles = remove_outer_rectangles_with_multiple_inner(merged_rectangles)

def is_point_inside_rect(point, rect):
    x, y = point
    x_min, y_min = np.min(rect, axis=0)
    x_max, y_max = np.max(rect, axis=0)
    return x_min <= x <= x_max and y_min <= y <= y_max

def is_rectangle_inside(inner_rect, outer_rect):
    return all(is_point_inside_rect(point, outer_rect) for point in inner_rect)

def remove_inner_rectangles(rectangles):
    to_remove = set()
    num_rectangles = len(rectangles)

    for i in range(num_rectangles):
        for j in range(num_rectangles):
            if i != j and is_rectangle_inside(rectangles[i], rectangles[j]):
                to_remove.add(i)

    filtered_rectangles = [rect for idx, rect in enumerate(rectangles) if idx not in to_remove]
    return filtered_rectangles
f2iltered_rectangles = remove_inner_rectangles(filtered_rectangles)
#----------------------------------------------------

def find_close_rectangles(rectangles, y_tolerance=10, x_tolerance=10):
    close_pairs = []
    for i, rect1 in enumerate(rectangles):
        for j, rect2 in enumerate(rectangles):
            if i < j:
                y_close = any(abs(point1[1] - point2[1]) <= y_tolerance for point1 in rect1 for point2 in rect2)
                x_close = any(abs(point1[0] - point2[0]) <= x_tolerance for point1 in rect1 for point2 in rect2)
                if y_close and x_close:
                    close_pairs.append((i, j))
    return close_pairs

def merge_rectangles(rect1, rect2):
    min_x = min(np.min(rect1[:, 0]), np.min(rect2[:, 0]))
    max_x = max(np.max(rect1[:, 0]), np.max(rect2[:, 0]))
    min_y = min(np.min(rect1[:, 1]), np.min(rect2[:, 1]))
    max_y = max(np.max(rect1[:, 1]), np.max(rect2[:, 1]))
    return np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]], dtype=np.int32)

def merge_close_rectangles2(rectangles, y_tolerance, x_tolerance=10):
    close_pairs = find_close_rectangles(rectangles, y_tolerance, x_tolerance)
    merged_rectangles = []
    merged_indices = set()

    for i, j in close_pairs:
        if i not in merged_indices and j not in merged_indices:
            merged_rect = merge_rectangles(rectangles[i], rectangles[j])
            merged_rectangles.append(merged_rect)
            merged_indices.add(i)
            merged_indices.add(j)

    for k in range(len(rectangles)):
        if k not in merged_indices:
            merged_rectangles.append(rectangles[k])

    return merged_rectangles

last_rect=merge_close_rectangles2(f2iltered_rectangles,y_tolerance)
#----------------------------------------------------
print(filtered_rectangles)
print("merged:", len(merged_rectangles))
print("filtered:", len(filtered_rectangles))
print("filtered2:", len(f2iltered_rectangles))
print("last_rect:", len(last_rect))

for rect in filtered_rectangles:
    print(rect)
draw_rectangles(image_path, last_rect)

