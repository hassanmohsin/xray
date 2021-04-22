import numpy as np


def split_box(n, box, turn=0, min_split_frac=0.2, random_turn=False, min_dim_thr=10, vol_thr=4000):
    sides = abs(box[3:] - box[:3])
    min_dim = min(sides)
    vol = np.prod(sides)
    if n > 0 and min_dim > min_dim_thr and vol > vol_thr:
        r = np.random.random() * (1 - 2 * min_split_frac) + min_split_frac
        box1 = np.copy(box)
        box2 = np.copy(box)
        box1[turn + 3] = box[turn + 3] * r
        box2[turn + 3] = box[turn + 3] * (1 - r)
        box2[turn] = box[turn] + box[turn + 3] * r
        if random_turn:
            turn1, turn2 = np.random.randint(3), np.random.randint(3)
        else:
            turn1 = (turn + 1) % 3
            turn2 = turn1
        return np.vstack((split_box(n - 1, box1, turn1), split_box(n - 1, box2, turn2)))
    else:
        return box


def sum_volume(boxes):
    # To test if the partitions are done correctly
    if len(boxes.shape) == 1:
        boxes = boxes.reshape(1, -1)
        print(boxes)
    return np.sum(np.prod(boxes[:, 3:], axis=1))


if __name__ == "__main__":
    np.set_printoptions(precision=2)
    box = np.array([0, 0, 0, 200, 100, 80], dtype=np.float32)  # Initial box location and size, box = [x,y,z,dx,dy,dz]
    boxes = split_box(3, box, random_turn=False)
    print(boxes)
    print(sum_volume(boxes))
