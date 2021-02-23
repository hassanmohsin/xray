def lines_to_voxels(line_list, pixels):
    for x in range(len(pixels)):
        is_black = False
        # Choose the lines where x lies
        lines = list(find_relevant_lines(line_list, x))
        if not is_black and not lines:
            continue
        target_ys = list(map(lambda line: int(generate_y(line, x)), lines))
        for y in range(len(pixels[x])):
            if is_black:
                pixels[x][y] = True
            if y in target_ys:
                for line in lines:
                    if on_line(line, x, y):
                        is_black = not is_black
                        pixels[x][y] = True

        if is_black:
            print("an error has occurred at x:%d, z:%d" % (x, line_list[0][0][2]))


def find_relevant_lines(line_list, x, ind=0):
    for line in line_list:
        same = False
        above = False
        below = False
        for pt in line:
            if pt[ind] > x:
                above = True
            elif pt[ind] == x:
                same = True
            else:
                below = True
        if above and below:
            yield line
        elif same and above:
            yield line


def generate_y(line, x):
    if line[1][0] == line[0][0]:
        return -1
    ratio = (x - line[0][0]) / (line[1][0] - line[0][0])
    y_dist = line[1][1] - line[0][1]
    new_y = line[0][1] + ratio * y_dist
    return new_y


def on_line(line, x, y):
    new_y = generate_y(line, x)
    if int(new_y) != y:
        return False
    if int(line[0][0]) != x and int(line[1][0]) != x and (
            max(line[0][0], line[1][0]) < x or min(line[0][0], line[1][0]) > x):
        return False
    if int(line[0][1]) != y and int(line[1][1]) != y and (
            max(line[0][1], line[1][1]) < y or min(line[0][1], line[1][1]) > y):
        return False
    return True
