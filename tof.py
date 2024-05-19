import cv2 as cv
import numpy as np

from points import c, r, a, rc


RED = (0, 0, 255)
BLUE = (255, 0, 0)


def rotate_image(image, angle):
  if angle == 0:
      return image
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result


def paint_rect(map, tup: tuple):
    cv.rectangle(map, (tup[2], tup[0]), (tup[3], tup[1]), RED, 1)


def find_m(is_v: bool, data):
    h, w = data.shape[:2]
    m = 0
    cnt = 0
    for y in range(h):
        for x in range(w):
            if data[y][x] == 255:
                m += x if is_v else y
                cnt += 1
    if cnt == 0:
        print('EMPTY')
        exit(1)
    return int(m / cnt)


def tof(idx, rotate: bool):
    path = f'./data/boxt{idx}.{"pgm" if (a[idx] == 0) or rotate else "png"}'
    data_orig = cv.imread(path)
    data_orig_scaled = cv.resize(data_orig, (480, 480))

    data = rotate_image(data_orig, a[idx]) if rotate else data_orig
    data_scaled = cv.resize(data, (480, 480))

    if rotate:
        if a[idx] != 0:
             print(cv.imwrite(f'./new_boxt{idx}.png', data))
        cv.imshow('data_orig', data_orig)
        cv.imshow('data', data)
        cv.imshow('data_orig_scaled', data_orig_scaled)
        cv.imshow('data_scaled', data_scaled)
        cv.waitKey(0)
        return

    map = data[c[idx][0]:data.shape[0]+c[idx][1], c[idx][2]:data.shape[1]+c[idx][3]].copy()
    rects = map.copy()
    map = cv.cvtColor(map, cv.COLOR_BGR2GRAY)
    cv.imshow('find', map)
    _, binary = cv.threshold(map, 128, 255, cv.THRESH_BINARY_INV)

    paint_rect(rects, r[idx][0])
    paint_rect(rects, r[idx][1])
    paint_rect(rects, r[idx][2])
    paint_rect(rects, r[idx][3])

    paint_rect(rects, rc[idx][0])
    paint_rect(rects, rc[idx][1])
    paint_rect(rects, rc[idx][2])
    paint_rect(rects, rc[idx][3])

    hu = binary[r[idx][0][0]:r[idx][0][1]+1, r[idx][0][2]:r[idx][0][3]+1].copy()
    hl = binary[r[idx][1][0]:r[idx][1][1]+1, r[idx][1][2]:r[idx][1][3]+1].copy()
    vl = binary[r[idx][2][0]:r[idx][2][1]+1, r[idx][2][2]:r[idx][2][3]+1].copy()
    vr = binary[r[idx][3][0]:r[idx][3][1]+1, r[idx][3][2]:r[idx][3][3]+1].copy()

    bhu = binary[rc[idx][0][0]:rc[idx][0][1]+1, rc[idx][0][2]:rc[idx][0][3]+1].copy()
    bhl = binary[rc[idx][1][0]:rc[idx][1][1]+1, rc[idx][1][2]:rc[idx][1][3]+1].copy()
    bvl = binary[rc[idx][2][0]:rc[idx][2][1]+1, rc[idx][2][2]:rc[idx][2][3]+1].copy()
    bvr = binary[rc[idx][3][0]:rc[idx][3][1]+1, rc[idx][3][2]:rc[idx][3][3]+1].copy()
    
    yhu = find_m(False, hu)
    yhl = find_m(False, hl)
    xvl = find_m(True, vl)
    xvr = find_m(True, vr)
    hu = cv.cvtColor(hu, cv.COLOR_GRAY2BGR)
    hl = cv.cvtColor(hl, cv.COLOR_GRAY2BGR)
    vl = cv.cvtColor(vl, cv.COLOR_GRAY2BGR)
    vr = cv.cvtColor(vr, cv.COLOR_GRAY2BGR)

    ybhu = find_m(False, bhu)
    ybhl = find_m(False, bhl)
    xbvl = find_m(True, bvl)
    xbvr = find_m(True, bvr)
    bhu = cv.cvtColor(bhu, cv.COLOR_GRAY2BGR)
    bhl = cv.cvtColor(bhl, cv.COLOR_GRAY2BGR)
    bvl = cv.cvtColor(bvl, cv.COLOR_GRAY2BGR)
    bvr = cv.cvtColor(bvr, cv.COLOR_GRAY2BGR)

    cv.line(hu, (0, yhu), (hu.shape[1], yhu), RED, 1)
    cv.line(hl, (0, yhl), (hl.shape[1], yhl), RED, 1)
    cv.line(vl, (xvl, 0), (xvl, vl.shape[0]), RED, 1)
    cv.line(vr, (xvr, 0), (xvr, vr.shape[0]), RED, 1)

    cv.line(bhu, (0, ybhu), (bhu.shape[1], ybhu), BLUE, 1)
    cv.line(bhl, (0, ybhl), (bhl.shape[1], ybhl), BLUE, 1)
    cv.line(bvl, (xbvl, 0), (xbvl, bvl.shape[0]), BLUE, 1)
    cv.line(bvr, (xbvr, 0), (xbvr, bvr.shape[0]), BLUE, 1)

    map = cv.cvtColor(map, cv.COLOR_GRAY2BGR)
    cv.line(map, (0, r[idx][0][0]+yhu), (map.shape[1], r[idx][0][0]+yhu), RED, 1)
    cv.line(map, (0, r[idx][1][0]+yhl), (map.shape[1], r[idx][1][0]+yhl), RED, 1)
    cv.line(map, (r[idx][2][2]+xvl, 0), (r[idx][2][2]+xvl, map.shape[0]), RED, 1)
    cv.line(map, (r[idx][3][2]+xvr, 0), (r[idx][3][2]+xvr, map.shape[0]), RED, 1)

    cv.line(map, (0, rc[idx][0][0]+ybhu), (map.shape[1], rc[idx][0][0]+ybhu), BLUE, 1)
    cv.line(map, (0, rc[idx][1][0]+ybhl), (map.shape[1], rc[idx][1][0]+ybhl), BLUE, 1)
    cv.line(map, (rc[idx][2][2]+xbvl, 0), (rc[idx][2][2]+xbvl, map.shape[0]), BLUE, 1)
    cv.line(map, (rc[idx][3][2]+xbvr, 0), (rc[idx][3][2]+xbvr, map.shape[0]), BLUE, 1)

    hsv = cv.cvtColor(map, cv.COLOR_BGR2HSV)
    red = cv.inRange(hsv, (0,50,50), (10,255,255))
    blue = cv.inRange(hsv, (100,150,0), (140,255,255))
    
    h, w = red.shape[:2]
    corners_red = []
    corners_blue = []
    for i in range(1, h-1):
        for j in range(1, w-1):
            if red[i+1, j] == 255 and red[i, j+1] == 255 and red[i-1, j] == 255 and red[i, j-1] == 255:
                corners_red.append((j, i))
            if blue[i+1, j] == 255 and blue[i, j+1] == 255 and blue[i-1, j] == 255 and blue[i, j-1] == 255:
                corners_blue.append((j, i))

    pts = np.zeros(map.shape, dtype='uint8')
    for corner in corners_red:
        cv.circle(pts, corner, 4, RED, 1)
    for corner in corners_blue:
        cv.circle(pts, corner, 4, BLUE, 1)

    pts_scaled = cv.resize(pts, (480, 480))
    map_scaled = cv.resize(map, (480, 480))

    print(idx)

    cv.imshow('red', red)
    cv.imshow('pts', pts)
    cv.imshow('pts_scaled', pts_scaled)


    cv.imshow('hu', hu)
    cv.imshow('hl', hl)
    cv.imshow('vl', vl)
    cv.imshow('vr', vr)

    cv.imshow('bhu', bhu)
    cv.imshow('bhl', bhl)
    cv.imshow('bvl', bvl)
    cv.imshow('bvr', bvr)
    
    cv.imshow('rects', rects)
    cv.imshow('map', map)
    cv.imshow('data_orig_scaled', data_orig_scaled)
    cv.imshow('data_scaled', data_scaled)
    cv.imshow(f'map_scaled', map_scaled)
    cv.imshow('binary', binary)
    cv.waitKey(0)

    return map


def draw_lines_tof():
    map_list = []
    for i in range(10):
        map_list.append(tof(i, False))
    for i in range(10):
        cv.imshow(f'map {i}', map_list[i])
    cv.waitKey(0)


if __name__ == '__main__':
    # tof(3, False)
    draw_lines_tof()
