import cv2 as cv
import numpy as np


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
    print('NOT EMPTY')
    return int(m / cnt)


def lidar(idx):
    path = f'./data/boxl{idx}.pgm'
    data = cv.imread(path)

    map = data[:, :].copy()
    map = cv.cvtColor(map, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(map, 128, 255, cv.THRESH_BINARY_INV)

    # paint_rect(rects, rl[idx][0])
    # paint_rect(rects, rl[idx][1])
    # paint_rect(rects, rl[idx][2])
    # paint_rect(rects, rl[idx][3])

    hu = binary[0:20, 20:binary.shape[1]-20].copy()
    hl = binary[binary.shape[0]-20:binary.shape[0], 20:binary.shape[1]-20].copy()
    vl = binary[20:binary.shape[0]-20, 0:20].copy()
    vr = binary[20:binary.shape[0]-20, binary.shape[1]-20:binary.shape[1]].copy()

    cy = binary.shape[0] // 2
    cx = binary.shape[1] // 2

    bhu = binary[cy-40:cy, cx-8:cx+8].copy()
    bhl = binary[cy:cy+40, cx-8:cx+8].copy()
    bvl = binary[cy-8:cy+8, cx-40:cx].copy()
    bvr = binary[cy-8:cy+8, cx:cx+40].copy()

    yhu = find_m(False, hu)
    yhl = find_m(False, hl)
    xvl = find_m(True, vl)
    xvr = find_m(True, vr)
    hu = cv.cvtColor(hu, cv.COLOR_GRAY2BGR)
    hl = cv.cvtColor(hl, cv.COLOR_GRAY2BGR)
    vl = cv.cvtColor(vl, cv.COLOR_GRAY2BGR)
    vr = cv.cvtColor(vr, cv.COLOR_GRAY2BGR)

    cv.line(hu, (0, yhu), (hu.shape[1], yhu), RED, 1)
    cv.line(hl, (0, yhl), (hl.shape[1], yhl), RED, 1)
    cv.line(vl, (xvl, 0), (xvl, vl.shape[0]), RED, 1)
    cv.line(vr, (xvr, 0), (xvr, vr.shape[0]), RED, 1)

    ybhu = find_m(False, bhu)
    ybhl = find_m(False, bhl)
    xbvl = find_m(True, bvl)
    xbvr = find_m(True, bvr)
    bhu = cv.cvtColor(bhu, cv.COLOR_GRAY2BGR)
    bhl = cv.cvtColor(bhl, cv.COLOR_GRAY2BGR)
    bvl = cv.cvtColor(bvl, cv.COLOR_GRAY2BGR)
    bvr = cv.cvtColor(bvr, cv.COLOR_GRAY2BGR)

    map = cv.cvtColor(map, cv.COLOR_GRAY2BGR)
    cv.line(map, (0, yhu), (map.shape[1], yhu), RED, 1)
    cv.line(map, (0, binary.shape[0]-20+yhl), (map.shape[1], binary.shape[0]-20+yhl), RED, 1)
    cv.line(map, (xvl, 0), (xvl, map.shape[0]), RED, 1)
    cv.line(map, (binary.shape[1]-20+xvr, 0), (binary.shape[1]-20+xvr, map.shape[0]), RED, 1)

    cv.line(map, (0, cy - 40 + ybhu), (map.shape[1], cy - 40 + ybhu), BLUE, 1)
    cv.line(map, (0, cy + ybhl), (map.shape[1], cy + ybhl), BLUE, 1)
    cv.line(map, (cx - 40 + xbvl, 0), (cx - 40 + xbvl, map.shape[0]), BLUE, 1)
    cv.line(map, (cx+xbvr, 0), (cx+xbvr, map.shape[0]), BLUE, 1)

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
    for c in corners_red:
        cv.circle(pts, c, 4, RED, 1)
    for c in corners_blue:
        cv.circle(pts, c, 4, BLUE, 1)

    cv.imshow('red', red)
    cv.imshow('pts', pts)

    # cv.imshow('hu', hu)
    # cv.imshow('hl', hl)
    # cv.imshow('vl', vl)
    # cv.imshow('vr', vr)

    # cv.imshow('bhu', bhu)
    # cv.imshow('bhl', bhl)
    # cv.imshow('bvl', bvl)
    # cv.imshow('bvr', bvr)
    
    # cv.imshow('rects', rects)
    cv.imshow('map', map)
    # cv.imshow('binary', binary)
    cv.waitKey(0)

    return map


def draw_lines():
    map_list = []
    for i in range(10):
        map_list.append(lidar(i))
    for i in range(10):
        cv.imshow(f'map {i}', map_list[i])
    cv.waitKey(0)


if __name__ == '__main__':
    draw_lines()
