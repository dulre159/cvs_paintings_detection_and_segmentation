import cv2
import numpy as np
from scipy.spatial import distance as dist
from lu_vp_detect import VPDetection
import math
from numpy import ones,vstack
from numpy.linalg import lstsq

# Function to find distance
def shortest_distance(x1, y1, a, b, c):
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
    return d

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None, None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def determine_distance_between_midpoint_and_most_near_vp_red_and_green_line(point, vpsimage):
    image_hsv = cv2.cvtColor(vpsimage, cv2.COLOR_BGR2HSV)
    red_lb = np.array([-10, 255, 255])
    red_up = np.array([10, 255, 255])
    green_lb = np.array([50, 255, 255])
    green_up = np.array([70, 255, 255])
    vps_red_mask = cv2.inRange(image_hsv, red_lb, red_up)
    vps_green_mask = cv2.inRange(image_hsv, green_lb, green_up)

    linesP_red = cv2.HoughLinesP(vps_red_mask, 1, np.pi / 180, 50, None, 60, 100)
    linesP_green = cv2.HoughLinesP(vps_green_mask, 1, np.pi / 180, 50, None, 60, 100)

    cvcmaks_red = np.zeros(vps_red_mask.shape, 'uint8')
    cvcmaks_green = np.zeros(vps_red_mask.shape, 'uint8')

    ppp = (int(point[0]), int(point[1]))

    red_lines_distance = []
    green_lines_distance = []
    if linesP_red is not None:
        for i in range(0, len(linesP_red)):
            l = linesP_red[i][0]
            x1 = l[0]
            y1 = l[1]
            x2 = l[2]
            y2 = l[3]
            a = y1 - y2
            b = x2 - x1
            c = x1*y2 - x2*y1
            dist = shortest_distance(ppp[0], ppp[1], a, b, c)
            red_lines_distance.append(dist)
            cv2.line(cvcmaks_red, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)

    if linesP_green is not None:
        for i in range(0, len(linesP_green)):
            l = linesP_green[i][0]
            x1 = l[0]
            y1 = l[1]
            x2 = l[2]
            y2 = l[3]
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            dist = shortest_distance(ppp[0], ppp[1], a, b, c)
            green_lines_distance.append(dist)
            cv2.line(cvcmaks_green, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)

    most_near_red_line_dist = None
    most_near_green_line_dist = None
    if red_lines_distance is not None and len(red_lines_distance) >= 1:
        red_lines_distance = sorted(red_lines_distance)
        most_near_red_line_dist = red_lines_distance[0]
    else:
        most_near_red_line_dist = 1000000000
    if green_lines_distance is not None and len(green_lines_distance) >= 1:
        green_lines_distance = sorted(green_lines_distance)
        most_near_green_line_dist = green_lines_distance[0]
    else:
        most_near_green_line_dist = 1000000000
        cv2.circle(cvcmaks_red, (ppp[0], ppp[1]), 8, 255, -1)
        cv2.circle(cvcmaks_green, (ppp[0], ppp[1]), 8, 255, -1)

        # cv2.imshow('vpsimage', vp_image)
        #         # cv2.imshow('vpsimage_mask', vps_red_mask)
        #         # cv2.imshow('vpsimage_mask_create_red', cvcmaks_red)
        #         # cv2.imshow('vpsimage_mask_create_green', cvcmaks_green)
        #         # key = cv2.waitKey(0)
    return most_near_red_line_dist, most_near_green_line_dist
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def order_points_old(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Use implementation of Lu's Vanishing Point Algorithm to detect the vps in the image
# and create an image where lines near the edges are clustred based on the vp to which they contribute to
# in particular 4 types of diffrent colours red,green,blue or black lines are drawn on the image near the edges of objects
# red lines are the ones that
def findVPSAndCreateVPImage(inputFrame):
    vpd = VPDetection(60)
    vps = vpd.find_vps(inputFrame)
    vps_2D = vpd.vps_2D
    VPImage = vpd.create_debug_VP_image()
    return vps_2D, VPImage

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def rectify(inputFrame, mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    approx = cv2.approxPolyDP(contours[0], 0.01 * cv2.arcLength(contours[0], True), True)
    #print("The mask polygon has " +str(len(approx)) +" sides")
    if len(approx) <= 6:
        approx = cv2.approxPolyDP(contours[0], 0.04 * cv2.arcLength(contours[0], True), True)
    m1 = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)

    # m2 = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    # c1 = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    # c2 = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)

    cv2.drawContours(m1, [approx], 0, 255, 3)
    # cv2.drawContours(m2, [approx], 0, 255, 3)

    box1 = cv2.minAreaRect(approx)
    box1 = cv2.boxPoints(box1)
    box1 = np.int0(box1)
    box1_o = order_points_old(box1)
    # print(box1)
    # print(box1_o)
    # cv2.circle(m1, (box1_o[0][0], box1_o[0][1]), 3, 255, -1)
    # cv2.circle(m1, (box1_o[1][0], box1_o[1][1]), 3, 255, -1)
    # cv2.circle(m1, (box1_o[2][0], box1_o[2][1]), 3, 255, -1)
    # cv2.circle(m1, (box1_o[3][0], box1_o[3][1]), 3, 255, -1)
    (tl, tr, br, bl) = box1_o
    # print((tl, tr, bl, br))
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    m1_bm = m1.copy()
    cv2.drawContours(m1_bm, [box1], 0, 255, 1)
    cv2.circle(m1_bm, (int(tltrX), int(tltrY)), 8, 255, -1)
    cv2.circle(m1_bm, (int(blbrX), int(blbrY)), 8, 255, -1)
    cv2.circle(m1_bm, (int(tlblX), int(tlblY)), 8, 255, -1)
    cv2.circle(m1_bm, (int(trbrX), int(trbrY)), 8, 255, -1)

    # Check if the mid points are on the contour
    (x,y) =  (tltrX, tltrY)
    failedToFindPointOnContour = False
    while cv2.pointPolygonTest(approx, (x,y), False) < 0:
        y += 1
        if y >= mask.shape[0]:
            failedToFindPointOnContour = True
            break
    if not failedToFindPointOnContour:
        (tltrX, tltrY) = (x,y)

    (x, y) = (blbrX, blbrY)
    while cv2.pointPolygonTest(approx, (x, y), False) < 0:
        y -= 1
        if y <= 0:
            failedToFindPointOnContour = True
            break
    if not failedToFindPointOnContour:
        (blbrX, blbrY) = (x,y)

    (x, y) = (tlblX, tlblY)
    while cv2.pointPolygonTest(approx, (x, y), False) < 0:
        x += 1
        if x >= mask.shape[1]:
            failedToFindPointOnContour = True
            break
    if not failedToFindPointOnContour:
        (tlblX, tlblY) = (x,y)

    (x, y) = (trbrX, trbrY)
    while cv2.pointPolygonTest(approx, (x, y), False) < 0:
        x -= 1
        if x <= 0:
            failedToFindPointOnContour = True
            break
    if not failedToFindPointOnContour:
        (trbrX, trbrY) = (x,y)

    # Once midpoints are moved on the contour
    # we try to check if the bottom and upper midpoints are near the green vp lines or red vp lines
    # for this we need to compute the distance between the nearest green and red vp line
    vps_2d, vp_image = findVPSAndCreateVPImage(inputFrame)
    upper_midpoint_distance_to_nearest_red_vp_line, upper_midpoint_distance_to_nearest_green_vp_line = determine_distance_between_midpoint_and_most_near_vp_red_and_green_line((tltrX, tltrY), vp_image)
    bottom_midpoint_distance_to_nearest_red_vp_line, bottom_midpoint_distance_to_nearest_green_vp_line = determine_distance_between_midpoint_and_most_near_vp_red_and_green_line((blbrX, blbrY), vp_image)

    upper_midpoint_vp = (vps_2d[0, 0], vps_2d[0, 1]) if upper_midpoint_distance_to_nearest_red_vp_line < upper_midpoint_distance_to_nearest_green_vp_line else (vps_2d[1, 0], vps_2d[1, 1])
    lower_midpoint_vp = (vps_2d[0, 0], vps_2d[0, 1]) if bottom_midpoint_distance_to_nearest_red_vp_line < bottom_midpoint_distance_to_nearest_green_vp_line else (vps_2d[1, 0], vps_2d[1, 1])

    # print((int(tltrX), int(tltrY)))
    # print((int(blbrX), int(blbrY)))
    # print((int(tlblX), int(tlblY)))
    # print((int(trbrX), int(trbrY)))

    cv2.circle(m1, (int(tltrX), int(tltrY)), 8, 255, -1)
    cv2.circle(m1, (int(blbrX), int(blbrY)), 8, 255, -1)
    cv2.circle(m1, (int(tlblX), int(tlblY)), 8, 255, -1)
    cv2.circle(m1, (int(trbrX), int(trbrY)), 8, 255, -1)

    new_m1 = m1.copy()
    cv2.circle(new_m1, (int(tltrX), int(tltrY)), 8, 255, -1)
    cv2.circle(new_m1, (int(blbrX), int(blbrY)), 8, 255, -1)
    cv2.circle(new_m1, (int(tlblX), int(tlblY)), 8, 255, -1)
    cv2.circle(new_m1, (int(trbrX), int(trbrY)), 8, 255, -1)

    tri_x,tri_y = line_intersection(((int(tltrX), int(tltrY)), (upper_midpoint_vp[0], upper_midpoint_vp[1])), ((int(trbrX), int(trbrY)), (vps_2d[2, 0], vps_2d[2, 1])))
    tli_x,tli_y = line_intersection(((int(tltrX), int(tltrY)), (upper_midpoint_vp[0], upper_midpoint_vp[1])), ((int(tlblX), int(tlblY)), (vps_2d[2, 0], vps_2d[2, 1])))

    bri_x, bri_y = line_intersection(((int(blbrX), int(blbrY)), (lower_midpoint_vp[0], lower_midpoint_vp[1])),
                                     ((int(trbrX), int(trbrY)), (vps_2d[2, 0], vps_2d[2, 1])))
    bli_x, bli_y = line_intersection(((int(blbrX), int(blbrY)), (lower_midpoint_vp[0], lower_midpoint_vp[1])),
                                     ((int(tlblX), int(tlblY)), (vps_2d[2, 0], vps_2d[2, 1])))

    vps_failed = False
    # If any of the points create using vanishing points are inside the contour the the method failed!
    if cv2.pointPolygonTest(approx, (tri_x,tri_y), False) > 0 or cv2.pointPolygonTest(approx, (tli_x,tli_y), False) > 0 or cv2.pointPolygonTest(approx, (bri_x, bri_y), False) > 0 or cv2.pointPolygonTest(approx, (bli_x, bli_y), False) > 0:
        vps_failed = True
    # If the points create using vanishing points are way too far from the painting well then it failed too!
    if cv2.pointPolygonTest(approx, (tri_x,tri_y), True) > 100 or cv2.pointPolygonTest(approx, (tli_x,tli_y), True) > 100 or cv2.pointPolygonTest(approx, (bri_x, bri_y), True) > 100 or cv2.pointPolygonTest(approx, (bli_x, bli_y), True) > 100:
        vps_failed = True

    cv2.circle(new_m1, (int(tli_x), int(tli_y)), 8, 255, -1)
    cv2.circle(new_m1, (int(tri_x), int(tri_y)), 8, 255, -1)
    cv2.circle(new_m1, (int(bli_x), int(bli_y)), 8, 255, -1)
    cv2.circle(new_m1, (int(bri_x), int(bri_y)), 8, 255, -1)

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # cv2.circle(m1, (box1_o[0][0], box1_o[0][1]), 8, 255, -1)
    cv2.drawContours(m1, [box1], 0, 255, 1)

    # box2 = cv2.minAreaRect(approx)
    # box2 = cv2.boxPoints(box2)
    # box2 = np.int0(box2)
    # box2_o = order_points_old(box2)
    # cv2.drawContours(m2, [box2], 0, 255, 1)

    # Compute center of shapes
    # M1 = cv2.moments(contours[0])
    # cX1 = int(M1["m10"] / M1["m00"])
    # cY1 = int(M1["m01"] / M1["m00"])
    #
    # M2 = cv2.moments(approx)
    # cX2 = int(M2["m10"] / M2["m00"])
    # cY2 = int(M2["m01"] / M2["m00"])

    # cv2.circle(m1, (int(tltrX), int(tltrY)), 3, 255, -1)
    # cv2.circle(m1, (int(blbrX), int(blbrY)), 3, 255, -1)
    # cv2.circle(m1, (int(tlblX), int(tlblY)), 3, 255, -1)
    # cv2.circle(m1, (int(trbrX), int(trbrY)), 3, 255, -1)

    # y = int(cYO - int(dA/2) -50)
    h = int(dA)
    # x = int(cXO - int(dB/2) -50)
    w = int(dB)
    # (tl, tr, br, bl) = box1_o

    if h > 600 or w > 600:
        h = int(h * 0.4)
        w = int(w * 0.4)

    p11 = None
    p21 = None

    vp_p11 = None
    vp_p21 = None

    is_rect = False
    if len(approx) == 4:
        approx_l = np.squeeze(approx, 1)
        sorted_approx = order_points_old(approx_l)
        (tl1, tr1, br1, bl1) = sorted_approx

        h = int(dist.euclidean((int(tl1[0]), int(tl1[1])), (int(bl1[0]), int(bl1[1]))))
        w = int(dist.euclidean((int(tl1[0]), int(tl1[1])), (int(tr1[0]), int(tr1[1]))))

        if h > 600 or w > 600:
            h = int(h*0.4)
            w = int(w*0.4)

        cv2.circle(m1, (int(tl1[0]), int(tl1[1])), 10, 255, -1)
        cv2.circle(m1, (int(tr1[0]), int(tr1[1])), 10, 255, -1)
        cv2.circle(m1, (int(br1[0]), int(br1[1])), 10, 255, -1)
        cv2.circle(m1, (int(bl1[0]), int(bl1[1])), 10, 255, -1)
        p11 = np.float32([ [int(tl1[0]), int(tl1[1])], [int(tr1[0]), int(tr1[1])], [int(br1[0]), int(br1[1])], [int(bl1[0]), int(bl1[1])]])
        p21 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        is_rect = True
    else:

        p11 = np.float32(
            [[int(tltrX), int(tltrY)], [int(blbrX), int(blbrY)], [int(tlblX), int(tlblY)], [int(trbrX), int(trbrY)]])
        p21 = np.float32([[int(w / 2), 0], [int(w / 2), h], [0, int(h / 2)], [w, int(h / 2)]])
        # p11 = np.float32([[tl[0], tl[1]], [tr[0], tr[1]], [int(tlblX), int(tlblY)], [int(trbrX), int(trbrY)]])
        # p21 = np.float32([[int(w / 2), 0], [int(w / 2), h], [0, int(h / 2)], [w, int(h / 2)]])
        #

    (tl1, tr1, br1, bl1) = ((tli_x, tli_y), (tri_x, tri_y), (bri_x, bri_y), (bli_x, bli_y))

    vp_h = int(dist.euclidean((int(tl1[0]), int(tl1[1])), (int(bl1[0]), int(bl1[1]))))
    vp_w = int(dist.euclidean((int(tl1[0]), int(tl1[1])), (int(tr1[0]), int(tr1[1]))))

    if vp_h > 600 or vp_w > 600:
        vp_h = int(vp_h * 0.4)
        vp_w = int(vp_w * 0.4)

    vp_p11 = np.float32([[int(tl1[0]), int(tl1[1])], [int(tr1[0]), int(tr1[1])], [int(br1[0]), int(br1[1])],
                         [int(bl1[0]), int(bl1[1])]])
    vp_p21 = np.float32([[0, 0], [vp_w, 0], [vp_w, vp_h], [0, vp_h]])

    imc = inputFrame.copy()
    imc[mask == 0] = (0, 0, 0)
    vp_r1 = None
    if vp_p11 is not None and vp_p21 is not None:
        vp_matrix1 = cv2.getPerspectiveTransform(vp_p11, vp_p21)
        vp_r1 = cv2.warpPerspective(imc, vp_matrix1, (vp_w, vp_h))

    matrix1 = cv2.getPerspectiveTransform(p11, p21)
    r1 = cv2.warpPerspective(imc, matrix1, (w, h))

    # cv2.circle(m1, (cX1, cY1), 1, 255, -1)
    # cv2.circle(m2, (cX2, cY2), 1, 255, -1)

    # print((cX1, cY1))
    # cv2.imshow('m2', m2)
    #m1_r = cv2.resize(m1, (int(inputFrame.shape[1] * 0.4), int(inputFrame.shape[0] * 0.4)), interpolation=cv2.INTER_AREA)
    #cv2.imshow('m1', m1_r)
    # cv2.imshow('r1', r1)
    # cv2.imshow('c1', c1)
    # cv2.imshow('c2', c2)
    #key = cv2.waitKey(0)
    return r1, p11, is_rect, vp_r1, vp_p11, vps_failed