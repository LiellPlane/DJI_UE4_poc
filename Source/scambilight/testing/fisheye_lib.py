import math
import numpy as np
import collections
import cv2

class fisheye_tool():
    def __init__(
            self,
            img_width_height: tuple[int, int],
            image_circle_size: int) -> None:
        self.image_circle_size = image_circle_size
        self.width = img_width_height[0]
        self.height = img_width_height[1]
        self.offset_y =  ((image_circle_size//2) - (self.height//2))
        self.offset_x =  ((image_circle_size//2) - (self.width//2))

    def pt_to_reverse_fisheye_inefficient(self, x, y):
        y_off = y + self.offset_y
        ny = ((2 * y_off) / (self.image_circle_size)) - 1
        ny2 = ny * ny
        x_off = x + self.offset_x
        nx = ((2 * x_off) / (self.image_circle_size)) - 1
        nx2 = nx * nx
        r = math.sqrt(nx2 + ny2)
        #  discard pixels outside from circle!
        if (0.0 <= r and r <= 1.0):
            nr = math.sqrt(1.0 - (r * r))
            nr = (r + (1.0 - nr)) / 2.0
            # discard radius greater than 1.0
            if (nr <= 1.0):
                theta = math.atan2(ny, nx)
                nxn = nr * math.cos(theta)
                nyn = nr * math.sin(theta)
                x2 = int(
                    ((nxn + 1) * self.image_circle_size) / 2.0)
                y2 = int(
                    ((nyn + 1) * self.image_circle_size) / 2.0)
                x2 = x2 - self.offset_x
                y2 = y2 - self.offset_y
                if x2 < self.width and y2 < self.height:
                    #self.check_inverse_pt(x2, y2)
                    return (x2, y2)
        return None
    
    def check_inverse_pt(self, x, y):
        x2, y2 = x, y
        if x2 > self.width or y2 > self.height:
            return None
        y2 = y2 + self.offset_y
        x2 = x2 + self.offset_x

        nyn = ((y2 * 2) / self.image_circle_size) - 1
        nxn = ((x2 * 2) / self.image_circle_size) - 1

        theta = math.asin(nyn / nr)

    def fish_eye_image(self, img, reverse):
        fish_eye = np.empty_like(img)
        for y in range(0, self.height):
            for x in range(0, self.width):
                trans_ = self.pt_to_reverse_fisheye_inefficient(x, y)
                if trans_ is not None:
                    trans_x, trans_y = trans_
                    if reverse is False:
                        fish_eye[y, x, :] = img[trans_y, trans_x, :]
                    else:
                        fish_eye[trans_y, trans_x, :] = img[y, x, :]
        return fish_eye
    
    def bruteforce_fish_eye_img(self, img):
        """experiment create fish eye image"""
        fish_eye = np.empty_like(img)
        for y in range(0, self.height, 2):
            for x in range(0, self.width, 2):
                trans_ = self.brute_force_find_fisheye_pt((x, y))
                if trans_ is not None:
                    trans_x, trans_y = trans_
                    if 0 < trans_y < self.height -1:
                        if 0 < trans_x < self.width -1:
                            fish_eye[trans_y, trans_x, :] = img[y, x, :]
        return fish_eye

    def brute_force_find_fisheye_pt(self, point):
        """for each input pt, we want to find out the
        fish-eye pt. Ideally we can reverse the radial function
        but for MVP lets brute force it"""

        dist, fisheyed_pt = self.get_nearest_approx(
            step=int(self.width/10),
            start_pt_x=0,
            start_pt_y=0,
            end_pt_x = self.width,
            end_pt_y = self.height,
            pt = point
        )

        return fisheyed_pt
    
    def get_nearest_approx(
            self,
            step,
            start_pt_x,
            start_pt_y,
            end_pt_x,
            end_pt_y,
            pt):
        pts = {}

        for x in range(start_pt_x, end_pt_x, step):
            for y in range(start_pt_y, end_pt_y, step):
                res = self.pt_to_reverse_fisheye_inefficient(x, y)
                if res is not None:
                    dist = np.linalg.norm(np.asarray(pt)-np.asarray(res))
                    pts[dist] = [x, y]

        od = collections.OrderedDict(sorted(pts.items()))

        dist, fisheyed_pt =  list(od.keys())[0], od[list(od.keys())[0]]

        if step / 2 < 2:
            return dist, fisheyed_pt

        if dist > 2:
            dist, fisheyed_pt = self.get_nearest_approx(
                step=int(step/2),
                start_pt_x = int(fisheyed_pt[0]-(step)),
                start_pt_y = int(fisheyed_pt[1]-(step)),
                end_pt_x = int(fisheyed_pt[0]+(step)),
                end_pt_y = int(fisheyed_pt[1]+(step)),
                pt = pt
            )
        return dist, fisheyed_pt


def convert_pts_to_convex_hull(points:list[list[int, int]]):
    return cv2.convexHull(np.array(points, dtype='int32'))
