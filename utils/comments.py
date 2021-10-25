import math


class PixelSize:
    def __init__(self, box):
        self.box = box

    def comment(self):
        p1, p2 = (int(self.box[0]), int(self.box[1])), (int(self.box[2]), int(self.box[3]))
        dw = abs(p2[0] - p1[0])
        dh = abs(p2[1] - p1[1])
        return 'Size: %i X %i' % (dw, dh)


class DistanceToObject:
    def __init__(self, box, name, conf,
                 dv_gps=(37.47492775833474, 126.94356693587622),
                 fov=84,  # fov angle of Mavic-2-Pro
                 isz=(1920, 1080),  # image pixel size of Mavic-2-Pro
                 obj_name="person",  # human
                 obj_h_m=1.8  # human height (meter)
                 ):
        self.box = box
        self.name = name
        self.conf = conf
        self.obj_name = obj_name
        self.dv_pos = dv_gps
        fov = math.pi * fov / 180
        # eps = pix_h * 0.5 / tan(fov/2)
        self.pixh_eps = isz[1] / (2 * math.tan(fov / 2))
        self.obj_h = obj_h_m

    def comment(self):
        p1, p2 = (int(self.box[0]), int(self.box[1])), (int(self.box[2]), int(self.box[3]))
        pix_h = abs(p2[1] - p1[1])
        if self.name == self.obj_name:
            mm_pix = self.obj_h / pix_h
            # dist = mm/pix * eps
            return "%.2f(m) < %.2f, %.2f>" % (mm_pix * self.pixh_eps, self.dv_pos[0], self.dv_pos[1])
        else:
            return ""


class RealDistanceToObject:
    def __init__(self, box, name, conf,
                 dv_gps=(37.47492775833474, 126.94356693587622),
                 dv_height=8.5,  # the Mavic-2-Pro height of floating
                 fov=84,  # fov angle of Mavic-2-Pro
                 isz=(1920, 1080),  # image pixel size of Mavic-2-Pro
                 obj_name="person",  # human
                 obj_h_m=1.8,  # human height (meter)
                 thresh_conf=0.85,  # inspection ratio
                 ):
        self.box = box
        self.name = name
        self.conf = conf
        self.obj_name = obj_name
        self.dv_pos = dv_gps
        self.dv_height = dv_height
        fov = math.pi * fov / 180
        # eps = pix_h * 0.5 / tan(fov/2)
        self.pixh_eps = isz[1] / (2 * math.tan(fov / 2))
        self.obj_h = obj_h_m
        self.thresh_conf = thresh_conf

    def comment(self, eps=1.414214):  # Trust the real distance after 45 degrees
        p1, p2 = (int(self.box[0]), int(self.box[1])), (int(self.box[2]), int(self.box[3]))
        pix_h = abs(p2[1] - p1[1])
        if self.name == self.obj_name and self.conf > self.thresh_conf:
            mm_pix = self.obj_h / pix_h
            obj_dist = (mm_pix * self.pixh_eps)  # dist = mm/pix * eps
            real_dist = math.sqrt(obj_dist ** 2 - self.dv_height ** 2) if obj_dist > self.dv_height * eps else obj_dist
            return "%.2f(m) < %.2f, %.2f>" % (real_dist, self.dv_pos[0], self.dv_pos[1])
        else:
            return ""


class CommentWriter:
    def __init__(self, *module):
        self.modules = module

    def __call__(self):
        str_ = ""
        for m in self.modules:
            str_ += (m.comment() + " ")
        return str_
