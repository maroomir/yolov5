import math


class PixelSize:
    def __init__(self, box):
        self.box = box

    def comment(self):
        p1, p2 = (int(self.box[0]), int(self.box[1])), (int(self.box[2]), int(self.box[3]))
        dw = abs(p2[0] - p1[0])
        dh = abs(p2[1] - p1[1])
        return 'Size: %i X %i' % (dw, dh)


class DistanceByObjectHeight:
    def __init__(self, box, name,
                 fov=77,  # fov angle of Mavic-2-Pro
                 full_sz=(5472, 3648),  # full pixel size of Mavic-2-Pro
                 obj_name="person",  # human
                 obj_h_m=1.8         # human height (meter)
                 ):
        self.box = box
        self.name = name
        self.obj_name = obj_name
        fov = math.pi * fov / 180
        # eps = pix_h * 0.5 / tan(fov/2)
        self.pixh_eps = full_sz[1] / (2 * math.tan(fov / 2))
        self.obj_h = obj_h_m

    def comment(self):
        p1, p2 = (int(self.box[0]), int(self.box[1])), (int(self.box[2]), int(self.box[3]))
        pix_h = abs(p2[1] - p1[1])
        if self.obj_name == "person":
            mm_pix = self.obj_h / pix_h
            # dist = mm/pix * eps
            return "Dist: %.2f m" % (mm_pix * self.pixh_eps)
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
