import cv2
import numpy as np

class Designer:

    available_interpolation = [
     'INTER_NEAREST',
     'INTER_LINEAR',
     'INTER_AREA',
     'INTER_CUBIC',
     'INTER_LANCZOS4'
    ]

    def __init__(self, input_size,
                       output_size,
                       thickness,
                       range_value,
                       fading,
                       interpolation,
                       crop_to_draw_region):
        self.crop_to_draw_region = crop_to_draw_region
        self.region_min = None
        self.region_max = None
        self.input_size = input_size
        self.output_size = output_size
        # compute the thickness according to the diagonal
        diagonal = sum([x ** 2 for x in input_size]) ** 0.5
        self.thickness = int(thickness * diagonal)
        self.range_value = range_value
        self.fading = fading

        # catch random interpolation
        self.interpolation_method = interpolation
        if self.interpolation_method == 'RANDOM':
            interpolation = np.random.choice(Designer.available_interpolation)
        self.interpolation = eval(f'cv2.{interpolation}')

        # to store the current state
        self.img = None
        self.out_img = None

    def new_image(self):
        # create new input and output image with zeros
        input_shape = list(self.input_size)[::-1]
        output_shape = list(self.output_size)[::-1]
        self.img = np.zeros(input_shape)
        self.out_img = np.zeros(output_shape)
        self.region_min = None
        self.region_max = None

        # catch random interpolation
        if self.interpolation_method == 'RANDOM':
            interpolation = np.random.choice(Designer.available_interpolation)
            self.interpolation = eval(f'cv2.{interpolation}')

    def check_and_expand_region(self, pt):
        padding = 2
        min_point_x = max(pt[0] - (self.thickness / 2) - padding, 0)
        min_point_y = max(pt[1] - (self.thickness / 2) - padding, 0)
        if self.region_min is None:
            self.region_min = np.array(pt)
        if min_point_x < self.region_min[0]:
            self.region_min[0] = min_point_x
        if min_point_y < self.region_min[1]:
            self.region_min[1] = min_point_y

        max_point_x = min(pt[0] + (self.thickness / 2) + padding, self.img.shape[1])
        max_point_y = min(pt[1] + (self.thickness / 2) + padding, self.img.shape[0])
        if self.region_max is None:
            self.region_max = np.array(pt)
        if max_point_x > self.region_max[0]:
            self.region_max[0] = max_point_x
        if max_point_y > self.region_max[1]:
            self.region_max[1] = max_point_y

    def draw(self, pt_1, pt_2):
        value = np.random.randint(*self.range_value) # random pixel value
        line_img = self.img * 0 # create an image to draw the new linea
        # fade line
        if self.fading != 1:
            # superimposed sub-thickness-lines beginnig by the thicker one
            for t in range(self.thickness, 0, -1):
                # compute the linear fade ratio
                fade = (self.thickness - t - 1) / self.thickness
                fade = self.fading + fade * (1 - self.fading)
                fade_value = int(value * fade)
                # draw the sub thickness line
                line_img = cv2.line(line_img, pt_1, pt_2, fade_value, t)

        # update the current imgs
        self.img = self.img + (255 - self.img) / 255 * line_img
        # Crop the output image to only the drawn region
        if self.crop_to_draw_region:
            self.check_and_expand_region(pt_2)
            if self.region_min is not None and self.region_max is not None:
                x = self.region_min[0]
                y = self.region_min[1]
                width = self.region_max[0] - self.region_min[0]
                height = self.region_max[1] - self.region_min[1]
                if width > 0 and height > 0:
                    self.out_img = cv2.resize(self.img[y:y+height, x:x+width], self.output_size, self.interpolation).astype(np.uint8)
                else:
                    self.out_img = cv2.resize(self.img, self.output_size, self.interpolation).astype(np.uint8)
        else:
            self.out_img = cv2.resize(self.img, self.output_size, self.interpolation).astype(np.uint8)

    def get_output(self):
        return self.out_img
