from skimage.color import rgb2gray
from skimage.segmentation import flood
from skimage.measure import regionprops
from skimage.morphology import label, binary_opening, binary_closing, disk
import numpy as np
import cv2
import albumentations


def _get_center_by_edge(mask):
    mask_props = regionprops(mask)
    center= mask_props[0].centroid
    axis_length = mask_props[0].major_axis_length

    center = (mask.shape[0]//2, mask.shape[1]//2)

    return center, axis_length


def _get_radius_by_mask_center(mask,center,input_radius=None):
    if input_radius is not None:
        return input_radius

    mask = mask.astype(np.uint8)
    ksize = max(mask.shape[1]//400*2+1,3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    index = np.where(mask>0)
    d_int = np.sqrt((index[0]-center[0])**2+(index[1]-center[1])**2)
    radius = np.max(d_int).astype(int)

    return radius


def _get_circle_by_center_bbox(shape,center,bbox,radius):
    center_mask = np.zeros(shape=shape).astype('uint8')
    tmp_mask = np.zeros(shape=bbox[2:4])
    center_tmp = (int(center[0]),int(center[1]))
    center_mask = cv2.circle(center_mask,center_tmp[::-1],int(radius),(1),-1)

    return center_mask


def black_cut_fun(img_rgb):

    h,w,c = img_rgb.shape
    corner_pixel = np.mean(img_rgb[-1,-1]).astype(np.int)
    img_zero = corner_pixel * np.ones((h+80,w+80,c), dtype=img_rgb.dtype)
    img_zero[40:-40,40:-40,:] = img_rgb
    img_rgb = img_zero

    img_gray = (rgb2gray(img_rgb) * 255).astype(np.uint8)*1.5
    img_shape = img_gray.shape

    marker_set = ((5, 5), (5, img_shape[1] - 5), (img_shape[0] - 5, 5), (img_shape[0] - 5, img_shape[1] - 5))
    marker_color_set = []
    for marker_coords in marker_set:
        marker_color_set.append(img_gray[(marker_coords[0], marker_coords[1])])
    marker_color_set_std = np.std(marker_color_set)
    marker_color_std_tol = 15
    if marker_color_set_std >= marker_color_std_tol:
        raise Exception("CHANGE BORDER COLOR MODULE: failed to get consistent border color based on four markers.")

    border_mask = np.zeros(img_gray.shape, np.uint8)
    tol_min = 10
    for marker_coords in marker_set:
        curr_mask = flood(img_gray, marker_coords, tolerance = 5)
        border_mask = np.maximum(curr_mask, border_mask)
    fg_mask = 1 - border_mask
    fg_label = label(fg_mask)
    fg_prop = regionprops(fg_label)
    fg_prop.sort(key=lambda x:x.area,reverse=True)
    fg_mask = (fg_label == fg_prop[0].label)

    border_smoothing_size = 5
    disk_ele = disk(radius = border_smoothing_size)
    fg_mask = binary_closing(fg_mask, disk_ele)
    tmp_mask = 1 - fg_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,  50))

    center, major_axis_length=_get_center_by_edge(tmp_mask)
    radius=_get_radius_by_mask_center(tmp_mask,center)
    h,w = img_shape
    s_h = max(0,int(center[0] - radius))
    s_w = max(0, int(center[1] - radius))
    bbox = (s_h, s_w, min(h-s_h,2 * radius), min(w-s_w,2 * radius))
    circle_mask=_get_circle_by_center_bbox(img_shape,(h//2,w//2),bbox,radius)

    if np.sum(fg_mask) > np.sum(circle_mask):
        result_mask = fg_mask
    else:
        result_mask = circle_mask
    mask = result_mask

    img = img_rgb * mask[:,:,np.newaxis]
    img = img[np.ix_(mask.any(1), mask.any(0))]
    return  radius , img


def pad_square(image):
    max_dim = max(image.shape[0], image.shape[1])
    padder = albumentations.augmentations.transforms.PadIfNeeded(
        min_height=max_dim,
        min_width=max_dim,
        border_mode=0,
        always_apply=True)
    image = padder(image=image)["image"]

    return image


def process_fundus(img, out_radius=500, square_pad=True):
    '''
    description: Processing fundus image
    out_radius: int,The radius of the output circle
    square_pad: bool,Whether to fill it as a square
    '''
    radius, img = black_cut_fun(img.copy())

    scale = out_radius * 1.0 / radius
    image = cv2.resize(img, (0,0), fx=scale, fy=scale)

    if square_pad:
        image = pad_square(image)

    return image


class RandomCenterCut(albumentations.core.transforms_interface.ImageOnlyTransform):
    """ Center cut """

    def __init__(self, scale = 0, always_apply=False, p=0.5):

        '''
        scale:  radius * scale
        '''
        super(RandomCenterCut, self).__init__(always_apply, p)
        self.scale = np.random.uniform(1 -scale,1)


    def apply(self, img, **params):

        h ,w ,d = img.shape
        x , y  = w//2 , h//2
        r = min(x,y) * self.scale

        mask = np.zeros((h,w) ,np.uint8)
        mask = cv2.circle(mask, (x,y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=mask)

        return img.astype(np.uint8)
