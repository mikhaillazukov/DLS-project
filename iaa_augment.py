import numpy as np
import imgaug


class IaaAugment():
    def __init__(self, augmenter):
        self.augmenter = augmenter

    def __call__(self, data):
        image = data['img']
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            data['img'] = aug.augment_image(image)
            data = self.may_augment_annotation(aug, data, shape)
        return data

    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for poly in data['text_polys']:
            new_poly = self.may_augment_poly(aug, shape, poly)
            line_polys.append(new_poly)
        data['text_polys'] = np.array(line_polys)
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly
