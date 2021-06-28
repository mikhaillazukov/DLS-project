import cv2


class ResizeShortSize:
    def __init__(self, short_size, resize_text_polys=True):
        self.short_size = short_size
        self.resize_text_polys = resize_text_polys

    def __call__(self, data: dict) -> dict:
        im = data['img']
        text_polys = data['text_polys']

        h, w, _ = im.shape
        short_edge = min(h, w)
        if short_edge < self.short_size:
            scale = self.short_size / short_edge
            new_height = int(round(h * scale / 32) * 32)
            new_width = int(round(w * scale / 32) * 32)
            im = cv2.resize(im, (new_width, new_height))
            scale = (new_width / w, new_height / h)
            if self.resize_text_polys:
                text_polys[:, 0] *= scale[0]
                text_polys[:, 1] *= scale[1]

        data['img'] = im
        data['text_polys'] = text_polys
        return data
