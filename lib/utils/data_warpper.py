class DataWarpper(object):
    def __init__(self, data, transforms_rgb, transforms_contour=None):
        self.data = data
        self.transforms_rgb = transforms_rgb
        self.transforms_contour = None
        if transforms_contour is not None:
            self.transforms_contour = transforms_contour

    def __getitem__(self, idx):
        img_path, pid, camid, img, contour_img = self.data[idx]

        img = self.transforms_rgb(img)
        contour_img = self.transforms_contour(contour_img)

        return img, contour_img, pid, camid, img_path

    def __len__(self):
        return len(self.data)

