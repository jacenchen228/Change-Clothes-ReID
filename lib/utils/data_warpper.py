class DataWarpper(object):
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path, pid, camid, img, contour_img = self.data[idx]

        img, contour_img = self.transforms(img, contour_img)

        return img, contour_img, pid, camid, img_path

    def __len__(self):
        return len(self.data)

