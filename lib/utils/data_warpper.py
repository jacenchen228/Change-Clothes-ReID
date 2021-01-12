class DataWarpper(object):
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms

    def __getitem__(self, idx):
        data_item = self.data[idx]

        if len(data_item) == 5:
            img_path, pid, camid, img, contour_img = data_item
            img, contour_img = self.transforms(img, contour_img)
            return img, contour_img, pid, camid, img_path

        img_path, pid, camid, clothid, img, contour_img = data_item
        img, contour_img = self.transforms(img, contour_img)

        return img, contour_img, pid, camid, img_path, clothid

    def __len__(self):
        return len(self.data)

