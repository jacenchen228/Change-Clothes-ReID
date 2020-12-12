import os.path as osp


class Dataset(object):
    def __init__(self, train, query, gallery):
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids,  self.num_train_cams = self.parse_data(self.train)
        
        super(Dataset, self).__init__()

    def parse_data(self, data):
        """
        :param data: input data list (img_path, pid, camid, ...)
        """

        pids = set()
        camids = set()
        for item in data:
            pids.add(item[1])
            camids.add(item[2])

        return len(pids), len(camids)

    def show_summary(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print('  train    | {:5d} | {:8d} | {:9d}'.format(num_train_pids, len(self.train), num_train_cams))
        print('  query    | {:5d} | {:8d} | {:9d}'.format(num_query_pids, len(self.query), num_query_cams))
        print('  gallery  | {:5d} | {:8d} | {:9d}'.format(num_gallery_pids, len(self.gallery), num_gallery_cams))
        print('  ----------------------------------------')

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))


class ImageDataset(Dataset):
    def __init__(self, train, query, gallery):
        super(ImageDataset, self).__init__(train, query, gallery)
