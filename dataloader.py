from torch.utils.data.dataset import Dataset
import numpy as np
import glob

class DatasetFromFolder(Dataset):
    def __init__(self, fp, data_type='train'):
        super(DatasetFromFolder, self).__init__()
        chrom_data = glob.glob(fp+data_type+'/*')
        hr_shape = np.load(chrom_data[0])['hr'].shape[-1]
        lr_shape = np.load(chrom_data[0])['lr'].shape[-1]

        hr_combined = np.empty((0,1,hr_shape,hr_shape))
        lr_combined = np.empty((0,1,lr_shape,lr_shape))

        for chrom in chrom_data:
        	hr_combined = np.append(hr_combined, np.load(chrom)['hr'], axis=0)
        	lr_combined = np.append(lr_combined, np.load(chrom)['lr'], axis=0)

        self.lr = np.nan_to_num(lr_combined)[:]
        self.hr = np.nan_to_num(hr_combined)[:]

    def get_set(self):
    	return self.hr, self.lr

    def get_shape(self):
    	return self.hr.shape, self.lr.shape

    def __getitem__(self, index):
        high_res = self.hr[index]
        low_res = self.lr[index]
        return high_res, low_res

    def __len__(self):
    	return len(self.lr)

