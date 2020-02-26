import argparse
import numpy as np
import glob
import os
import shutil

class HiCPreprocessor():
    def __init__(self, args):
        self.input_filepath = args.input
        self.hr_filepaths = glob.glob(self.input_filepath+'hr/*')
        self.lr_filepaths = glob.glob(self.input_filepath+'lr/*')
        self.output_filepath = args.output

    def save_hic_mat(self, mat, outfile):
        np.savetxt(outfile, mat, delimiter=' ', fmt='%.8f')

    def save_splits(self, normalize, chromosome, cell_type, lr_samples, hr_samples, lr_coords, hr_coords):
        if normalize:
            split_save_filepath = self.output_filepath+'HiCSR_dataset/samples/{}-{}-HiCSR-dataset-normalized'.format(chromosome, cell_type)
            idx_save_filepath = self.output_filepath+'HiCSR_dataset/indices/{}-{}-HiCSR-dataset-normalized'.format(chromosome, cell_type)
        else:
            split_save_filepath = self.output_filepath+'HiCSR_dataset/samples/{}-{}-HiCSR-dataset'.format(chromosome, cell_type)
            idx_save_filepath = self.output_filepath+'HiCSR_dataset/indices/{}-{}-HiCSR-dataset'.format(chromosome, cell_type)

        np.savez(split_save_filepath+'-samples', hr = hr_samples, lr = lr_samples)
        np.savez(idx_save_filepath+'-idxs', hr = np.asarray(hr_coords), lr = np.asarray(lr_coords))

    def normalize_pairs(self):
        for lr_path, hr_path in zip(self.lr_filepaths, self.hr_filepaths):
            print('Normalizing {}'.format(lr_path.split('/')[-1]))
            lr_mat, hr_mat = np.loadtxt(lr_path), np.loadtxt(hr_path)
            lr_mat_norm, hr_mat_norm = self.normalize(lr_mat), self.normalize(hr_mat)

            chromosome = lr_path.split('/')[-1].split('-')[0]
            cell_type = lr_path.split('/')[-1].split('-')[1]
            self.save_hic_mat(lr_mat_norm, self.output_filepath+'normalized/lr/{}-{}-16-norm.txt.gz'.format(chromosome, cell_type))
            self.save_hic_mat(hr_mat_norm, self.output_filepath+'normalized/hr/{}-{}-1-norm.txt.gz'.format(chromosome, cell_type))

    def normalize(self, mat):
        log_mat = np.log2(mat+1)
        norm_mat = 2*(log_mat/np.max(log_mat)) - 1
        return norm_mat

    def diagonal_splits(self, max_dist, lr_size, hr_size, normalized):
        if normalized:
            lr_paths, hr_paths = glob.glob(self.output_filepath+'normalized/lr/*'), glob.glob(self.output_filepath+'normalized/hr/*')
        else:
            lr_paths, hr_paths = glob.glob(self.input_filepath+'lr/*'), glob.glob(self.input_filepath+'hr/*')

        for lr_path, hr_path in zip(lr_paths, hr_paths):
            print('Splitting {}'.format(lr_path.split('/')[-1]))
            lr_mat, hr_mat = np.loadtxt(lr_path), np.loadtxt(hr_path)
            n_bins = lr_mat.shape[0]

            lr_samples = []
            lr_coords = []
            hr_samples = []
            hr_coords = []

            for i in range(0, n_bins, hr_size):
                for j in range(i, n_bins, hr_size):
                    if(abs(i-j) > max_dist or i+lr_size >= n_bins or j+lr_size >= n_bins):
                        continue
                    else:
                        lr_submatrix = lr_mat[i:i+lr_size, j:j+lr_size]
                        start_x = int((lr_size - hr_size)/2)
                        start_y = start_x
                        end_x = start_x + hr_size
                        end_y = start_y + hr_size
                        hr_submatrix = hr_mat[i+start_x:i+end_x, j+start_y:j+end_y]
                        lr_submatrix_coords = [(i,j), (i+lr_size,j+lr_size)]
                        hr_submatrix_coords = [(i+start_x,j+start_y), (i+end_x,j+end_y)]
                        lr_samples.append([lr_submatrix])
                        lr_coords.append(lr_submatrix_coords)
                        hr_samples.append([hr_submatrix])
                        hr_coords.append(hr_submatrix_coords)

            chromosome = lr_path.split('/')[-1].split('-')[0]
            cell_type = lr_path.split('/')[-1].split('-')[1]
            self.save_splits(normalized, chromosome, cell_type, lr_samples, hr_samples, lr_coords, hr_coords)

    def build_output_dir(self):
        os.mkdir(self.output_filepath)
        dataset_fp = self.output_filepath+'HiCSR_dataset'
        os.mkdir(dataset_fp)
        os.mkdir(dataset_fp+'/samples/')
        os.mkdir(dataset_fp+'/indices/')

        if args.normalize:
            norm_fp = self.output_filepath+'normalized'
            os.mkdir(norm_fp)
            os.mkdir(norm_fp+'/lr/')
            os.mkdir(norm_fp+'/hr/')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True,
            help="root directory where both high resolution (hr) and low resolution (lr) intrachromosomal Hi-C interaction matrices are stored")
    parser.add_argument('--output', type=str, required=True,
            help="output directory to write preprocessing results to")
    parser.add_argument('--normalize', type=int, default=1,
            help="option to normalize input before splitting data, for information about the normalization method see the paper, if set, normalized low and high resolution contact maps will be stored also  default = 1")
    parser.add_argument('--max_dist', type=int, default=200,
            help="maximum genomic distance to create submatrices, splits with a loci greater than this value are not created. Value is a number \
                    of pixels and is dependant on the resolution of the dataset to be processed, default = 200 pixels")
    parser.add_argument('--input_size', type=int, default=40,
            help="submatrix split size for low resolution input, default = 40")
    parser.add_argument('--output_size', type=int, default=28,
            help="submatrix split size for high resolution label, default = 28")
    args = parser.parse_args()

    s = HiCPreprocessor(args)

    if os.path.exists(args.output):
        print("output path already exists. Overwrite? (y/n)")
        overwrite = input()
        if overwrite.lower() == 'y':
            shutil.rmtree(args.output)
            s.build_output_dir()
        else:
            exit()

    else:
        s.build_output_dir()

    if args.normalize:
        s.normalize_pairs()

    s.diagonal_splits(args.max_dist, args.input_size, args.output_size, args.normalize)
