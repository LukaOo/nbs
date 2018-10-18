import torch.utils.data as dt
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
to_img = ToPILImage()


class CarvanaDataset(dt.Dataset):
    """ 
        Carvana features dataset.  Override torch Dataset class to implements reading from h5 files

    """

    def __init__(self, data_path, mask_path, input_size=224):
        """
        Args:
            data_path (string): Path to the images data files.
            mask_path (string): Path were images masks are placed
        """
        self.files = os.listdir(data_path)
        self.files.sort()
        self.mask_files = os.listdir(mask_path)
        self.mask_files.sort()
        self.data_path = data_path
        self.mask_path = mask_path
        assert (len(self.files) == len(self.mask_files))
        self.input_size = input_size

        self.preprocess = transforms.Compose([
            transforms.Scale((input_size, input_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def pil_load(self, path, is_input=True):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def pil_save(self, t, img_path):
        a = to_img(t)
        a.save(img_path, 'PNG')

    def __getitem__(self, idx):
        f_name = os.path.join(self.data_path, self.files[idx])
        m_name = os.path.join(self.mask_path, self.mask_files[idx])

        if os.path.exists(f_name) == False:
            raise Exception('Missing file with name ' + f_name + ' in dataset')

        input = self.pil_load(f_name)
        target = self.pil_load(m_name, False)

        input = self.preprocess(input)
        target = self.preprocess(target)
        target = torch.sum(target, dim=0).unsqueeze(0)
        target[ torch.gt(target, 0) ] = 1

        return input, target
