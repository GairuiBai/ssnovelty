
import numpy
import torch
from torchvision import transforms

transform_0 = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.RandomRotation(degrees=(0, 180)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
transform_1 = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=(90, 90)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

transform_2 = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=(180, 180)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

transform_3 = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=(270, 270)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
transform_trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=(90, 90)),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

def rotate_img(img, rot):
    eyes = numpy.eye(5)
    if rot == 0:  # 0 degrees rotation
        label_r = 0
        img_x = transform_0(img)
        return img_x ,label_r
    elif rot == 1:  # 90 degrees rotation
        label_r = 1
        img_x = transform_1(img)
        # img_x = numpy.flipud(numpy.transpose(img, (1, 0, 2)))
        return img_x ,label_r
    elif rot == 2:  # 90 degrees rotation
        label_r = 2
        img_x = transform_2(img)
        # img_x = numpy.fliplr(numpy.flipud(img))
        return img_x ,label_r

    elif rot == 3:  # 270 degrees rotation / or -90
        label_r = 3
        img_x = transform_3(img)
        # img_x = numpy.transpose(numpy.flipud(img), (1, 0, 2))
        return img_x ,label_r
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

def rotate_img_trans(img,rot):
    if rot == 0:  # 0 degrees rotation

        img_x = transform_0(img)
        return img_x
    elif rot == 1:  # 90 degrees rotation

        # img_x = numpy.rot90(img)
        img_x = numpy.rot90(img)
        img_x = numpy.transpose(img_x,(1,0,2))
        img_x = torch.from_numpy(img_x[:,:,::-1].copy())
        return img_x
    elif rot == 2:  # 90 degrees rotation

        img_x = transform_2(img)
        # img_x = numpy.fliplr(numpy.flipud(img))
        return img_x

    elif rot == 3:  # 270 degrees rotation / or -90

        img_x = transform_3(img)
        # img_x = numpy.transpose(numpy.flipud(img), (1, 0, 2))
        return img_x
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


def rotate_img_recon(img, rot,img_r):
    eyes = numpy.eye(5)
    if rot == 0:  # 0 degrees rotation
        label_r = 0
        return img ,label_r
    elif rot == 1:  # 90 degrees rotation
        label_r = 1
        img_x = transform_trans(img)
        # img_x = numpy.flipud(numpy.transpose(img, (1, 0, 2)))
        return img_x ,label_r
    elif rot == 2:  # 90 degrees rotation
        label_r = 2
        img_x = transform_2(img)
        # img_x = numpy.fliplr(numpy.flipud(img))
        return img_x ,label_r

    elif rot == 3:  # 270 degrees rotation / or -90
        label_r = 3
        img_x = transform_3(img)
        # img_x = numpy.transpose(numpy.flipud(img), (1, 0, 2))
        return img_x ,label_r

    elif rot == 4:  # 270 degrees rotation / or -90
        label_r = 4

        return img_r ,label_r
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')