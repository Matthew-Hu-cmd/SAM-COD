import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os, argparse
from segment_anything import sam_model_registry, build_sam, SamPredictor 

def get_bbox(gt):
    """
    Given a ground truth segmentation image, returns a bounding box that completely encloses the segmentation.
    :param img: A ground truth segmentation image
    :return: A numpy array of the form [x1, y1, x2, y2] representing the bounding box
    """
    # Convert the image to binary
    _, binary = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY)    # Find the contours of the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find the bounding box of the contours
    x, y, w, h = cv2.boundingRect(contours[0])
    # Return the bounding box as a numpy array
    return np.array([x, y, x+w, y+h])


if __name__ == '__main__':
    sam_checkpoint = "/root/autodl-tmp/segment-anything-pth/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    # 单卡推理
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for _data_name in ['CAMO', 'COD10K', 'CHAMELEON']:  # iterate all datasets
        data_path = '/root/bachelor-thesis/Dataset/TestDataset/{}/'.format(_data_name)
        save_path = '/root/autodl-tmp/res/SAM_COD/{}/'.format(_data_name)

        sam = sam_model_registry[model_type](checkpoint= sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        
        # Load images and corresponding ground truth from data_path/Imgs and data_path/GT respectively
        img_path = os.path.join(data_path, 'Imgs')
        gt_path = os.path.join(data_path, 'GT')
        for img_file in os.listdir(img_path):
            img = cv2.imread(os.path.join(img_path, img_file))
            gt_file = img_file.split('.')[0] + '.png'
            gt = cv2.imread(os.path.join(gt_path, gt_file), cv2.IMREAD_GRAYSCALE)
            bbox = get_bbox(gt)
            # print('***load image and gt!***\n')
            predictor.set_image(img)    # embedding image using SAM
            mask, _, _ = predictor.predict(point_coords= None,     # 推理，生成Masks
                                            point_labels= None,
                                            box = bbox[None, :],
                                            multimask_output=False)
            # Save the generated mask as a PNG file
            mask_path = os.path.join(save_path, img_file.split('.')[0] + '.png')
            mask_int = (mask * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask_int)
            print('>')
        print("finish processing dataset:"+_data_name+'\n')
            
            







