from typing import Tuple, Callable, Optional, Any, List
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from utils.util import HRSC_CLASSES, DOTA_CLASSES, tricube_kernel

if sys.version_info[0] == 2:
    raise ValueError('Please use Python 3.')
else:
    import xml.etree.ElementTree as ET


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

diameter = 400
gaussian_map = tricube_kernel(diameter, 7)
gaussian_poly = np.float32([[0, 0], [0, diameter], [diameter, diameter], [diameter, 0]])


def gaussian(mask, area, box, size, label):
    if type(size) is tuple:
        size = size[0] * size[1]
        
    H, W = mask.shape[:2]
        
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    
    if x1*x2*x3*x4*y1*y2*y3*y4 < 0:
        return mask, area

    mask_w = max(distance([x1, y1], [x2, y2]), distance([x3, y3], [x4, y4]))
    mask_h = max(distance([x3, y3], [x2, y2]), distance([x1, y1], [x4, y4]))
    
    if mask_w > 0 and mask_h > 0:
        weight_mask = np.zeros((H, W), dtype=np.float32)

        mask_area = max(1, mask_w * mask_h)
        img_area = size

        M = cv2.getPerspectiveTransform(gaussian_poly, box.reshape((4, 2)))
        dst = cv2.warpPerspective(gaussian_map, M, (H, W), flags=cv2.INTER_LINEAR)

        mask_area = (img_area/mask_area)

        weight_mask = cv2.fillPoly(weight_mask, box.astype(np.int32).reshape((-1,4,2)), color=mask_area)

        mask[:, :, label] = np.maximum(mask[:, :, label], dst)
        area[:, :, label] = np.maximum(area[:, :, label], weight_mask)
        
    return mask, area



class HRSCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(HRSC_CLASSES, range(len(HRSC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for objs in target.iter('HRSC_Objects'):
            for obj in target.iter('HRSC_Object'):

                difficult = int(obj.find('difficult').text) == 1
                if not self.keep_difficult and difficult:
                    continue
                #label = HRSC_CLASSES.index(obj.find('Class_ID').text)
                cx = float( obj.find('mbox_cx').text )
                cy = float( obj.find('mbox_cy').text )
                w = float( obj.find('mbox_w').text )
                h = float( obj.find('mbox_h').text )
                ang = float( obj.find('mbox_ang').text ) * 180 / np.pi
                
                box = np.array([[cx-w/2, cy-h/2], [cx-w/2, cy+h/2], 
                                [cx+w/2, cy+h/2], [cx+w/2, cy-h/2]], dtype=np.float32)
                
                M = cv2.getRotationMatrix2D((cx, cy), -ang, 1.0)
                box = np.hstack((box, np.ones((box.shape[0],1))))
                rbox = np.dot(M, box.T).T
                rbox = rbox.reshape(-1)

                rbb = [rbox[0]/width, rbox[1]/height, 
                       rbox[2]/width, rbox[3]/height, 
                       rbox[4]/width, rbox[5]/height, 
                       rbox[6]/width, rbox[7]/height, 
                       0]

                res += [rbb]
            
        return res  # [[cx, cy, w, h, ang, label], ... ]

############################# UNTOUCHED BEFORE THIS #############################

class ListDataset(data.Dataset):

    """
    Dataset class for DOTA-like formatted data.
    Args:
        - root (str): the directory containing the splits
        - class (List[str]): the list of classes' names
        - out_size (Tuple[int, int]): the size of the returned data
        - split (str): the name of the split used
        - transform (Callable | None): optional transform
        - evaluation (bool): if the dataset is used during evaluation phase
        - extension (str): the extension of img files
    """

    def __init__(self, 
        root: str, 
        classes: List[str],
        out_size: Tuple[int, int], 
        split: str,
        transform: Optional[Callable[..., Any]] = None, 
        evaluation: bool = False,
        extension: str = '.png',
    ) -> None:
        # Check attributes
        assert split in ('train', 'val', 'trainval', 'test'), f"Unknown mode: {split}"
        # Save attributes
        self.root = root
        self.out_size = out_size
        self.transform = transform
        self.evaluation = evaluation
        self.extension = extension
        # Compute attributes
        self.num_classes = len(classes)
        self.classes = {class_ : i for i, class_ in enumerate(classes)}
        self.ann_path = osp.join(self.root, split, 'annfiles/')
        self.img_path = osp.join(self.root, split, 'images/')
        self.ids = sorted([x.rstrip('.txt') for x in os.listdir(self.ann_path)])
        # Set cv2 num threads
        cv2.setNumThreads(0)
                
    def get_target(self, 
        img_id: str,
    ) -> Tuple[List[float], str, cv2.Mat]:
        # Load image
        img_path = self.img_path + img_id + self.extension
        img = cv2.cvtColor(cv2.imread(img_path),  cv2.COLOR_BGR2RGB)
        assert img.shape[0] == img.shape[1], f"Only supports square images but image size is {img.shape}"
        size = img.shape[0]
        # Eventual exit point of evaluation mode
        if self.evaluation:
            return [], img_path, img
        # Load annotation
        ann_path = self.ann_path + img_id + '.txt'
        annotations = []
        with open(ann_path) as file:
            for ann in file.readlines():
                ann = ann.split()
                assert len(ann) == 10, f"ann is of length {len(ann)} and not 10 = 8 coords + 1 difficulty + 1 class"
                annotations.append(
                    [float(x)/size for x in ann[:8]] + [self.classes[ann[8]]]
                )
        return annotations, img_path, img

    def __getitem__(self, 
        index: int,
    ):
        # Retrieve data
        data_id = self.ids[index]
        target, img_path, img = self.get_target(data_id)
        # Eventual exit point of evaluation mode
        if self.evaluation: 
            return img, img_path, target

############################# UNTOUCHED BEYOND THIS #############################

        mask = np.zeros((self.out_size[0], self.out_size[1], self.num_classes), dtype=np.float32)
        area = np.zeros((self.out_size[0], self.out_size[1], self.num_classes), dtype=np.float32)

        target = np.array(target)
        
        boxes = target[:, :8] if target.shape[0]!=0 else None
        labels = target[:, 8] if target.shape[0]!=0 else None
        
        img, boxes, labels = self.transform(img, boxes, labels)
        total_size = 1

        if boxes is not None:
            target_wh = np.array([self.out_size[1], self.out_size[0]], dtype=np.float32)
            boxes = (boxes.clip(0, 1) * np.tile(target_wh, 4)).astype(np.float32)
            
            labels = labels.astype(np.int32)

            numobj = max(len(boxes), 1)
            total_size = self.sum_of_size(boxes)
            
            for box, label in zip(boxes, labels):
                mask, area = gaussian(mask, area, box, total_size/numobj, label)

        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        mask = torch.from_numpy(mask.astype(np.float32))
        area = torch.from_numpy(area.astype(np.float32))
        total_size = torch.from_numpy(np.array([total_size], dtype=np.float32))

        return img, mask, area, total_size
        
        
    def __len__(self):
        return len(self.ids)

    def sum_of_size(self, boxes):
        size_sum = 0
        
        for (x1, y1, x2, y2, x3, y3, x4, y4) in boxes:
            if x1*x2*x3*x4*y1*y2*y3*y4 < 0:
                continue

            mask_w = max(distance([x1, y1], [x2, y2]), distance([x3, y3], [x4, y4]))
            mask_h = max(distance([x3, y3], [x2, y2]), distance([x1, y1], [x4, y4]))
            size_sum = size_sum + mask_w*mask_h
            
        return size_sum
