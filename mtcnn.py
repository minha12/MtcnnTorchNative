from collections import OrderedDict
import math
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as F_trans

from utils import get_reference_facial_points, warp_and_crop_face

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _generate_bboxes(probs, offsets, scale, threshold):
    """Generate bounding boxes at places where there is probably a face using PyTorch.

    Arguments:
      probs: a float tensor of shape [n, m].
      offsets: a float tensor of shape [1, 4, n, m].
      scale: a float number, width and height of the image were scaled by this number.
      threshold: a float number.

    Returns:
      a float tensor of shape [n_boxes, 9]
    """

    stride = 2
    cell_size = 12

    # indices of boxes where there is probably a face
    inds = (probs > threshold).nonzero(as_tuple=False)

    if inds.size(0) == 0:
        return torch.empty((0, 9))

    # transformations of bounding boxes
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[:, 0], inds[:, 1]] for i in range(4)]

    score = probs[inds[:, 0], inds[:, 1]]

    # P-Net is applied to scaled images, so we need to rescale bounding boxes back
    bounding_boxes = torch.stack(
        [
            torch.round((stride * inds[:, 1].float() + 1.0) / scale),
            torch.round((stride * inds[:, 0].float() + 1.0) / scale),
            torch.round((stride * inds[:, 1].float() + 1.0 + cell_size) / scale),
            torch.round((stride * inds[:, 0].float() + 1.0 + cell_size) / scale),
            score,
            tx1,
            ty1,
            tx2,
            ty2,
        ],
        dim=1,
    )

    return bounding_boxes


def _preprocess(img):
    """Preprocessing step before feeding the network.

    Arguments:
        img: a float numpy array of shape [h, w, c].

    Returns:
        a float numpy array of shape [1, c, h, w].
    """
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5) * 0.0078125
    return img


def nms(boxes, overlap_threshold=0.5, mode="union"):
    """Non-maximum suppression in PyTorch supporting 'union' and 'min' modes.

    Arguments:
        boxes: a float tensor of shape [n, 5],
               where each row is (xmin, ymin, xmax, ymax, score).
        overlap_threshold: a float number, the IoU threshold for NMS.
        mode: 'union' or 'min'.

    Returns:
        list with indices of the selected boxes
    """
    # print("boxes", boxes.shape)
    if boxes.shape[0] == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    picked = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            picked.append(i)
            break

        i = order[0].item()
        picked.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        if mode == "union":
            overlap = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "min":
            overlap = inter / torch.min(areas[i], areas[order[1:]])

        ids = (overlap <= overlap_threshold).nonzero(as_tuple=False).squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]

    return picked


def run_first_stage(image, net, scale, threshold):
    """Run P-Net, generate bounding boxes, and do NMS using only PyTorch.

    Arguments:
      image: a torch tensor of shape [1, c, h, w] after preprocessing.
      net: an instance of pytorch's nn.Module, P-Net.
      scale: a float number, scale width and height of the image by this number.
      threshold: a float number, threshold on the probability of a face when generating
                 bounding boxes from predictions of the net.

    Returns:
      a float tensor of shape [n_boxes, 9], bounding boxes with scores and offsets (4 + 1 + 4).
    """
    # print("threshold", threshold)
    # scale the image and convert it to a tensor
    width, height = image.size
    sw, sh = math.ceil(width * scale), math.ceil(height * scale)
    img = image.resize((sw, sh), Image.BILINEAR)

    img = np.asarray(img, "float32")

    img = torch.FloatTensor(_preprocess(img)).to(device)

    output = net(img.to(device))
    probs = output[1][0, 1, :, :]  # Get the probability map
    offsets = output[0]  # Get the offsets

    boxes = _generate_bboxes(probs, offsets, scale, threshold)
    # print("boxes", boxes)
    # print("shape of boxes", boxes.shape)
    if len(boxes) == 0:
        return None

    keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
    return boxes[keep]


# reference facial points, a list of coordinates (x,y)
def calibrate_box(bboxes, offsets):
    """Transform bounding boxes to be more like true bounding boxes.
    'offsets' is one of the outputs of the nets.

    Arguments:
        bboxes: a float tensor of shape [n, 5].
        offsets: a float tensor of shape [n, 4].

    Returns:
        a float tensor of shape [n, 5].
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0

    # Adding an extra dimension to w and h to facilitate broadcasting
    w = w.unsqueeze(1)
    h = h.unsqueeze(1)

    # Element-wise multiplication
    translation = torch.cat([w, h, w, h], dim=1) * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes


def convert_to_square(bboxes):
    """Convert bounding boxes to a square form using PyTorch.

    Arguments:
        bboxes: a float tensor of shape [n, 5].

    Returns:
        a float tensor of shape [n, 5], squared bounding boxes.
    """

    square_bboxes = torch.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = torch.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes


def correct_bboxes(bboxes, width, height):
    """Crop boxes that are too big and get coordinates
    with respect to cutouts using PyTorch.

    Arguments:
        bboxes: a float tensor of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        width: a float number.
        height: a float number.

    Returns:
        dy, dx, edy, edx: int tensors of shape [n],
            coordinates of the boxes with respect to the cutouts.
        y, x, ey, ex: int tensors of shape [n],
            corrected ymin, xmin, ymax, xmax.
        h, w: int tensors of shape [n],
            just heights and widths of boxes.

        in the following order:
            [dy, edy, dx, edx, y, ey, x, ex, w, h].
    """

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    x, y, ex, ey = x1.clone(), y1.clone(), x2.clone(), y2.clone()

    dx, dy = torch.zeros(num_boxes), torch.zeros(num_boxes)
    edx, edy = w.clone() - 1, h.clone() - 1

    # Right
    ind = ex > width - 1
    edx[ind] = w[ind] + width - 2 - ex[ind]
    ex[ind] = width - 1

    # Bottom
    ind = ey > height - 1
    edy[ind] = h[ind] + height - 2 - ey[ind]
    ey[ind] = height - 1

    # Left
    ind = x < 0
    dx[ind] = 0 - x[ind]
    x[ind] = 0

    # Top
    ind = y < 0
    dy[ind] = 0 - y[ind]
    y[ind] = 0

    # Convert to integers
    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.int() for i in return_list]

    return return_list


def get_image_boxes(bounding_boxes, img, size=24):
    """Cut out boxes from the image using PyTorch.

    Arguments:
        bounding_boxes: a float tensor of shape [n, 5].
        img: an instance of PIL.Image.
        size: an integer, size of cutouts.

    Returns:
        a float tensor of shape [n, 3, size, size].
    """

    num_boxes = len(bounding_boxes)
    width, height = img.size

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(
        bounding_boxes.to("cpu"), width, height
    )

    img_boxes = np.zeros((num_boxes, 3, size, size), "float32")
    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), "uint8")

        img_array = np.asarray(img, "uint8")
        img_box[dy[i] : (edy[i] + 1), dx[i] : (edx[i] + 1), :] = img_array[
            y[i] : (ey[i] + 1), x[i] : (ex[i] + 1), :
        ]

        # resize
        img_box = Image.fromarray(img_box)
        img_box = img_box.resize((size, size), Image.BILINEAR)
        img_box = np.asarray(img_box, "float32")

        img_boxes[i, :, :, :] = _preprocess(img_box)

    return img_boxes


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 10, 3, 1)),
                    ("prelu1", nn.PReLU(10)),
                    ("pool1", nn.MaxPool2d(2, 2, ceil_mode=True)),
                    ("conv2", nn.Conv2d(10, 16, 3, 1)),
                    ("prelu2", nn.PReLU(16)),
                    ("conv3", nn.Conv2d(16, 32, 3, 1)),
                    ("prelu3", nn.PReLU(32)),
                ]
            )
        )

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

        weights = np.load("weights/pnet.npy", allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = F.softmax(a, dim=-1)
        return b, a


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 28, 3, 1)),
                    ("prelu1", nn.PReLU(28)),
                    ("pool1", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv2", nn.Conv2d(28, 48, 3, 1)),
                    ("prelu2", nn.PReLU(48)),
                    ("pool2", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv3", nn.Conv2d(48, 64, 2, 1)),
                    ("prelu3", nn.PReLU(64)),
                    ("flatten", Flatten()),
                    ("conv4", nn.Linear(576, 128)),
                    ("prelu4", nn.PReLU(128)),
                ]
            )
        )

        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

        weights = np.load("weights/rnet.npy", allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        a = F.softmax(a, dim=-1)
        return b, a


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 32, 3, 1)),
                    ("prelu1", nn.PReLU(32)),
                    ("pool1", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv2", nn.Conv2d(32, 64, 3, 1)),
                    ("prelu2", nn.PReLU(64)),
                    ("pool2", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv3", nn.Conv2d(64, 64, 3, 1)),
                    ("prelu3", nn.PReLU(64)),
                    ("pool3", nn.MaxPool2d(2, 2, ceil_mode=True)),
                    ("conv4", nn.Conv2d(64, 128, 2, 1)),
                    ("prelu4", nn.PReLU(128)),
                    ("flatten", Flatten()),
                    ("conv5", nn.Linear(1152, 256)),
                    ("drop5", nn.Dropout(0.25)),
                    ("prelu5", nn.PReLU(256)),
                ]
            )
        )

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

        weights = np.load("weights/onet.npy", allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        a = F.softmax(a, dim=-1)
        return c, b, a


class MTCNN:
    def __init__(self):
        self.pnet = PNet().to(device)
        self.rnet = RNet().to(device)
        self.onet = ONet().to(device)
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        self.device = device
        self.refrence = get_reference_facial_points(default_square=True)

    def align(self, img):
        _, landmarks = self.detect_faces(img)
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        warped_face = warp_and_crop_face(
            np.array(img), facial5points, self.refrence, crop_size=(112, 112)
        )
        return Image.fromarray(warped_face)

    def align_multi(self, img, limit=None, min_face_size=30.0):
        boxes, landmarks = self.detect_faces(img, min_face_size)
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]
        faces = []
        for landmark in landmarks:
            facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(
                np.array(img), facial5points, self.refrence, crop_size=(112, 112)
            )
            faces.append(Image.fromarray(warped_face))
        return boxes, faces

    def stage1(self, image, scales, thresholds, nms_thresholds):
        bounding_boxes = []

        for s in scales:
            boxes = run_first_stage(image, self.pnet, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)

        bounding_boxes = [i for i in bounding_boxes if i is not None]
        if len(bounding_boxes) == 0:
            return None

        bounding_boxes = torch.cat(bounding_boxes, dim=0)
        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])

        bounding_boxes = convert_to_square(bounding_boxes)

        bounding_boxes[:, 0:4] = torch.round(bounding_boxes[:, 0:4])

        return bounding_boxes

    def stage2(self, bounding_boxes, image, thresholds, nms_thresholds, device):
        if bounding_boxes is None or len(bounding_boxes) == 0:
            return None, None

        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        img_boxes = torch.FloatTensor(img_boxes).to(device)

        output = self.rnet(img_boxes)
        offsets = output[0]
        probs = output[1]

        keep = torch.where(probs[:, 1] > thresholds[1])[0]

        bounding_boxes = bounding_boxes[keep]

        bounding_boxes[:, 4] = probs[keep, 1].view(-1)

        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = torch.round(bounding_boxes[:, 0:4])

        return bounding_boxes

    def stage3(self, bounding_boxes, image, thresholds, nms_thresholds, device):
        if bounding_boxes is None or len(bounding_boxes) == 0:
            return [], []

        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        img_boxes = torch.FloatTensor(img_boxes).to(device)

        output = self.onet(img_boxes)
        landmarks = output[0]
        offsets = output[1]
        probs = output[2]

        keep = torch.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].view(-1)
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]

        landmarks[:, 0:5] = xmin.unsqueeze(1) + width.unsqueeze(1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = (
            ymin.unsqueeze(1) + height.unsqueeze(1) * landmarks[:, 5:10]
        )

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode="min")

        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return bounding_boxes, landmarks

    def detect_faces(
        self,
        image,
        min_face_size=20.0,
        thresholds=[0.6, 0.7, 0.8],
        nms_thresholds=[0.7, 0.7, 0.7],
    ):
        """
        Arguments:
            image_tensor: a torch tensor of shape [1, c, h, w] after preprocessing.
            min_face_size: a float number.
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.

        Returns:
            two torch tensors of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """

        # Extract height and width from the image tensor
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor**factor_count)
            min_length *= factor
            factor_count += 1

        bounding_boxes = []

        # Stage 1
        # print("sclaes", scales)
        bounding_boxes = self.stage1(image, scales, thresholds, nms_thresholds)
        print("stage1", bounding_boxes)
        # Stage 2
        bounding_boxes = self.stage2(
            bounding_boxes, image, thresholds, nms_thresholds, self.device
        )
        print("stage2", bounding_boxes)
        # Stage 3
        bounding_boxes, landmarks = self.stage3(
            bounding_boxes, image, thresholds, nms_thresholds, self.device
        )
        print("stage3", bounding_boxes, landmarks)
        return bounding_boxes, landmarks
