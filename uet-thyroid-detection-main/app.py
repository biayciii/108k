import gradio as gr
from src.faster_rcnn.models.thyroid_module import ThyroidDetector
from src.faster_rcnn.datamodules import get_test_transforms
import numpy as np
import pydicom
import torch
from scipy import signal
import cv2
from PIL import Image

# Common functions
def get_dcm_im(dcm_path):
    img = pydicom.dcmread(dcm_path).pixel_array
    # img = np.concatenate([img[..., None]] * 3, axis=2)
    return img

def irrelevant_cutting(im: np.ndarray, cutting_threshold):
    # Cut irrelevant part of images
    if im.shape[:2] != (512, 512):
        im = im[:cutting_threshold, :cutting_threshold]

    return im

def increase_count(im, factor=1):
    # scharr = np.array([[1, 1, 1],

    #                 [1, 2, 1],

    #                 [1, 1, 1]])*factor
    
    if factor <= 0:
        return im

    scharr = (
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 2, 2, 2, 1],
                [1, 2, 4, 2, 1],
                [1, 2, 2, 2, 1],
                [1, 1, 1, 1, 1],
            ]
        )
        * factor
    )

    im = signal.convolve2d(im, scharr, boundary="symm", mode="same")
    # im = ndimage.gaussian_filter(im.astype(np.float32)*1000*factor, sigma=1, mode='reflect').astype(np.uint16)
    im[im < 0] = 0
    im[im > 255] = 255
    return im

def create_imbatch(im: np.ndarray, brighness_levels: int):
    # im shape HxW
    bright_factors = [2**i for i in range(brighness_levels - 1)]
    bright_factors.insert(0, 0) # Original image
    
    im_batch = np.vstack([increase_count(im, factor) for factor in bright_factors]).reshape(brighness_levels, *im.shape)
    im_batch = im_batch.transpose(1, 2, 0) # transpose to HxWxnum_level
    return im_batch.astype(np.uint8)

def draw_box(rgb, color, tl, tf, box, label):
    rgb = rgb.copy()
    if box is None:
        return rgb
    img_with_box = cv2.rectangle(rgb, tuple(box[:2]), tuple(box[2:]), color, thickness=tl, lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    img_with_box = cv2.rectangle(rgb, (box[0], box[1] - t_size[1] - 3), (box[0] + t_size[0], box[1]), color, -1)
    img_with_box = cv2.putText(
        img_with_box, 
        label,
        (box[0], box[1] - 2), 
        0, 
        tl / 3, 
        (255, 255, 255), 
        thickness=tf,
        lineType=cv2.LINE_AA
    )
    return img_with_box

# Faster-RCNN
def create_faster_rcnn_input(img):
    # Prepare input for faster-rcnn
    dummy_bbox = torch.rand(4).unsqueeze(0) + 1
    dummy_bbox[:, 2:4] += dummy_bbox[:, 0:2]
    dummy_labels = torch.tensor([1], dtype=torch.int64)
    params = {"image": img, "bboxes": dummy_bbox, "labels": dummy_labels}
    # Get coressponding faster-rcnn transforms
    faster_rcnn_transforms = get_test_transforms()
    input = faster_rcnn_transforms(**params)["image"]
    
    return input

def get_frcnn_box_scores(pred, org_sz):
    pred = pred[0]
    shoulder_idx = pred['scores'][pred['labels'] == 1].argmax().item() # 1 mean shoulder
    shoulder_score = pred['scores'][pred['labels'] == 1][shoulder_idx]
    thyroid_idx = pred['scores'][pred['labels'] == 2].argmax().item() # 2 mean thyroid
    thyroid_score = pred['scores'][pred['labels'] == 2][thyroid_idx]

    thyroid_box = pred['boxes'][pred['labels'] == 2][thyroid_idx].tolist()
    thyroid_box = [int(c) for c in thyroid_box]
    shoulder_box = pred['boxes'][pred['labels'] == 1][shoulder_idx].tolist()
    shoulder_box = [int(c) for c in shoulder_box]
    
    return shoulder_box, shoulder_score, thyroid_box, thyroid_score

# Load faster-rcnn model
faster_rcnn_module = ThyroidDetector(num_level=2, iou_thresholds=[0.5, 0.5])
faster_rcnn_module = faster_rcnn_module.load_from_checkpoint('/media/vinh/3f144d46-6de0-49c5-aedb-ede7be595d7c/Research/project/models/faster-rcnn/2023-04-23_20-33-55/checkpoints/epoch_011.ckpt', map_location=torch.device('cpu'))
frcnn = faster_rcnn_module.model
frcnn.eval()

# DETR
from easydict import EasyDict
from src.detr.models.backbone import build_backbone
from src.detr.models.transformer import build_transformer
from src.detr.models.detr import DETR
import src.detr.datasets.transforms as T

def make_detr_transforms(brighness_levels):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    return T.Compose([
        T.ToTensor(),
        T.Normalize(
            means * (brighness_levels // 3) + means[:brighness_levels % 3], 
            stds * (brighness_levels // 3) + stds[:brighness_levels % 3]
        )
    ])

def create_detr_input(img):    
    input, _ = make_detr_transforms(img.shape[-1])(img, None)
    return input

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b

def get_detr_box_scores(pred, org_sz):
    probas = pred['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.65

    bboxes_scaled = rescale_bboxes(pred['pred_boxes'][0, keep], org_sz).numpy()
    probas, labels = probas.max(-1)
    probas = probas[keep].data.numpy()
    labels = labels[keep].data.numpy()
    shoulder_box = bboxes_scaled[labels == 1][0].astype(np.int32).tolist()
    shoulder_score = probas[labels == 1][0]
    thyroid_box = bboxes_scaled[labels == 2][0].astype(np.int32).tolist()
    thyroid_score = probas[labels == 1][0]
    return shoulder_box, shoulder_score, thyroid_box, thyroid_score

def get_detr_args():
    args = {}
    args['frozen_weights'] = None
    # * Backbone
    args['lr_backbone']=1e-5
    args['backbone']='resnet50'
    args['dilation'] = False
    args['brighness_levels']=4
    args['position_embedding']='sine'

    # * Transformer
    args['enc_layers']=6
    args['dec_layers']=6
    args['dim_feedforward']=2048
    args['hidden_dim']=256
    args['dropout']=0.1
    args['nheads']=8
    args['num_queries']=75
    args['pre_norm'] = False
    
    args['thresh']=0.65

    # * Segmentation
    args['masks'] = False
    return EasyDict(args)

args = get_detr_args()
backbone = build_backbone(args)
transformer = build_transformer(args)

detr = DETR(
    backbone,
    transformer,
    num_classes=3,
    num_queries=75,
    aux_loss=True,
)
checkpoint = torch.load('/media/vinh/3f144d46-6de0-49c5-aedb-ede7be595d7c/Research/project/models/detr/output/resnet50_75nq_coscl2_4bl/resnet50_75nq_coscl2_4bl.pth', map_location='cpu')
detr.load_state_dict(checkpoint['model'])
detr.eval()

# YOLOv7
# Load yolov7
import sys
sys.path.insert(0, './src/yolov7')

from models.experimental import attempt_load
from utils.general import scale_coords, non_max_suppression

file = '/media/vinh/3f144d46-6de0-49c5-aedb-ede7be595d7c/Research/project/models/yolov7/3bl.pt'
yolov7 = attempt_load(file, map_location='cpu')
yolov7.eval()

def get_yolov7_box_scores(pred, im_sz):
    out = pred[0] # Return both prediction and losss -> Get prediction at position 0
    out = non_max_suppression(out, conf_thres=0.25, iou_thres=0.65)
    shapes = ((512, 512), ((1.0, 1.0), (0.0, 0.0)))
    predn = out[0].clone()
    predn[:, :4] = scale_coords(im_sz, predn[:, :4], shapes[0], shapes[1]).round()
    # Get bounding box that have maximum probability
    labels = predn[:, -1]
    scores = predn[:, -2]
    boxes = predn[:, :4]
    
    if torch.sum(labels == 0) > 0:
        shoulder_idx = scores[labels == 0].argmax()
        shoulder_score = scores[labels == 0][shoulder_idx].item()
        shoulder_box = boxes[labels == 0][shoulder_idx].type(torch.int32).tolist()
    else:
        shoulder_score = -1
        shoulder_box = None
    
    if torch.sum(labels == 1) > 0:
        thyroid_idx = scores[labels == 1].argmax()
        thyroid_score = scores[labels == 1][thyroid_idx].item()
        thyroid_box = boxes[labels == 1][thyroid_idx].type(torch.int32).tolist()
    else:
        thyroid_score = -1
        thyroid_box = None

    return shoulder_box, shoulder_score, thyroid_box, thyroid_score

def create_yolov7_input(img):
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    input = torch.from_numpy(img)
    return input

# Load logsitic model for classification residual thyroid tissues
import pickle
# Save
with open('logistic_model.pkl','rb') as f:
    logistic_model = pickle.load(f)

# Compose
get_box_scores = {
    'frcnn': get_frcnn_box_scores,
    'detr': get_detr_box_scores,
    'yolov7': get_yolov7_box_scores
}

model = {
    'frcnn': frcnn,
    'detr': detr,
    'yolov7': yolov7
}

brightness_dict = {
    'frcnn': 2,
    'detr': 4,
    'yolov7': 3
}

create_model_input = {
    'frcnn': create_faster_rcnn_input,
    'detr': create_detr_input,
    'yolov7': create_yolov7_input
}

# Functions for computing RSi
def truncate_bbox(bbox, h, w):
    """

    Args:
        bbox (list): [x0, y0, x1, y1]
    """
    
    bbox = [e if e >= 0 else 0 for e in bbox]
    bbox[0] = bbox[0] if bbox[0] < w else w - 1
    bbox[2] = bbox[2] if bbox[2] < w else w - 1
    bbox[1] = bbox[1] if bbox[1] < h else h - 1
    bbox[3] = bbox[3] if bbox[3] < h else h - 1
    return bbox

def cal_avg_uptake(img, bbox):
    # bbox = [int(x) for x in bbox]
    bbox = truncate_bbox(bbox, *img.shape)
    a = (bbox[2] - bbox[0]) / 2
    b = (bbox[3] - bbox[1]) / 2
    # area = a * b * 4
    c_x = (bbox[0] + bbox[2]) / 2
    c_y = (bbox[1] + bbox[3]) /2
    
    x = list(range(img.shape[1]))
    y = list(range(img.shape[0]))
    xx, yy = np.meshgrid(x, y)
    
    uptake = np.sum(img[((xx - c_x)**2 / (a**2) + (yy - c_y)**2 / (b**2)) <= 1]) 
    # ROI = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return np.log(uptake)

def predict(model_name: str, f: gr.components.File):
    img = irrelevant_cutting(get_dcm_im(f.name), 256)
    org_im = img.copy()
    brightness_levels = brightness_dict[model_name]
    if model_name == 'yolov7':
        img = increase_count(img, 2**(brightness_levels - 2)) # 0, 1, 2, 4, 8, ...
        img = np.stack([img]*3, axis=-1)
        img = img.astype(np.float32) / 255.0
        # img = np.uint8(img)
    else:
        img = create_imbatch(img, brightness_levels).astype(np.float32) / 255.0
    # Different between each model
    input = create_model_input[model_name](img)
    r = org_im.shape[0] / input.shape[1]
    with torch.no_grad():
        pred = model[model_name](input.unsqueeze(0))
    shoulder_box, shoulder_score, thyroid_box, thyroid_score = get_box_scores[model_name](pred, img.shape[:2][::-1])
    shoulder_box = [int(c*r) for c in shoulder_box] if shoulder_box else None
    thyroid_box = [int(c*r) for c in thyroid_box] if thyroid_box else None
    
    # Common
    tl = 1
    tf = 1
    rgb = Image.fromarray(img[..., 1].copy() * 255).convert('RGB')
    rgb = np.asarray(rgb)
    
    label = 'Shoulder {:.2f}'.format(shoulder_score)
    img_with_box = draw_box(rgb, (0, 0, 255), tl, tf, shoulder_box, label)
    label = 'Thyroid {:.2f}'.format(thyroid_score)
    img_with_box = draw_box(img_with_box, (255, 0, 0), tl, tf, thyroid_box, label)
    img_with_box = Image.fromarray(img_with_box)
    
    # Code for computing RSi and diagnosis residual, nonresidual
    rsi = cal_avg_uptake(org_im, thyroid_box) - cal_avg_uptake(org_im, shoulder_box) if thyroid_box is not None and shoulder_box is not None else -1 
    probs = logistic_model.predict_proba([[rsi]])[0] if rsi != -1 else [-1, -1]
    return rsi, img_with_box, {'non-residual': probs[0], 'residual': probs[1]}

demo = gr.Interface(
    fn=predict, 
    inputs=[
        gr.components.Dropdown(choices=['detr', 'frcnn', 'yolov7'], value='detr', label='Select model'), 
        gr.components.File(type='file', label='Select DICOM scan image'), 
    ],
    outputs=[
        gr.components.Textbox(label='RSI'),
        gr.components.Image(type='pil', shape=(200, 200)),
        gr.outputs.Label(num_top_classes=2, label='Thyroid residual tissues diagnosis')
    ],
    description='Residual thyroid tissues detection'
)

if __name__=="__main__":             
    demo.launch()