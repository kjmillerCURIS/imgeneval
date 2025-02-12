import os
import sys
import clip
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
sys.path.append('VPEval/src')
sys.path.append('VPEval/src/dino')
from dino.vpeval.model.modeling import Model as DinoModel
from detection_plotter import draw_detections


DETECTION_THRESHOLD = 0.3 #default is 0.4
BLUR_SIGMA = 5.0
SPATIAL_DIFF_THRESHOLD = 0.1 #on same 0-1 scale as bbox
SPATIAL_CONTAINMENT_THRESHOLD = 0.5
ATTRIBUTE_THRESHOLD = 0.0
ATTRIBUTE_COMPARISON_THRESHOLD = 0.0


def setup_models():
    
    #GroundingDINO and DPT for object detection
    my_dino_model = DinoModel()
    my_dino_model = my_dino_model.to('cuda')
    my_dino_model.eval()

    #BLIP 2 VQA (with FlanT5XL) for semantic relationships
    vqa_model, vqa_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device='cuda')

    #CLIP for attributes
    clip_model, clip_preprocess = clip.load('RN50', device='cuda')

    models = {'GroundingDINO' : my_dino_model, 'vqa_model' : vqa_model, 'vqa_processors' : vqa_processors, 'clip_model' : clip_model, 'clip_preprocess' : clip_preprocess}
    return models


def preprocess_with_bboxes(image_pil, bboxes, sigma=BLUR_SIGMA):
    numI = np.array(image_pil)
    h, w, _ = numI.shape  # Get image dimensions

    # Convert normalized bbox coordinates to absolute pixel coordinates
    abs_bboxes = [(int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)) for x1, y1, x2, y2, _ in bboxes]

    # Compute the tightest bounding box containing all bboxes
    x_min = min(x1 for x1, _, _, _ in abs_bboxes)
    y_min = min(y1 for _, y1, _, _ in abs_bboxes)
    x_max = max(x2 for _, _, x2, _ in abs_bboxes)
    y_max = max(y2 for _, _, _, y2 in abs_bboxes)

    # Compute the average center of all bounding boxes
    center_x = int(np.mean([(x1 + x2) / 2 for x1, _, x2, _ in abs_bboxes]))
    center_y = int(np.mean([ (y1 + y2) / 2 for _, y1, _, y2 in abs_bboxes]))

    # Determine the required square size
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    crop_size = max(bbox_width, bbox_height)  # Take the tightest square that fits all bboxes

    # Ensure the crop stays within image bounds while centering as much as possible
    x_start = max(0, min(center_x - crop_size // 2, w - crop_size))
    y_start = max(0, min(center_y - crop_size // 2, h - crop_size))
    x_end = min(x_start + crop_size, w)
    y_end = min(y_start + crop_size, h)

    # Crop the image
    numIcrop = numI[y_start:y_end, x_start:x_end]

    # Compute the convex hull of all bounding boxes
    bbox_points = np.array(
        [(x1 - x_start, y1 - y_start) for x1, y1, _, _ in abs_bboxes] + 
        [(x2 - x_start, y1 - y_start) for _, y1, x2, _ in abs_bboxes] + 
        [(x2 - x_start, y2 - y_start) for _, _, x2, y2 in abs_bboxes] + 
        [(x1 - x_start, y2 - y_start) for x1, _, _, y2 in abs_bboxes]
    )

    hull = cv2.convexHull(bbox_points)  # Compute convex hull

    # Create a mask for the convex hull
    mask = np.zeros_like(numIcrop[:, :, 0], dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)

    # Blur the entire cropped image
    numIblur = cv2.GaussianBlur(numIcrop, (0, 0), sigma)

    # Apply mask: Keep inside the convex hull unchanged, use blurred image outside
    numIout = np.where(mask[:, :, None] == 255, numIcrop, numIblur)

    # Compute new bounding boxes relative to the cropped image
    new_bboxes = [((x1 - x_start) / (x_end - x_start), (y1 - y_start) / (x_end - x_start), 
                   (x2 - x_start) / (x_end - x_start), (y2 - y_start) / (y_end - y_start), z) for (x1, y1, x2, y2), (_, _, _, _, z) in zip(abs_bboxes, bboxes)]

    return Image.fromarray(numIout), new_bboxes


def query_VQA_with_bboxes(image_pil, object_name_A, bboxA, object_name_B, bboxB, predicate, models, logger):
    bboxAstr = '(' + ','.join(['%.3f'%(v) for v in bboxA[:4]]) + ')'
    bboxBstr = '(' + ','.join(['%.3f'%(v) for v in bboxB[:4]]) + ')'
    bbox_definition = 'Define a bounding-box as (x1,y1,x2,y2), where x=0 is left, x=1 is right, y=0 is top, y=1 is bottom.'
    main_question = 'Is the %s at %s %s the %s at %s?'%(object_name_A, bboxAstr, predicate, object_name_B, bboxBstr)
    prompt = 'Question: %s %s Choices: yes, no Answer:'%(bbox_definition, main_question)
    image_prepro = models['vqa_processors']["eval"](image_pil).unsqueeze(0).to('cuda')
    answer = models['vqa_model'].generate({"image": image_prepro, "prompt": prompt})
    answer = answer[0].strip().replace('.', '').replace('!', '').replace('"', '').lower()
    if answer in ['yes', 'no']:
        return {'yes' : 1, 'no' : 0}[answer]
    else:
        logger.log('SUSPICIOUS VQA ANSWER: ' + answer)
        could_be_yes = int('yes' in answer)
        could_be_no = int('no' in answer)
        assert(could_be_yes + could_be_no == 1)
        return could_be_yes


def check_scene_description(image_pil, scene_description, models, logger):
    question = 'Could the scene or style of this image be described as "%s"?'%(scene_description)
    prompt = 'Question: %s Choices: yes, no Answer:'%(question)
    image_prepro = models['vqa_processors']["eval"](image_pil).unsqueeze(0).to('cuda')
    answer = models['vqa_model'].generate({"image": image_prepro, "prompt": prompt})
    answer = answer[0].strip().replace('.', '').replace('!', '').replace('"', '').lower()
    if answer in ['yes', 'no']:
        return (question, {'yes' : 1, 'no' : 0}[answer])
    else:
        logger.log('SUSPICIOUS VQA ANSWER: ' + answer)
        could_be_yes = int('yes' in answer)
        could_be_no = int('no' in answer)
        assert(could_be_yes + could_be_no == 1)
        return (question, could_be_yes)


#will return max(CLIP(obj+attr), CLIP(attr+obj)) - CLIP(obj)
#this could be use as an ingredient in both attribute-check and attribute comparison
def measure_attribute(image_pil, object_name, attribute_name, bbox, models):
    prompt_obj = 'A photo of a %s.'%(object_name)
    prompt_attrobj = 'A photo of a %s %s.'%(attribute_name, object_name)
    prompt_objattr = 'A photo of a %s %s.'%(object_name, attribute_name)
    image_prepro = models['clip_preprocess'](image_pil).unsqueeze(0).to('cuda')
    texts = clip.tokenize([prompt_obj, prompt_attrobj, prompt_objattr]).to('cuda')
    with torch.no_grad():
        image_feats = models['clip_model'].encode_image(image_prepro)
        text_feats = models['clip_model'].encode_text(texts)
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        cossims = torch.squeeze(image_feats @ text_feats.t())
        cossims = cossims.cpu().numpy()

    return max(cossims[2], cossims[1]) - cossims[0]


#should check "<bboxA> <predicate> <bboxB>" e.g. "<bboxA> to the right of <bboxB>"
#['next to', 'on', 'in', 'above', 'below', 'to the right of', 'to the left of', 'near', 'in front of', 'behind']
#loosely based on VPEval, but I have different opinions about some things
#at some point I might change/improve some of these
def predefined_spatial(bboxA, bboxB, predicate, logger):
    #FIXME: this is a hacky way to skip self-relations so they don't "cheat" the mean-max rule
    if bboxA == bboxB:
        return np.nan
    
    #also skip anything that involves a None. The caller will pre-check taht this doesn't result in an overall NaN
    #I'm not sure how much "cheating" that could lead to but let's cross that bridge when we get there I guess
    #can't just give it a zero because it might be negated
    if bboxA is None or bboxB is None:
        return np.nan

    cXA = (bboxA[0] + bboxA[2]) / 2
    cYA = (bboxA[1] + bboxA[3]) / 2
    cXB = (bboxB[0] + bboxB[2]) / 2
    cYB = (bboxB[1] + bboxB[3]) / 2
    if predicate == 'next to':
        return int(np.fabs(cXA - cXB) > SPATIAL_DIFF_THRESHOLD)
    elif predicate == 'to the left of':
        return int(cXB - cXA > SPATIAL_DIFF_THRESHOLD)
    elif predicate == 'to the right of':
        return int(cXA - cXB > SPATIAL_DIFF_THRESHOLD)
    elif predicate in ['on', 'above']:
        return int(cYB - cYA > SPATIAL_DIFF_THRESHOLD)
    elif predicate == 'below':
        return int(cYA - cYB > SPATIAL_DIFF_THRESHOLD)
    elif predicate == 'near':
        return int(np.sqrt((cXA - cXB) ** 2 + (cYA - cYB) ** 2) < SPATIAL_DIFF_THRESHOLD)
    elif predicate == 'in':
        overlap = (min(bboxA[2], bboxB[2]) - max(bboxA[0], bboxB[0])) * (min(bboxA[3], bboxB[3]) - max(bboxA[1], bboxB[1]))
        areaA = (bboxA[2] - bboxA[0]) * (bboxB[3] - bboxB[1])
        return int(overlap / areaA > SPATIAL_CONTAINMENT_THRESHOLD)
    elif predicate == 'in front of':
        return int(bboxA[4] < bboxB[4])
    elif predicate == 'behind':
        return int(bboxB[4] < bboxA[4])
    else:
        logger('Unrecognized predefined spatial predicate "%s"'%(predicate))
        assert(False)


#return dict mapping from each object name to list of bboxes
#each bbox should be [x1, y1, x2, y2, depth] on 0-1 scale (except for depth which is on its own scale)
#for depth, I give the multiplicative inverse of what DPT gives. So what I return should give smaller depths for closer objects.
def run_object_detector(image_pil, object_names, models):
    assert(len(set(object_names)) == len(object_names))
    datum = {'image_pil' : image_pil, 'gt_labels' : object_names}
    labels, bboxes, _ = models['GroundingDINO']([datum], box_threshold=DETECTION_THRESHOLD)
    labels, bboxes = labels[0], bboxes[0]
    W, H = image_pil.size
    detections = {name : [] for name in object_names}
    for label, bbox in zip(labels, bboxes):
        assert(label in detections)
        detections[label].append([bbox[0] / W, bbox[1] / H, bbox[2] / W, bbox[3] / H, 1.0 / bbox[4]])

    return detections


#bboxA and bboxB should be single bboxes
def check_semantic_relationship_helper(image_pil, object_name_A, bboxA, object_name_B, bboxB, predicate, models, caches, logger):
    #FIXME: this is a hacky way to skip self-relations so they don't "cheat" the mean-max rule
    if bboxA == bboxB:
        return np.nan

    #also skip anything that involves a None. The caller will pre-check taht this doesn't result in an overall NaN
    #I'm not sure how much "cheating" that could lead to but let's cross that bridge when we get there I guess
    #can't just give it a zero because it might be negated
    if bboxA is None or bboxB is None:
        return np.nan

    k = (object_name_A, tuple(bboxA), object_name_B, tuple(bboxB), predicate, image_pil.size)
    if k in caches['semantic_relationship_cache']:
        return caches['semantic_relationship_cache'][k]


    imageFocus_pil, [new_bboxA, new_bboxB] = preprocess_with_bboxes(image_pil, [bboxA, bboxB])
    answer = query_VQA_with_bboxes(imageFocus_pil, object_name_A, new_bboxA, object_name_B, new_bboxB, predicate, models, logger)
    caches['semantic_relationship_cache'][k] = answer
    return answer


def check_attribute_comparison_relationship_helper(image_pil,object_name_A,bboxA,object_name_B,bboxB,attribute_name,crocodile,models,caches):
    #FIXME: this is a hacky way to skip self-relations so they don't "cheat" the mean-max rule
    if bboxA == bboxB:
        return np.nan
    
    #also skip anything that involves a None. The caller will pre-check taht this doesn't result in an overall NaN
    #I'm not sure how much "cheating" that could lead to but let's cross that bridge when we get there I guess
    #can't just give it a zero because it might be negated
    if bboxA is None or bboxB is None:
        return np.nan
    
    assert(crocodile in ['>', '<'])
    
    k = (object_name_A, tuple(bboxA), object_name_B, tuple(bboxB), attribute_name, crocodile, image_pil.size)
    if k in caches['attribute_comparison_relationship_cache']:
        return caches['attribute_comparison_relationship_cache'][k]
    
    imageA_pil, [new_bboxA] = preprocess_with_bboxes(image_pil, [bboxA])
    imageB_pil, [new_bboxB] = preprocess_with_bboxes(image_pil, [bboxB])
    attr_val_A = measure_attribute(imageA_pil, object_name_A, attribute_name, new_bboxA, models)
    attr_val_B = measure_attribute(imageB_pil, object_name_B, attribute_name, new_bboxB, models)
    if crocodile == '>':
        answer = int(attr_val_A - attr_val_B > ATTRIBUTE_COMPARISON_THRESHOLD)
    elif crocodile == '<':
        answer = int(attr_val_B - attr_val_A > ATTRIBUTE_COMPARISON_THRESHOLD)
    else:
        assert(False)

    caches['attribute_comparison_relationship_cache'][k] = answer
    return answer


def check_attribute_helper(image_pil, object_name, bbox, attribute_name, models, caches):
    #skip anything that involves a None. The caller will pre-check taht this doesn't result in an overall NaN
    #I'm not sure how much "cheating" that could lead to but let's cross that bridge when we get there I guess
    #can't just give it a zero because it might be negated
    if bbox is None:
        return np.nan

    k = (object_name, tuple(bbox), attribute_name, image_pil.size)
    if k in caches['attribute_cache']:
        return caches['attribute_cache'][k]

    imageAttr_pil, [new_bbox] = preprocess_with_bboxes(image_pil, [bbox])
    attr_val = measure_attribute(imageAttr_pil, object_name, attribute_name, new_bbox, models)
    answer = int(attr_val > ATTRIBUTE_THRESHOLD)
    caches['attribute_cache'][k] = answer
    return answer


def make_bbox_description_onebox(image_pil, object_name, bbox):
    W, H = image_pil.size
    return '%s(%d,%d)'%(object_name, W * (bbox[0] + bbox[2]) / 2, H * (bbox[1] + bbox[3]) / 2)


def make_bbox_description(image_pil, object_name, bbox):
    if bbox is None or bbox == []:
        return '%s(None)'%(object_name)
    elif isinstance(bbox[0], list):
        if len(bbox) == 1:
            return make_bbox_description_onebox(image_pil, object_name, bbox[0])
        else:
            return '%ss[%d]'%(object_name, len(bbox))
    else:
        return make_bbox_description_onebox(image_pil, object_name, bbox)


#bboxA and bboxB could be single bboxes or lists of bboxes
#if there are multiple, we will take 0.5*avg_{subject in subjects}[max_{object in objects} answer(subject, object)] + 0.5*viceversa
#negate will subtract this final answer from 1
#return log entry
def check_semantic_relationship(image_pil, object_name_A, bboxA, object_name_B, bboxB, predicate, models, caches, logger, negate=False):
    #make description (ahead of time)
    A_desc = make_bbox_description(image_pil, object_name_A, bboxA)
    B_desc = make_bbox_description(image_pil, object_name_B, bboxB)
    if negate:
        description = 'Check semantic relationship: negate(%s %s %s)'%(A_desc, predicate, B_desc)
    else:
        description = 'Check semantic relationship: %s %s %s'%(A_desc, predicate, B_desc)

    if bboxA is None or bboxB is None or bboxA == [] or bboxB == [] or (isinstance(bboxA[0], list) and all([bbb is None for bbb in bboxA])) or (isinstance(bboxB[0], list) and all([bbb is None for bbb in bboxB])):
        return (description, 0)

    A_is_list = isinstance(bboxA[0], list)
    B_is_list = isinstance(bboxB[0], list)
    if not (A_is_list or B_is_list):
        answer = check_semantic_relationship_helper(image_pil, object_name_A, bboxA, object_name_B, bboxB, predicate, models, caches, logger)
        assert(not np.isnan(answer))
        if negate:
            answer = 1 - answer

        return (description, answer)
    else:
        if not A_is_list:
            bboxA = [bboxA]

        if not B_is_list:
            bboxB = [bboxB]

        answer_matrix = np.zeros((len(bboxA), len(bboxB)))
        for i in range(len(bboxA)):
            for j in range(len(bboxB)):
                 answer_matrix[i,j] = check_semantic_relationship_helper(image_pil, object_name_A, bboxA[i], object_name_B, bboxB[j], predicate, models, caches, logger)

        answer = 0.5 * np.nanmean(np.nanmax(answer_matrix, axis=0)) + 0.5 * np.nanmean(np.nanmax(answer_matrix, axis=1))
        assert(not np.isnan(answer))
        if negate:
            answer = 1 - answer

        return (description, answer)


#bboxA and bboxB could be single bboxes or lists of bboxes
#if there are multiple, we will take 0.5*avg_{subject in subjects}[max_{object in objects} answer(subject, object)] + 0.5*viceversa
#negate will subtract this final answer from 1
#only taking image and object names to make for better logging, they're not actually necessary for the algo
#return log entry
def check_predefined_spatial_relationship(image_pil, object_name_A, bboxA, object_name_B, bboxB, predicate, models, logger, negate=False):
    #make description (ahead of time)
    A_desc = make_bbox_description(image_pil, object_name_A, bboxA)
    B_desc = make_bbox_description(image_pil, object_name_B, bboxB)
    if negate:
        description = 'Check spatial relationship: negate(%s %s %s)'%(A_desc, predicate, B_desc)
    else:
        description = 'Check spatial relationship: %s %s %s'%(A_desc, predicate, B_desc)

    if bboxA is None or bboxB is None or bboxA == [] or bboxB == [] or (isinstance(bboxA[0], list) and all([bbb is None for bbb in bboxA])) or (isinstance(bboxB[0], list) and all([bbb is None for bbb in bboxB])):
        return (description, 0)

    A_is_list = isinstance(bboxA[0], list)
    B_is_list = isinstance(bboxB[0], list)
    if not (A_is_list or B_is_list):
        answer = predefined_spatial(bboxA, bboxB, predicate, logger)
        assert(not np.isnan(answer))
        if negate:
            answer = 1 - answer

        return (description, answer)
    else:
        if not A_is_list:
            bboxA = [bboxA]

        if not B_is_list:
            bboxB = [bboxB]

        answer_matrix = np.zeros((len(bboxA), len(bboxB)))
        for i in range(len(bboxA)):
            for j in range(len(bboxB)):
                 answer_matrix[i,j] = predefined_spatial(bboxA[i], bboxB[j], predicate, logger)

        answer = 0.5 * np.nanmean(np.nanmax(answer_matrix, axis=0)) + 0.5 * np.nanmean(np.nanmax(answer_matrix, axis=1))
        assert(not np.isnan(answer))
        if negate:
            answer = 1 - answer

        return (description, answer)


#bboxA and bboxB could be single bboxes or lists of bboxes
#if there are multiple, we will take 0.5*avg_{subject in subjects}[max_{object in objects} answer(subject, object)] + 0.5*viceversa
#negate will subtract this final answer from 1
#return log entry
def check_attribute_comparison_relationship(image_pil,object_name_A,bboxA,object_name_B,bboxB,attribute_name,crocodile,models,caches):
    #make description (ahead of time)
    A_desc = make_bbox_description(image_pil, object_name_A, bboxA)
    B_desc = make_bbox_description(image_pil, object_name_B, bboxB)
    description = 'Check attribute-comparison relationship: %s %s{%s} %s'%(A_desc, crocodile, attribute_name, B_desc)
    if bboxA is None or bboxB is None or bboxA == [] or bboxB == [] or (isinstance(bboxA[0], list) and all([bbb is None for bbb in bboxA])) or (isinstance(bboxB[0], list) and all([bbb is None for bbb in bboxB])):
        return (description, 0)

    A_is_list = isinstance(bboxA[0], list)
    B_is_list = isinstance(bboxB[0], list)
    if not (A_is_list or B_is_list):
        answer = check_attribute_comparison_relationship_helper(image_pil, object_name_A, bboxA, object_name_B, bboxB, attribute_name, crocodile, models, caches)
        assert(not np.isnan(answer))
        return (description, answer)
    else:
        if not A_is_list:
            bboxA = [bboxA]

        if not B_is_list:
            bboxB = [bboxB]

        answer_matrix = np.zeros((len(bboxA), len(bboxB)))
        for i in range(len(bboxA)):
            for j in range(len(bboxB)):
                 answer_matrix[i,j] = check_attribute_comparison_relationship_helper(image_pil, object_name_A, bboxA[i], object_name_B, bboxB[j], attribute_name, crocodile, models, caches)

        answer = 0.5 * np.nanmean(np.nanmax(answer_matrix, axis=0)) + 0.5 * np.nanmean(np.nanmax(answer_matrix, axis=1))
        assert(not np.isnan(answer))

        return (description, answer)


#bbox can be either single bbox or list
#return answer (or average of answers if multiple)
def check_attribute(image_pil, object_name, bbox, attribute_name, models, caches, negate=False):
    bbox_desc = make_bbox_description(image_pil, object_name, bbox)
    if negate:
        description = 'Check negate(attribute "%s") for %s'%(attribute_name, bbox_desc)
    else:
        description = 'Check attribute "%s" for %s'%(attribute_name, bbox_desc)

    if bbox is None or bbox == [] or (isinstance(bbox[0], list) and all([bbb is None for bbb in bbox])):
        return (description, 0)

    is_list = isinstance(bbox[0], list)
    if not is_list:
        answer = check_attribute_helper(image_pil, object_name, bbox, attribute_name, models, caches)
        if negate:
            answer = 1 - answer

        return (description, answer)

    answers = []
    for i in range(len(bbox)):
        answer = check_attribute_helper(image_pil, object_name, bbox[i], attribute_name, models, caches)
        answers.append(answer)

    answer = np.mean(answers)
    if negate:
        answer = 1 - answer

    return (description, answer)


if __name__ == '__main__':
    models = setup_models()
    image_pil = Image.open('cat_chasing_cat.jpeg').convert('RGB')
    detections = run_object_detector(image_pil, ['cat', 'bush'], models)
    print(detections)
    imageFocusCats_pil, new_cat_bboxes = preprocess_with_bboxes(image_pil, detections['cat'])
    draw_detections(imageFocusCats_pil, {'cat' : new_cat_bboxes}, 'purr.png')
    ans01 = query_VQA_with_bboxes(imageFocusCats_pil, 'cat', new_cat_bboxes[0], 'cat', new_cat_bboxes[1], 'chasing', models)
    ans10 = query_VQA_with_bboxes(imageFocusCats_pil, 'cat', new_cat_bboxes[1], 'cat', new_cat_bboxes[0], 'chasing', models)
    print((new_cat_bboxes[0][-1], new_cat_bboxes[1][-1], ans01))
    print((new_cat_bboxes[1][-1], new_cat_bboxes[0][-1], ans10))
    for meowdex in [0,1]:
        imageOneKitty_pil, [bbox_one_kitty] = preprocess_with_bboxes(image_pil, [detections['cat'][meowdex]])
        for cattribute in ['siamese', 'tabby', 'running', 'orange', 'sleeping', 'with a bowtie', 'wearing a bowtie']:
            attr_val = measure_attribute(imageOneKitty_pil, 'cat', cattribute, bbox_one_kitty, models)
            print('Measure attribute "%s" for kitty at depth %.3f: %f'%(cattribute, detections['cat'][meowdex][-1], attr_val))
