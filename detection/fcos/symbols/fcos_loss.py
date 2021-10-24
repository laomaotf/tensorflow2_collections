import tensorflow as tf

INF = 999999
STRIDES = [8, 16, 32, 64, 128]  # stride for each scale
MINMAX_RANGE = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]]  # size supported by each scale


def flatten(tensor):
    B, H, W, C = tensor.shape
    return tf.reshape(tensor, (B, -1, C))


def sigmoid_focalloss(pred, target, from_logisit, alpha=0.25, gamma=2.0, flt_min=1e-9):
    if from_logisit:
        prob = tf.sigmoid(pred)
    else:
        prob = pred
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)
    classes_num = prob.shape[-1]
    # batch_size = prob.shape[0]

    p_value = (1 - prob) ** gamma * tf.math.log(tf.maximum(prob, flt_min))
    n_value = prob ** gamma * tf.math.log(tf.maximum(1 - prob, flt_min))

    class_range = tf.range(0, classes_num, dtype=tf.float32)
    class_range = tf.reshape(class_range, (1, 1, classes_num))
    p_mask = tf.cast(tf.equal(class_range, target), dtype=tf.float32)
    n_mask = tf.cast(tf.not_equal(class_range, target), dtype=tf.float32)

    loss_value = p_mask * alpha * p_value + n_mask * (1 - alpha) * n_value
    if tf.reduce_sum(p_mask) > 0:
        loss_value = tf.reduce_sum(loss_value) / tf.reduce_sum(p_mask)
    else:
        loss_value = tf.reduce_sum(loss_value)
    return -loss_value


def iou_loss(preds, targets, iou_type="iou"):
    assert len(preds.shape) == 2 and len(targets.shape) == 2
    assert preds.shape[1] == 4 and targets.shape[1] == 4

    # preds = tf.exp(preds)
    pred_left = preds[:, 0]
    pred_top = preds[:, 1]
    pred_right = preds[:, 2]
    pred_bottom = preds[:, 3]

    target_left, target_top, target_right, target_bottom = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]

    area_target = (target_left + target_right) * (target_top + target_bottom)
    area_pred = (pred_left + pred_right) * (pred_top + pred_bottom)

    w_intersection = tf.minimum(target_left, pred_left) + tf.minimum(target_right, pred_right)
    h_intersection = tf.minimum(target_top, pred_top) + tf.minimum(target_bottom, pred_bottom)
    # w_intersection = tf.clip_by_value(w_intersection,0,10009)
    # h_intersection = tf.clip_by_value(h_intersection,0,10000)
    area_intersection = w_intersection * h_intersection
    area_union = area_pred + area_target - area_intersection

    ious = (area_intersection + 1) / (area_union + 1)

    if iou_type == "linear-iou":
        loss_value = 1 - ious
    elif iou_type == "iou":
        loss_value = -tf.math.log(ious)
    else:  ## 'giou':
        w_union = tf.maximum(target_left, pred_left) + tf.maximum(target_right, pred_right)
        h_union = tf.maximum(target_top, pred_top) + tf.maximum(target_bottom, pred_bottom)
        area_g_union = w_union * h_union
        giou = ious - (area_g_union - area_union) / area_g_union
        loss_value = 1 - giou

    return tf.reduce_sum(loss_value)



def get_original_locations(sizes, scales, minmax_ranges=None):
    assert len(sizes) == len(scales)
    if minmax_ranges is None:
        minmax_ranges = [(-1, -1) for _ in range(len(sizes))]
    else:
        assert len(sizes) == len(minmax_ranges)
    locations, locations_range = [], []
    for (h, w), scale, minmax in zip(sizes, scales, minmax_ranges):
        minmax = tf.reshape(minmax, (1, 2))
        xs, ys = tf.range(0, w, dtype=tf.float32), tf.range(0, h, dtype=tf.float32)
        xs, ys = tf.round(tf.meshgrid(xs, ys))
        xs, ys = tf.reshape(xs, (-1, 1)) * scale + scale / 2, tf.reshape(ys, (-1, 1)) * scale + scale / 2
        locations_one_scale = tf.concat([xs, ys], axis=1)
        locations_range_one_scale = [minmax for _ in range(len(xs))]
        locations.append(locations_one_scale)
        locations_range.extend(locations_range_one_scale)
    return tf.concat(locations, axis=0), tf.concat(locations_range, axis=0)


def gen_fcos_groundtruth(image_input, targets):
    B, H, W, _ = image_input.shape
    N = targets.shape[1]
    ####################################################################
    # map point under each scale to original image size
    sizes = []
    for scale in STRIDES:
        sizes.append((H // scale, W // scale))
    locations, locations_range = get_original_locations(sizes, STRIDES, MINMAX_RANGE)
    locations = tf.expand_dims(locations, axis=0)
    locations_range = tf.expand_dims(locations_range, axis=0)

    xs, ys = locations[0, :, 0], locations[0, :, 1]
    classes = targets[:, :, -1]
    bboxes = targets[:, :, 0:-1]
    # areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes] #area of each gt
    # locations2gt_area = tf.reshape(areas,(1,-1))
    # locations2gt_area = tf.repeat(locations2gt_area,repeats=len(locations),axis=0)
    locations2gt_area = (bboxes[:, :, 2] - bboxes[:, :, 0]) * (bboxes[:, :, 3] - bboxes[:, :, 1])
    locations2gt_area = tf.expand_dims(locations2gt_area, axis=1)
    ###############################################################
    # reg target of each location
    # left = xs[:,None] - bboxes[:,0][None]
    # top = ys[:,None] - bboxes[:,1][None]
    # right = bboxes[:,2][None] - xs[:,None]
    # bottom = bboxes[:,3][None] - ys[:,None]

    left = tf.reshape(xs, (1, -1, 1)) - tf.expand_dims(bboxes[:, :, 0], axis=1)  # (B,location_num,gt_num)
    top = tf.reshape(ys, (1, -1, 1)) - tf.expand_dims(bboxes[:, :, 1], axis=1)
    right = tf.expand_dims(bboxes[:, :, 2], axis=1) - tf.reshape(xs, (1, -1, 1))
    bottom = tf.expand_dims(bboxes[:, :, 3], axis=1) - tf.reshape(ys, (1, -1, 1))

    reg_target = tf.stack([left, top, right, bottom], axis=-1)
    is_in_bbox = (left > 0) & (top > 0) & (right > 0) & (bottom > 0)  # reg gt should be positive
    reg_target_max = tf.reduce_max(reg_target, axis=-1)
    # is_in_scale = (reg_target_max >= locations_range[:,[0]]) & (reg_target_max < locations_range[:,[1]]) # max rag should be in range
    is_in_scale = (reg_target_max >= tf.cast(tf.expand_dims(locations_range[:, :, 0], axis=-1), tf.float32)) & (
            reg_target_max < tf.cast(tf.expand_dims(locations_range[:, :, 1], axis=-1), tf.float32))

    # reg_target[is_in_scale == False] = INF
    # reg_target[is_in_bbox == False] = INF
    #
    # locations2gt_area[is_in_bbox==False] = INF
    # locations2gt_area[is_in_scale==False] = INF
    reg_target = tf.where(tf.expand_dims(is_in_scale, axis=-1), reg_target, INF)
    reg_target = tf.where(tf.expand_dims(is_in_bbox, axis=-1), reg_target, INF)

    locations2gt_area = tf.where(is_in_bbox, locations2gt_area, INF)
    locations2gt_area = tf.where(is_in_scale, locations2gt_area, INF)

    ######################################################
    # Resolve one point corresponds to more than one gt
    locations2gt_area_min = tf.reduce_min(locations2gt_area, axis=-1)
    locations2gt_index = tf.argmin(locations2gt_area, axis=-1)

    #########################################################
    # generate FCOS groundtruth
    # locations2class = tf.reshape(classes, (1, -1)).repeat(len(locations2gt_index), axis=0)
    # locations2class = locations2class[range(len(locations2gt_index)), locations2gt_index]
    # locations2class[locations2gt_area_min == INF] = -1  # only class >= 0 is positive
    locations2class = tf.expand_dims(classes, axis=1)
    locations2class = tf.repeat(locations2class, locations2gt_index.shape[1], axis=1)
    locations2gt_index_list = tf.reshape(locations2gt_index, (-1, 1))
    K = tf.range(0, len(locations2gt_index_list))
    K = tf.cast(K, tf.int64)
    K = tf.reshape(K, (-1, 1))
    indices = tf.concat([K, locations2gt_index_list], axis=-1)
    # indices = [[k,v] for k,v in enumerate(locations2gt_index_list)]
    locations2class = tf.gather_nd(tf.reshape(locations2class, (-1, N)), indices=indices)
    locations2class = tf.reshape(locations2class, (B, -1))
    locations2class = tf.where(locations2gt_area_min == INF, -1, locations2class)
    locations2class = tf.expand_dims(locations2class,axis=-1) #(B,num_anchor,1)

    # locations2reg = reg_target[range(len(locations)), locations2gt_index]  # reg gt of each location

    zeros = tf.constant(0, tf.int64, (len(locations2gt_index_list), 1))
    ones = tf.constant(1, tf.int64, (len(locations2gt_index_list), 1))
    twos = tf.constant(2, tf.int64, (len(locations2gt_index_list), 1))
    threes = tf.constant(3, tf.int64, (len(locations2gt_index_list), 1))
    indices_x0 = tf.concat([K, tf.reshape(locations2gt_index_list, (-1, 1)), zeros], axis=-1)
    indices_y0 = tf.concat([K, tf.reshape(locations2gt_index_list, (-1, 1)), ones], axis=-1)
    indices_x1 = tf.concat([K, tf.reshape(locations2gt_index_list, (-1, 1)), twos], axis=-1)
    indices_y1 = tf.concat([K, tf.reshape(locations2gt_index_list, (-1, 1)), threes], axis=-1)
    locations2reg_x0 = tf.gather_nd(tf.reshape(reg_target, (-1, N, 4)), indices=indices_x0)
    locations2reg_x0 = tf.reshape(locations2reg_x0, (B, -1, 1))
    locations2reg_y0 = tf.gather_nd(tf.reshape(reg_target, (-1, N, 4)), indices=indices_y0)
    locations2reg_y0 = tf.reshape(locations2reg_y0, (B, -1, 1))
    locations2reg_x1 = tf.gather_nd(tf.reshape(reg_target, (-1, N, 4)), indices=indices_x1)
    locations2reg_x1 = tf.reshape(locations2reg_x1, (B, -1, 1))
    locations2reg_y1 = tf.gather_nd(tf.reshape(reg_target, (-1, N, 4)), indices=indices_y1)
    locations2reg_y1 = tf.reshape(locations2reg_y1, (B, -1, 1))
    locations2reg = tf.concat([locations2reg_x0, locations2reg_y0, locations2reg_x1, locations2reg_y1], axis=-1)

    # locations2reg_x_min = tf.math.minimum(locations2reg[:, [0, 2]], axis=-1, keepdims=False)
    # locations2reg_x_max = tf.math.maximum(locations2reg[:, [0, 2]], axis=-1, keepdims=False)
    # locations2reg_y_min = tf.math.minimum(locations2reg[:, [1, 3]], axis=-1, keepdims=False)
    # locations2reg_y_max = tf.math.maximum(locations2reg[:, [1, 3]], axis=-1, keepdims=False)
    locations2reg_x_min = tf.math.minimum(locations2reg[:, :, 0], locations2reg[:, :, 2])
    locations2reg_x_max = tf.math.maximum(locations2reg[:, :, 0], locations2reg[:, :, 2])
    locations2reg_y_min = tf.math.minimum(locations2reg[:, :, 1], locations2reg[:, :, 3])
    locations2reg_y_max = tf.math.maximum(locations2reg[:, :, 1], locations2reg[:, :, 3])
    locations2reg_y_min, locations2reg_x_min = tf.clip_by_value(locations2reg_y_min, 0.001, INF), tf.clip_by_value(
        locations2reg_x_min, 0.001, INF)
    locations2centerness = tf.math.sqrt(
        (locations2reg_x_min / locations2reg_x_max) * (locations2reg_y_min / locations2reg_y_max))
    #locations2centerness[locations2gt_area_min == INF] = 0
    locations2centerness = tf.where(locations2gt_area_min == INF, 0, locations2centerness)
    locations2centerness = tf.expand_dims(locations2centerness,axis=-1)
    #locations2centerness = locations2centerness[:, None]
    #locations2class = locations2class[:, None]

    if 0:
        import cv2
        import numpy as np
        image_index = 1
        image_src = image_input[image_index].numpy()
        data_mean = np.array([103.939, 116.779, 123.68]).astype(np.float32).reshape(1, 1, 3)
        trt_reg = locations2reg.numpy()[image_index]
        trt_class = locations2class.numpy()[image_index]
        index = 0
        for (h,w) in sizes:
            image_scale = (image_src + data_mean).astype(np.uint8)
            s = W/w #asset
            offset = s/2
            for y in range(h):
                for x in range(w):
                    if trt_class[index] > 0:
                        left,top,right,bottom = trt_reg[index,:]
                        left, top = int(x * s - left + offset), int(y*s - top + offset)
                        right, bottom = int(x * s + right + offset), int(y*s + bottom + offset)
                        cv2.rectangle(image_scale,(left,top),(right,bottom),(255,0,0),3)
                        cv2.circle(image_scale,(int(x*W/w+offset), int(y*H/h+offset)), 2, (0,255,0),-1)
                    index += 1
            cv2.imwrite(f"{h}x{w}.jpg",image_scale)


    return locations2class, locations2reg, locations2centerness


def calc_loss(preds, images, targets, weights=None):
    gt_class, gt_reg, gt_centernesss = gen_fcos_groundtruth(images, targets)
    preds_class, preds_reg, preds_centerness = [], [], []
    for stage in range(5):
        pred_class, pred_reg, pred_centerness = preds[f'stage_{stage}_cls'], preds[f'stage_{stage}_reg'], preds[
            f'stage_{stage}_center']
        preds_class.append(flatten(pred_class))
        preds_reg.append(flatten(pred_reg))
        preds_centerness.append(flatten(pred_centerness))
    preds_class = tf.concat(preds_class, axis=1)
    preds_centerness = tf.concat(preds_centerness, axis=1)
    preds_reg = tf.concat(preds_reg, axis=1)

    loss_class = sigmoid_focalloss(preds_class, gt_class, from_logisit=True)

    preds_reg_flatten = tf.reshape(preds_reg, (-1, 4))  # (N,4)
    targets_reg_flatten = tf.reshape(gt_reg, (-1, 4))  # (N,4)
    # indices_pos = tf.where(tf.reshape(targets[0],(-1,1)) > 0)[:,0] #(N)
    indices_pos = tf.where(tf.greater(tf.reshape(gt_centernesss, (-1, 1)), 0))[:, 0]  # select pos by centerness map
    num_pos = indices_pos.shape[0]
    if num_pos > 0:
        loss_reg = iou_loss(tf.gather(preds_reg_flatten, indices_pos, axis=0),
                            tf.gather(targets_reg_flatten, indices_pos, axis=0), iou_type="iou") / num_pos
    else:
        loss_reg = tf.constant(0.0, dtype=tf.float32)

    loss_centerness = tf.keras.losses.binary_crossentropy(gt_centernesss, preds_centerness, from_logits=True)
    loss_centerness = tf.reduce_sum(loss_centerness) / tf.reduce_sum(tf.cast(tf.greater(gt_centernesss, 0.0), tf.float32))
    ##loss_centerness = tf.reduce_sum(loss_centerness)  / tf.reduce_sum( targets[2] ) ###########!!!!!!!!!!!!!!!!!!!

    if weights is None:
        loss_all = tf.add_n([loss_class, loss_reg, loss_centerness])
    else:
        loss_all = weights['class'] * loss_class + weights['reg'] * loss_reg + weights["centerness"] * loss_centerness

    return loss_all, loss_class, loss_reg, loss_centerness
