import tensorflow as tf

def flatten(tensor):
    B,H,W,C = tensor.shape
    return tf.reshape(tensor, (B,-1,C))


def sigmoid_focalloss(pred, target, from_logisit, alpha = 0.25, gamma = 2.0, flt_min = 1e-9):
    if from_logisit:
        prob = tf.sigmoid(pred)
    else:
        prob = pred
    alpha = tf.constant(alpha,dtype=tf.float32)
    gamma = tf.constant(gamma,dtype=tf.float32)
    classes_num = prob.shape[-1]
    #batch_size = prob.shape[0]


    p_value = (1-prob) ** gamma * tf.math.log( tf.maximum(prob, flt_min) )
    n_value = prob ** gamma * tf.math.log( tf.maximum(1 - prob, flt_min) )

    class_range = tf.range(0,classes_num,dtype=tf.float32)
    class_range = tf.reshape(class_range,(1,1,classes_num) )
    p_mask = tf.cast(tf.equal(class_range, target),dtype=tf.float32)
    n_mask = tf.cast(tf.not_equal(class_range, target),dtype=tf.float32)

    loss_value = p_mask * alpha * p_value + n_mask * (1-alpha) * n_value
    if tf.reduce_sum(p_mask) > 0:
        loss_value = tf.reduce_sum(loss_value) / tf.reduce_sum(p_mask)
    else:
        loss_value = tf.reduce_sum(loss_value)
    return -loss_value


def iou_loss(preds, targets, iou_type="iou"):
    assert len(preds.shape) == 2 and len(targets.shape) == 2
    assert preds.shape[1] == 4 and targets.shape[1] == 4

   # preds = tf.exp(preds)
    pred_left = preds[:,0]
    pred_top = preds[:,1]
    pred_right = preds[:,2]
    pred_bottom = preds[:,3]

    target_left, target_top, target_right, target_bottom = targets[:,0], targets[:,1], targets[:,2],targets[:,3]

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
    else: ## 'giou':
        w_union = tf.maximum(target_left, pred_left) + tf.maximum(target_right, pred_right)
        h_union = tf.maximum(target_top, pred_top) + tf.maximum(target_bottom, pred_bottom)
        area_g_union = w_union * h_union
        giou = ious - (area_g_union - area_union) / area_g_union
        loss_value =  1 - giou

    return tf.reduce_sum(loss_value)



def calc_loss(preds, targets, weights = None):
    preds_class, preds_reg, preds_centerness = [], [], []
    for stage in range(5):
        pred_class, pred_reg, pred_centerness = preds[f'stage_{stage}_cls'],preds[f'stage_{stage}_reg'],preds[f'stage_{stage}_center']
        preds_class.append(flatten(pred_class))
        preds_reg.append(flatten(pred_reg))
        preds_centerness.append(flatten(pred_centerness))
    preds_class = tf.concat(preds_class,axis=1)
    preds_centerness = tf.concat(preds_centerness,axis=1)
    preds_reg = tf.concat(preds_reg,axis=1)

    loss_class = sigmoid_focalloss(preds_class,targets[0],from_logisit=True)


    preds_reg_flatten = tf.reshape(preds_reg,(-1,4)) # (N,4)
    targets_reg_flatten = tf.reshape(targets[1],(-1,4)) # (N,4)
    #indices_pos = tf.where(tf.reshape(targets[0],(-1,1)) > 0)[:,0] #(N)
    indices_pos = tf.where(tf.greater( tf.reshape(targets[2],(-1,1)), 0))[:,0] #(N)
    num_pos = indices_pos.shape[0]
    if num_pos > 0:
        loss_reg = iou_loss(tf.gather(preds_reg_flatten,indices_pos,axis=0), tf.gather(targets_reg_flatten,indices_pos,axis=0), iou_type="iou") / num_pos
    else:
        loss_reg = tf.constant(0.0,dtype=tf.float32)

    loss_centerness = tf.keras.losses.binary_crossentropy(targets[2],preds_centerness,from_logits=True)
    loss_centerness = tf.reduce_sum(loss_centerness)  / tf.reduce_sum( tf.cast(tf.greater(targets[2],0.0),tf.float32) )
    ##loss_centerness = tf.reduce_sum(loss_centerness)  / tf.reduce_sum( targets[2] ) ###########!!!!!!!!!!!!!!!!!!!

    if weights is None:
        loss_all = tf.add_n([loss_class , loss_reg , loss_centerness])
    else:
        loss_all = weights['class'] * loss_class + weights['reg'] * loss_reg + weights["centerness"] * loss_centerness

    return loss_all, loss_class, loss_reg, loss_centerness