from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train, filter_boxes
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
from tensorflow.python.saved_model import tag_constants
from functools import reduce

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_string('weights', './data/yolov4.weights', 'pretrained weights')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_integer('input_size', 416, 'define input size of export model')
flags.DEFINE_float('score_thres', 0.2, 'define score threshold')
flags.DEFINE_string('framework', 'tf', 'define what framework do you want to convert (tf, trt, tflite)')

flags.DEFINE_integer('k_step', 4, 'ADMM step number')
flags.DEFINE_integer('all_percent', 30, 'want to delete percentile')
flags.DEFINE_integer('first_stage_epochs', 1, 'ADMM step(W update) with freeze layer not training epoch')
flags.DEFINE_integer('second_stage_epochs', 1, 'ADMM step(W update) with freeze layer training epoch')
flags.DEFINE_integer('retraining_first_stage_epochs', 4, 'After ADMM step, retraining epoch')
flags.DEFINE_integer('retraining_second_stage_epochs', 6, 'After ADMM step, retraining epoch')
#flags.DEFINE_float('parameter_lambda', 0.0000000001, 'ADMM step(W update) with freeze layer training epoch')
flags.DEFINE_float('parameter_rho', 0.5, 'ADMM step(W update) with freeze layer training epoch')


def admm_pruning():
    k_step = FLAGS.k_step
    all_percent = FLAGS.all_percent

    trainset = Dataset(FLAGS, is_training=True)
    logdir = "./data/log"
    isfreeze = False
    steps_per_epoch = len(trainset)
    print('steps_per_epoch : {}'.format(steps_per_epoch))
    first_stage_epochs = FLAGS.first_stage_epochs
    second_stage_epochs = FLAGS.second_stage_epochs
    retraining_first_stage_epochs = FLAGS.retraining_first_stage_epochs
    retraining_second_stage_epochs = FLAGS.retraining_second_stage_epochs
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64) # step 개수 count
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch # total_steps에서 얼만큼 warmup 할건지 step 수
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch # ADMM에서 k=1-step당 step 수
    retraining_total_steps = (retraining_first_stage_epochs + retraining_second_stage_epochs) * steps_per_epoch

    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)
    optimizer = tf.keras.optimizers.Adam() # admm_step에 사용
    optimizer_retraining = tf.keras.optimizers.Adam() # retraining에 사용(위의 optimizer를 사용하였더니 기존 정보를 참고하는듯?)

    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir) # metric에 관한 log 저장

    #모델 생성
    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)

    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    if FLAGS.tiny:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
    layers = model.layers

    Z_dict, U_dict = make_dict(model) # ADMM을 적용하기 위한 Z, U 값의 초기화(Z는 Weight에서 projection시킨 결과, U는 0 값으로 사용)


    def train_step(image_data, target, Z_dict=None, U_dict=None, is_admm =True, dict_nzidx=None, k=0, epoch=0): # W 학습
        """
        Weight 최적화
        :param image_data: Input
        :param target: Output
        :param Z_dict: ADMM step에 사용되는 varaible(Weight가 Z_dict-U_dict과 유사하게 훈련하도록 사용. Z_dict은 projection시킨 값)
        :param U_dict: ADMM step에 사용되는 varaible
        :param is_admm: True일 때 이전 단계의 Z_dict, U_dict을 통해 Weight 최적화, False일 때 gradient의 dict_nzidx를 참고하여 특정 부분을 0으로 바꾸고 Weight 최적화
        :param dict_nzidx: True(0이 아닌 weight)/False(0인 weight) [5차원(list로 씌워진 상태), key는 layer.name으로 구성]
        """
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS,
                                          IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            origin_loss = giou_loss + conf_loss + prob_loss
            weight_loss = 0
            admm_loss = 0

            if is_admm: # admm일때만 Z,U를 통해 최적화, retrain일때는 origin_loss만 사용
                for i,layer in enumerate(layers):
                    # conv2d_93, 101, 109는 frozen이라서 admm_step에서 제외
                    # convolution layer만 admm_pruning 적용
                    if 'conv2d' in layer.name and layer.name not in ['conv2d_93', 'conv2d_101','conv2d_109']:
                        # weight_loss = weight_loss + 100000*tf.nn.l2_loss(layer.weights[0])  # f(W)에서의 W에 대한 l2-loss. 전체layer로하면 batchnormal도껴서안됨
                        admm_loss = admm_loss + tf.nn.l2_loss([layer.weights[0] - Z_dict[layer.name][0] + U_dict[layer.name][0]])  # 7번식의 W-Z+U. get_weights는 값만 불러오는듯. layer.weights는 모델의 weight를 참조

            # 7번식 loss 완료
            # total_loss = origin_loss+ FLAGS.parameter_lambda*weight_loss + FLAGS.parameter_rho*admm_loss # 각 parameter를 곱해줌(lambda, rho)
            total_loss = origin_loss + FLAGS.parameter_rho * admm_loss  # 각 parameter를 곱해줌(lambda, rho). admm이 아닐때는 admm_loss가 0값을 가짐(origin_loss만 남음)
            gradients = tape.gradient(total_loss, model.trainable_weights) # 기울기 계산

            if not is_admm: # retraining step에서만 쓰임. gradient들을 projection시킴(기존 코드에서 apply_prune_on_grads와 같음)
                # gradients: 각 layer의 weight에 대한 gradient [5차원(gradients[i]로 부르면 dict_nzidx[name][0]와 형태 같음)]
                for i, trainable_weight in enumerate(model.trainable_weights): # model.trainable_weights는 trainable한 layer들의 이름과 weight 정보를 가지고 있음
                    name = trainable_weight.name.split('/')[0] # conv2d/kernel:0 형식으로 나와서 split
                    if 'conv2d' in name and name not in ['conv2d_93', 'conv2d_101','conv2d_109']:
                        gradients[i] = tf.multiply(tf.cast(tf.constant(dict_nzidx[name][0]), tf.float32), gradients[i]) # dict_nzidx는 0인 부분이 False인 형태이므로 이를 0.0으로 변경(True는 1.0)하고 gradients와 곱해줌

            if is_admm:
                optimizer.apply_gradients(zip(gradients, model.trainable_weights)) # admm-step에 대한 optimizer에 기울기 적용
            else:
                optimizer_retraining.apply_gradients(zip(gradients, model.trainable_weights)) # retraining에 대한 optimizer에 기울기 적용

            if is_admm:
                # admm step
                tf.print("=> [k-step : %d/%d, epoch : %d]  STEP %4d/%4d  lr: %.6f  giou_loss: %4.2f  conf_loss: %4.2f   "
                         "prob_loss: %4.2f  admm_loss: %4.2f  weight_loss: %4.2f  origin_loss: %4.2f  total_loss: %4.2f" % (
                         k+1, k_step, epoch+1, global_steps, total_steps, optimizer.lr.numpy(),
                         giou_loss, conf_loss,
                         prob_loss, admm_loss, weight_loss, origin_loss, total_loss))


            else:
                # retraining step
                tf.print("=> STEP %4d/%4d  lr: %.6f  giou_loss: %4.2f  conf_loss: %4.2f   "
                         "prob_loss: %4.2f  total_loss: %4.2f" % (
                         global_steps, retraining_total_steps, optimizer.lr.numpy(),
                         giou_loss, conf_loss,
                         prob_loss, total_loss))


            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            else:
                # lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                #     (1 + tf.cos(((global_steps-k*steps_per_epoch*(first_stage_epochs+second_stage_epochs)) - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                # )
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (
                                            total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
                tf.summary.scalar("loss/admm_loss", admm_loss, step=global_steps)
            writer.flush()

    # ADMM step
    for k in range(k_step): # k-step(ADMM step 개수)
        global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)  # step 개수 count
        for epoch in range(first_stage_epochs + second_stage_epochs): # Weight 학습
            #print('k-step : {},  epoch : {}'.format(k+1, epoch + 1))
            if epoch < first_stage_epochs: # freeze layer(93, 101, 109였나 )는 freeze 시키고 학습 안함
                if not isfreeze:
                    isfreeze = True
                    for name in freeze_layers:
                        freeze = model.get_layer(name)
                        freeze_all(freeze)
            elif epoch >= first_stage_epochs: # freeze layer를 unfreeze 시키고 학습
                if isfreeze:
                    isfreeze = False
                    for name in freeze_layers:
                        freeze = model.get_layer(name)
                        unfreeze_all(freeze)
            for image_data, target in trainset:
                train_step(image_data, target, Z_dict, U_dict, is_admm=True, k=k, epoch=epoch) # 현재 Z,U가지고 W를 학습, admm 학습 과정이므로 is_admm = True

        for layer in layers: # 학습된 W와 현재 U를 통해 Z를 학습
            if 'conv2d' in layer.name and layer.name not in ['conv2d_93', 'conv2d_101','conv2d_109']:  # conv2d_93, 101, 109는 frozen이라 shape가 좆같dma
                Z_dict[layer.name] = layer.get_weights()[0] + U_dict[layer.name][0]  # 우리가 알고있는 커널크기(4차원, numpy)에서 list 하나로 씌워진 형태라서 나중에 다시 해줄 예정
        Z_dict = my_projection(Z_dict)

        for layer in layers: # 학습된 W와 Z 값을 통해 U를 구함
            if 'conv2d' in layer.name and layer.name not in ['conv2d_93', 'conv2d_101', 'conv2d_109']:
                U_dict[layer.name] = [U_dict[layer.name][0] + layer.get_weights()[0] - Z_dict[layer.name][0]]

    model.save_weights("./checkpoints/admm_pruning/after_admm/yolov4") # ADMM step weight 저장

        # stop_condition 안넣었는데 넣으면 좋긴 함

    # Weight pruning
    W_dict = {} # Weight들을 제거하고자 하는 layer에 대응하여 dictionary 형태로 만듬(여기서는 convolution layer만)
    for layer in layers:
        if 'conv2d' in layer.name and layer.name not in ['conv2d_93', 'conv2d_101', 'conv2d_109']:
            W_dict[layer.name] = layer.get_weights()[0]

    def apply_prune(W_dict, all_percent=10):
        """
        위에 있는 my_projection과 거의 동일
        - W의 shape를 저장하고, W를 모두 펴준 후 concatenate 진행하고 all_percent만큼 pruning
        - 이후 저장된 shape를 통해 원래 형태로 reshape
        W_dict을 받아서 all_percent만큼 pruning하고 이를 set_weights를 통해 model weight update
        :param W_dict: key(제고하고자 하는 layer name), value(해당하는 layer의 weight)
        :param all_percent: 제거 비율
        :return dict_nzidx: True(0이 아닌 weight)/False(0인 weight) [5차원(list로 씌워진 상태), key는 layer.name으로 구성]
        """
        dict_nzidx = {}

        W = np.array(list(W_dict.values()))
        shape_list = list(map(lambda W: W.shape, W))  # W의 shape 저장
        W_reshaped = list(map(lambda W: W.reshape(-1), W))  # W들을 모두 reshape
        concat = np.concatenate(W_reshaped, axis=0)  # reshape한 것들을 concatenate
        pcen = np.percentile(abs(concat), all_percent) # all_percent에 해당하는 값을 구함
        print("percentile " + str(pcen))
        under_threshold = abs(concat) < pcen # pcen보다 작은 값들을 0으로 만들어줌(projection)
        concat[under_threshold] = 0

        length_list = [] # layer마다 weight들의 개수들을 저장(pruning 이후의 concatenate를 자르기 위해)
        flatten_result = []  # length_list를 이용하여 concatenate된 벡터를 자름
        result = [] # 최종적으로 flatten_result를 기존 형태로 reshape한 결과

        for i in range(len(shape_list)):
            length_list.append(reduce(lambda x, y: x * y, shape_list[i]))

        start = 0
        for length in length_list:
            flatten_result.append(concat[start: length + start])  # concat에서 합친 것을 나눠줌
            start = length + start

        for i, flatten in enumerate(flatten_result):
            result.append(flatten.reshape(shape_list[i]))  # reshape한 것을 되돌려줌

        for i, key, in enumerate(W_dict): # dict_nzidx 생성
            dict_nzidx[key] = [np.array(abs(result[i])) >= pcen]  # mask 완료

        i = 0
        for layer in layers: # pruning 완료된 W_dict를 이용하여 model weight update
            if 'conv2d' in layer.name and layer.name not in ['conv2d_93', 'conv2d_101', 'conv2d_109']:
                #연구 부분print('before : {}/{}'.format(np.sum(layer.get_weights()[0] == 0), reduce(lambda x, y: x * y, layer.get_weights()[0].shape)))
                layer.set_weights([np.array(result[i])])  # weight를 prune시켜 업데이트
                #연구 부print('after : {}/{} = {}'.format(np.sum(layer.get_weights()[0] == 0), reduce(lambda x, y: x * y, layer.get_weights()[0].shape), np.sum(layer.get_weights()[0] == 0)/reduce(lambda x, y: x * y, layer.get_weights()[0].shape)))
                i += 1

        return dict_nzidx

    dict_nzidx = apply_prune(W_dict, all_percent) # True(0이 아닌 weight)/False(0인 weight) [5차원(list로 씌워진 상태), key는 layer.name으로 구성]
    model.save_weights("./checkpoints/admm_pruning/weight_prune/yolov4")  # Retraining 결과 저장

    # 시각화를 위한 과정들은 나중에...(weight 분포)

    # Retraining
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)  # step 개수 count. retraining에서 다시 시작
    total_steps = retraining_total_steps # Retraining일 때의 전체 step 수

    for epoch in range(retraining_first_stage_epochs + retraining_second_stage_epochs):  # W 학습
        print('epoch : {}'.format(epoch + 1))
        if epoch < retraining_first_stage_epochs:  # freeze layer(93, 101, 109였나 )는 freeze 시키고 학습 안함
            if not isfreeze:
                isfreeze = True
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    freeze_all(freeze)
        elif epoch >= retraining_first_stage_epochs:  # freeze layer를 unfreeze 시키고 학습
            if isfreeze:
                isfreeze = False
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    unfreeze_all(freeze)
        for image_data, target in trainset:
            train_step(image_data, target,is_admm =False, dict_nzidx=dict_nzidx)  # Retraining이므로 is_admm=False

    model.save_weights("./checkpoints/admm_pruning/retraining/yolov4") # Retraining 결과 저장

def make_dict(model):
    """
    Z, U 초기값 만드는 과정
    Model weight들을 update하는 것이 아니므로 layer.get_weights()를 통해 값만 가져옴
    :param model: Layer Weight를 가져오기 위한 parameter
    :return Z_dict, U_dict: Weight Pruning에 사용되는 Parameter(초기값)
    """
    all_percent = FLAGS.all_percent
    layers = model.layers
    Z_dict = {}
    U_dict = {}
    for layer in layers:
        if 'conv2d' in layer.name and layer.name not in ['conv2d_93', 'conv2d_101', 'conv2d_109']:
            Z_dict[layer.name] = layer.get_weights()[0] # 우리가 알고있는 커널크기(4차원, numpy)에서 list 하나로 씌워진 형태라서 나중에 다시 해줄 예정
            U_dict[layer.name] = [np.zeros_like(layer.get_weights()[0])]
    Z_dict= my_projection(Z_dict, all_percent) # my_projection에 들어갈 때는 list(5차원) 벗기고 들어감(numpy(4차원)로만 들어가게)
    return Z_dict, U_dict


def my_projection(Z_dict, all_percent=10):
    """
    Z_dict을 all_percent만큼 projection(percent에 해당하는 값(pcen)보다 작을 경우 0으로 만들어줌)
    :param Z_dict: 대응되는 Layer의 Weight값을 4차원 형태로 표현
    :param all_percent: 제거 비율
    :return Z_dict: Projection 완료된 Z_dict(4차원 형태를 list로 씌워 5차원으로 표현)
    """
    Z = np.array(list(Z_dict.values()))
    shape_list = list(map(lambda Z: Z.shape, Z))  # Z의 shape 저장
    Z_reshaped = list(map(lambda Z: Z.reshape(-1), Z))  # Z들을 모두 reshape
    concat = np.concatenate(Z_reshaped, axis=0)  # reshape한 것들을 concatenate
    pcen = np.percentile(abs(concat), all_percent)  # all_percent에 해당하는 값을 구함
    print("percentile " + str(pcen))
    under_threshold = abs(concat) < pcen  # pcen보다 작은 값들을 0으로 만들어줌(projection)
    concat[under_threshold] = 0

    length_list = []  # layer마다 weight들의 개수들을 저장(pruning 이후의 concatenate를 자르기 위해)
    flatten_result = []  # length_list를 이용하여 concatenate된 벡터를 자름
    result = []  # 최종적으로 flatten_result를 기존 형태로 reshape한 결과

    for i in range(len(shape_list)):
        length_list.append(reduce(lambda x, y: x * y, shape_list[i]))

    start = 0
    for length in length_list:
        flatten_result.append(concat[start: length + start])  # concat에서 합친 것을 나눠줌
        start = length + start

    for i, flatten in enumerate(flatten_result):
        result.append(flatten.reshape(shape_list[i]))  # reshape한 것을 되돌려줌

    for i, key, in enumerate(Z_dict):
        Z_dict[key] = [np.array(result[i])]
    return Z_dict # 반환 값은 list로 씌워진 5차원 형태





def main(_argv):
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
    #     # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    admm_pruning()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
