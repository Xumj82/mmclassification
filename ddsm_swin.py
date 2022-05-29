
import time
import mmcv
import os.path as osp

from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.apis import train_model
from mmcv import Config
config_file = 'ddsm_swin_config.py'
checkpoint_file = 'checkpoints/swin_small_patch4_window7_224-cc7a01c9.pth'
cfg = Config.fromfile('ddsm_swin_config.py')

# 修改模型分类头中的类别数目


# 加载预训练权重
cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone')
cfg.model.head.num_classes = 3
cfg.model.head.topk = (1, )
# 根据你的电脑情况设置 sample size 和 workers 
cfg.data.samples_per_gpu = 4
cfg.data.workers_per_gpu = 1

# 指定训练集路径
cfg.data.train.data_prefix = '/home/xumingjie/dataset/patch_set/img_dir/train'
cfg.data.train.ann_file = '/home/xumingjie/dataset/patch_set/img_dir/train_meta.csv'
cfg.data.train.classes = ('bkg','calc','mass')

# 指定验证集路径
cfg.data.val.data_prefix = '/home/xumingjie/dataset/patch_set/img_dir/test'
cfg.data.val.ann_file = '/home/xumingjie/dataset/patch_set/img_dir/test_meta.csv'
cfg.data.val.classes = ('bkg','calc','mass')

# 指定测试集路径
cfg.data.test.data_prefix = '/home/xumingjie/dataset/patch_set/img_dir/test'
cfg.data.test.ann_file = '/home/xumingjie/dataset/patch_set/img_dir/test_meta.csv'
cfg.data.test.classes = ('bkg','calc','mass')

# 设定数据集归一化参数
normalize_cfg = dict(type='Normalize', mean=[124.508, 116.050, 106.438], std=[58.577, 57.310, 57.437], to_rgb=True)
cfg.data.train.pipeline[3] = normalize_cfg
cfg.data.val.pipeline[3] = normalize_cfg
cfg.data.test.pipeline[3] = normalize_cfg

# 修改评价指标选项
cfg.evaluation['metric_options']={'topk': (1, )}

# 设置优化器
cfg.optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
cfg.optimizer_config = dict(grad_clip=None)

# 设置学习率策略
cfg.lr_config = dict(policy='step', step=1, gamma=0.1)
cfg.runner = dict(type='EpochBasedRunner', max_epochs=100)

# 设置工作目录以保存模型和日志
cfg.work_dir = './work_dirs/ddsm_patch_dataset'

# 设置每 10 个训练批次输出一次日志
# cfg.log_config.interval = 10

# 设置随机种子，并启用 cudnn 确定性选项以保证结果的可重复性
from mmcls.apis import set_random_seed
cfg.seed = 0
set_random_seed(0, deterministic=True)

cfg.gpu_ids = range(1)


# 创建工作目录
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# 创建分类器
model = build_classifier(cfg.model)
model.init_weights()
# 创建数据集
datasets = [build_dataset(cfg.data.train)]
# 添加类别属性以方便可视化
model.CLASSES = datasets[0].CLASSES
# 开始微调
train_model(
    model,
    datasets,
    cfg,
    distributed=False,
    validate=True,
    timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
    meta=dict())