class Config:
    def __init__(self, ):
        self.cls = ['cat', 'dog']
        # model
        self.num_classes = 2
        self.backbone_mode = 'resnet18'
        self.pretrained = True
        self.pretrained_path = 'C:/Users/13133/Desktop/cat_dog/expreiments/pretrained_models'
        self.loss_mode = 'labelsmoothingcrossentropy'

        # train
        self.batch_size = 64
        self.max_epoch = 10
        self.num_works = 0
        self.lr = 1e-4
        self.backbone_lr = 1e-5
        self.weight_decay = 1e-4

        # dataset
        self.dataset_path = './kaggle/'

        # experiments
        self.checkpoint_path = 'C:/Users/13133/Desktop/cat_dog/expreiments/checkpoint'
        self.infenerce_path = 'C:/Users/13133/Desktop/cat_dog/expreiments/infenence'
        self.infer_img_path = './kaggle/test1'

    def update_exp(self, exp_name):
        self.checkpoint_path += '/{}'.format(exp_name)
        self.infenerce_path += '/{}'.format(exp_name)