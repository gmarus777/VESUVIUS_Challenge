

### DO NOT USE (THIS IS A TEMPLATE ONLY)


PATCH_SIZE = 224
Z_DIM = 4
COMPETITION_DATA_DIR_str = "kaggle/input/vesuvius-challenge-ink-detection/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CFG:
    device = DEVICE

    THRESHOLD = 0.4
    use_wandb = False

    ######### Dataset #########

    # stage: 'train' or 'test'
    stage = 'train'

    # location of competition Data
    competition_data_dir = COMPETITION_DATA_DIR_str

    # Number of slices in z-dim: 1<z_dim<65
    z_dim = Z_DIM

    # fragments to use for training avalaible [1,2,3]
    train_fragment_id = [2, 3]

    # fragments to use for validation
    val_fragment_id = [1]

    batch_size = 4

    # Size of the patch and stride for feeding the model
    patch_size = PATCH_SIZE
    stride = patch_size // 2

    num_workers = 0
    on_gpu = True

    ######## Model and Lightning Model paramters ############

    # MODEL
    model = monai.networks.nets.FlexibleUNet(in_channels=z_dim,
                                             out_channels=1,
                                             backbone='efficientnet-b3',
                                             pretrained=True,
                                             decoder_channels=(512, 256, 128, 64, 32),
                                             spatial_dims=2,
                                             norm=('instance', {'eps': 0.001, 'momentum': 0.1}),
                                             # act=('relu', {'inplace': True}),
                                             act=None,
                                             dropout=0.0,
                                             decoder_bias=False,
                                             upsample='deconv',
                                             interp_mode='nearest',
                                             is_pad=False)

    checkpoint = None
    save_directory = None

    accumulate_grad_batches = 48  # experiments showed batch_size * accumulate_grad = 192 is optimal
    learning_rate = 0.0001
    eta_min = 5e-7
    t_max = 50
    max_epochs = 60
    weight_decay = 0.00001
    precision = 16

    # checkpointing
    save_top_k = 5

    monitor = "FBETA"
    mode = "max"

    ####### Augemtnations ###############

    # Training Aug
    train_transforms = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(patch_size, patch_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
            A.GaussNoise(var_limit=[10, 50]),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_width=int(patch_size * 0.3), max_height=int(patch_size * 0.3),
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean=[0] * z_dim,
            std=[1] * z_dim
        ),
        ToTensorV2(transpose_mask=True),
    ]

    # Validaiton Aug
    val_transforms = [
        A.Resize(patch_size, patch_size),
        A.Normalize(
            mean=[0] * z_dim,
            std=[1] * z_dim
        ),
        ToTensorV2(transpose_mask=True),
    ]

    # Test Aug
    test_transforms = [
        A.Resize(patch_size, patch_size),
        A.Normalize(
            mean=[0] * z_dim,
            std=[1] * z_dim
        ),

        ToTensorV2(transpose_mask=True),
    ]





###### POSSIBLE TRANSFORMS TO USE

'''


Additional 

A..augmentations.geometric.transforms.Perspective(scale=(0.05, 0.1),
                                                    keep_size=True,
                                                     pad_mode=0, 
                                                     pad_val=0, 
                                                     mask_pad_val=0, 
                                                     fit_output=False, 
                                                     interpolation=1, 
                                                     always_apply=False, 
                                                     p=0.5)


A.augmentations.geometric.resize.RandomScale(scale_limit=0.1, 
                                                interpolation=1, 
                                                always_apply=False, 
                                                p=0.5)



A.augmentations.geometric.transforms.OpticalDistortion(distort_limit=0.05, 
                                                        shift_limit=0.05, 
                                                        interpolation=1, 
                                                        border_mode=4, 
                                                        value=None, 
                                                        mask_value=None, 
                                                        always_apply=False, 
                                                        p=0.5)    




A.augmentations.geometric.transforms.ElasticTransform(alpha=1, 
                                                        sigma=50, 
                                                        alpha_affine=50, 
                                                        interpolation=1, 
                                                        border_mode=4, 
                                                        value=None, 
                                                        mask_value=None, 
                                                        always_apply=False, 
                                                        approximate=False, 
                                                        same_dxdy=False, 
                                                        p=0.5)                                           






Original Transforms:


    class Image_Transforms:

    train_transforms = A.Compose(
        [
            # A.RandomResizedCrop(
            #     size, size, scale=(0.85, 1.0)),
            A.Resize(PATCH_SIZE, PATCH_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.75),
            A.ShiftScaleRotate(p=0.75),
            A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
            ], p=0.4),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.CoarseDropout(max_holes=1, max_width=int(PATCH_SIZE * 0.3), max_height=int(PATCH_SIZE * 0.3),
                            mask_fill_value=0, p=0.5),
            # A.Cutout(max_h_size=int(size * 0.6),
            #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
            A.Normalize(
                mean=[0] * Z_DIM,
                std=[1] * Z_DIM,
            ),
            ToTensorV2(transpose_mask=True),
        ]
    )

    val_transforms = A.Compose(
        [
        A.Resize(PATCH_SIZE, PATCH_SIZE),
        A.Normalize(
            mean=[0] * Z_DIM,
            std=[1] * Z_DIM
        ),

        ToTensorV2(transpose_mask=True),
    ]
    )







    Updated:


    class Image_Transforms:

    train_transforms = A.Compose(
        [
            # A.RandomResizedCrop(
            #     size, size, scale=(0.85, 1.0)),
            #A.Resize(PATCH_SIZE, PATCH_SIZE),
            A.augmentations.geometric.resize.RandomScale(scale_limit=0.1,
                                                         interpolation=1,
                                                         always_apply=False,
                                                         p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(.25, (-.3, .3), p=0.75),
            A.ShiftScaleRotate(p=0.75),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.GaussNoise(var_limit=[10, 50], p=0.4),

    A.OneOf([
                A.GaussNoise(var_limit=[10, 60]),
                A.GaussianBlur(blur_limit=(1, 9)),
                A.MotionBlur(blur_limit=9),
            ], p=0.3),

            A.augmentations.geometric.transforms.ElasticTransform(alpha=120,
                                                                  sigma=120*0.05,
                                                                  alpha_affine=120 * 0.03,
                                                                  interpolation=1,
                                                                  border_mode=cv2.BORDER_CONSTANT,
                                                                  value=0,
                                                                  mask_value=0,
                                                                  always_apply=False,
                                                                  approximate=False,
                                                                  same_dxdy=False,
                                                                  p=0.4),

            A.augmentations.geometric.transforms.OpticalDistortion(distort_limit=0.1,
                                                                   shift_limit=0.02,
                                                                   interpolation=1,
                                                                   border_mode=cv2.BORDER_CONSTANT,
                                                                   value=0,
                                                                   mask_value=0,
                                                                   always_apply=False,
                                                                   p=0.3),

            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.CoarseDropout(max_holes=1, max_width=64, max_height=64,
                            mask_fill_value=0, p=0.5),
            A.CoarseDropout(max_holes=1, max_width=38, max_height=32,
                            mask_fill_value=0, p=0.5),
            # A.Cutout(max_h_size=int(size * 0.6),
            #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
            A.Resize(PATCH_SIZE, PATCH_SIZE),
            A.Normalize(
                mean=[0] * Z_DIM,
                std=[1] * Z_DIM,
            ),
            ToTensorV2(transpose_mask=True),
        ]
    )

    val_transforms = A.Compose(
        [
        A.Resize(PATCH_SIZE, PATCH_SIZE),
        A.Normalize(
            mean=[0] * Z_DIM,
            std=[1] * Z_DIM
        ),

        ToTensorV2(transpose_mask=True),
    ]
    )



    '''

'''
