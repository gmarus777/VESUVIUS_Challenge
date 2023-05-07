
'''
GCP:
monai.networks.nets.FlexibleUNet(in_channels = self.z_dim,
                              out_channels =1 ,
                              backbone = 'efficientnet-b3',
                              pretrained=True,
                              decoder_channels=( 1024, 768, 512, 256, 128, 64, ),
                              spatial_dims=2,
                              norm=('batch', {'eps': 0.001, 'momentum': 0.1}),
                              #act=('relu', {'inplace': True}),
                              act = None,
                              dropout=0.0,
                              decoder_bias=False,
                              upsample='deconv',
                              interp_mode='nearest',
                              is_pad=False)



monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=self.z_dim,
            out_channels=1,
            channels=( 32, 64, 128, 256,512,512, 1204 ),
            strides=(2, 2, 2, 2,2, 2),
            num_res_units=4,
            dropout=0,
            norm='batch',
            bias =False,

        )

 monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels= self.z_dim,
            out_channels=1,
            channels=(  32, 64, 128, 256, 512, ),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0,
            norm = 'batch',
            bias =False,

        )




monai.networks.nets.FlexibleUNet(in_channels = self.z_dim,
                              out_channels =1 ,
                              backbone = 'efficientnet-b0',
                              pretrained=True,
                              decoder_channels=(512, 256, 128, 64, 32,),
                              spatial_dims=2,
                              norm=('batch', {'eps': 0.001, 'momentum': 0.1}),
                              act=('relu', {'inplace': True}),
                              dropout=0.0,
                              decoder_bias=True,
                              upsample='deconv',
                              interp_mode='nearest',
                              is_pad=False)

smp.Unet(
            encoder_name='se_resnext50_32x4d',
            encoder_weights='imagenet',
            in_channels=self.z_dim,
            classes=1,
            activation=None,
        )

### MONAI ###
        self.diceloss = monai.losses.DiceLoss(include_background=True,
                                              sigmoid=True,
                                              squared_pred=False,
                                              jaccard=False,
                                              batch=True
                                              )

        self.monai_tverskyLoss = monai.losses.TverskyLoss(include_background=True,
                                                          sigmoid=True,
                                                          softmax=False,
                                                          other_act=None,
                                                          alpha=0.5,
                                                          beta=0.5,
                                                          # reduction=LossReduction.MEAN,
                                                          smooth_nr=0,
                                                          smooth_dr=1e-06,
                                                          batch=True
                                                          )

        self.focalloss = monai.losses.FocalLoss(include_background=True,
                                                gamma=2.0,
                                                # weight=.25,
                                                # focal_weight=.25,
                                                )


 self.loss_focal = smp.losses.FocalLoss(
            mode='binary',
            # alpha=.1,
            gamma=2.0,
            ignore_index=None,

            normalized=False,
            reduced_threshold=None)

'''
