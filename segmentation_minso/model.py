import segmentation_models_pytorch as smp


def get_model(architecture, backbone, weight):
    print(architecture)
    if architecture == "Unet":
        model = smp.Unet(encoder_name=backbone,
                         encoder_weights=weight,
                         in_channels=3,
                         classes=12)
    elif architecture == "UnetPlusPlus":
        model = smp.UnetPlusPlus(encoder_name=backbone,
                                 encoder_weights=weight,
                                 in_channels=3,
                                 classes=12)
    elif architecture == "Linknet":
        model = smp.Linknet(encoder_name=backbone,
                            encoder_weights=weight,
                            in_channels=3,
                            classes=12)
    elif architecture == "FPN":
        model = smp.FPN(encoder_name=backbone,
                        encoder_weights=weight,
                        in_channels=3,
                        classes=12)
    elif architecture == "PSPNet":
        model = smp.PSPNet(encoder_name=backbone,
                           encoder_weights=weight,
                           in_channels=3,
                           classes=12)
    elif architecture == "PAN":
        model = smp.PAN(encoder_name=backbone,
                        encoder_weights=weight,
                        in_channels=3,
                        classes=12)
    elif architecture == "DeepLabV3":
        model = smp.DeepLabV3(encoder_name=backbone,
                              encoder_weights=weight,
                              in_channels=3,
                              classes=12)
    elif architecture == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(encoder_name=backbone,
                                  encoder_weights=weight,
                                  in_channels=3,
                                  classes=12)
    else:
        print(f"Wrong architecture check : https://smp.readthedocs.io/en/latest/index.html")
        quit()
    return model
