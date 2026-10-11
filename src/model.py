import segmentation_models_pytorch as smp


def build_model(cfg):
    m = cfg["model"]
    arch = getattr(smp, m["arch"])
    return arch(encoder_name=m["encoder"], encoder_weights=m["encoder_weights"], classes=m["classes"])
