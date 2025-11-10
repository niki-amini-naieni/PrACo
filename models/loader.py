
IMPLEMENTED_MODELS = ['CounTX', 'CLIP-Count', 'TFPOC', 'VLCounter', 'DAVE', 'ZSC', 'PseCo', 'GroundingREC', 'CountGD', 'FixedPointPromptCounting']

MULTICLASS_IMPLEMENTED_MODELS = IMPLEMENTED_MODELS.copy()

def load_model(model_name, img_directory, split_images, split_classes):
    """Load and return the specified model."""
    if model_name == 'CounTX':
        from models.countx_model import CounTXModel
        return CounTXModel(img_directory, split_images, split_classes)
    elif model_name == 'CLIP-Count':
        from models.clipcount_model import CLIPCountModel
        return CLIPCountModel(img_directory, split_images, split_classes)
    elif model_name == 'TFPOC':
        from models.TFPOC_model import ClipSAMModel
        return ClipSAMModel(img_directory, split_images, split_classes)
    elif model_name == 'VLCounter':
        from models.vlcounter_model import VLCounterModel
        return VLCounterModel(img_directory, split_images, split_classes)
    elif model_name == 'DAVE':
        from models.dave_model import DAVEModel
        return DAVEModel(img_directory, split_images, split_classes)
    elif model_name == 'ZSC':
        from models.ZSC_model import ZSCModel
        return ZSCModel(img_directory, split_images, split_classes)
    elif model_name == 'PseCo':
        from models.PseCo_model import PseCoModel
        return PseCoModel(img_directory, split_images, split_classes)
    elif model_name == 'GroundingREC':
        from models.GroundingREC_model import GroundingRECModel
        return GroundingRECModel(img_directory, split_images, split_classes)
    elif model_name == 'CountGD':
        from models.countgd_model import CountGDModel
        return CountGDModel(img_directory, split_images, split_classes)
    elif model_name == 'FixedPointPromptCounting':
        from models.fixedpointpromptcounting_model import FixedPointPromptCountingModel
        return FixedPointPromptCountingModel(img_directory, split_images, split_classes)
    else:
        raise ValueError(f"Model {model_name} is not implemented.")