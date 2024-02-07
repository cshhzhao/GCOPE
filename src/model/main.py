from importlib import import_module
import torch


class Model(torch.nn.Module):
    def __init__(
                self,
                backbone,
                answering = torch.nn.Identity(),
                saliency = torch.nn.Identity(),
            ):
        super().__init__()
        self.backbone = backbone
        self.answering = answering
        self.saliency = saliency

    def forward(self, data):
        data.x = self.saliency((data.x))
        return self.answering(self.backbone(data))
                                            

def get_model(
        backbone_kwargs,
        answering_kwargs = None,
        saliency_kwargs = None,
    ):

    backbone = import_module(f'model.backbone.{backbone_kwargs.pop("name")}').get_model(**backbone_kwargs)
    answering = torch.nn.Identity() if answering_kwargs is None else import_module(f'model.answering.{answering_kwargs.pop("name")}').get_model(**answering_kwargs)
    saliency = torch.nn.Identity() if saliency_kwargs is None else import_module(f'model.saliency.{saliency_kwargs.pop("name")}').get_model(**saliency_kwargs)

    return Model(backbone, answering, saliency)
