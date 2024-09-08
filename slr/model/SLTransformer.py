from typing import Any

from slr.model.SLRBaseModel import SLRBaseModel


class SLTransformer(SLRBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "SLTransformer"

        # 定义网络
        self._init_networks()

        # 定义解码器
        self._define_decoder()

        # 定义损失函数
        self._define_loss_function()

    def _init_networks(self):
        pass

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def step_forward(self, batch) -> Any:
        pass

    def _define_loss_function(self):
        pass

    def _define_decoder(self):
        pass
