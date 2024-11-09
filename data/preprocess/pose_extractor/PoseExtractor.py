from mmpose.apis import MMPoseInferencer


class PoseExtractor:
    def __init__(self, model, device='cuda:0'):
        self.model = model
        self.device = device

        self.inferencer = MMPoseInferencer(model, device=device)

    def __call__(self, img_or_path, show=False, pred_out_dir=None):
        result_generator = self.inferencer(img_or_path, show=show, pred_out_dir=pred_out_dir)
        return [result for result in result_generator]
