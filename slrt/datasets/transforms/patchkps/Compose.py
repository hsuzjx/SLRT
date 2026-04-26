class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, patches, kps):
        for t in self.transforms:
            patches, kps = t(patches, kps)
        return patches, kps
