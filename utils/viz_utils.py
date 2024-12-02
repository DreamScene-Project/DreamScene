import cv2
import numpy as np
import torch


def lat2rgb(x, rgb_latent_factors):
    return torch.clip(
        (x.permute(0, 2, 3, 1) @ rgb_latent_factors.to(x.dtype)).permute(0, 3, 1, 2),
        0.0,
        1.0,
    )


def rgb2sat(img, T=None):
    max_ = torch.max(img, dim=1, keepdim=True).values + 1e-5
    min_ = torch.min(img, dim=1, keepdim=True).values
    sat = (max_ - min_) / max_
    if T is not None:
        sat = (1 - T) * sat
    return sat


class TextCanvas:
    def __init__(self, device):
        self.text = np.zeros((512, 512 * 4, 3))
        self.text_line = 1
        self.device = device

    def putText(self, content):
        cv2.putText(
            self.text,
            content,
            (8, 48 * self.text_line),
            cv2.FONT_HERSHEY_COMPLEX,
            1.5,
            (255, 255, 255),
            2,
        )
        self.text_line += 1

    def toImage(self):
        return (
            torch.from_numpy(self.text / 255.0)
            .clip(0.0, 1.0)
            .to(self.device)
            .reshape(
                512,
                4,
                512,
                3,
            )
            .permute(1, 3, 0, 2)
        )
