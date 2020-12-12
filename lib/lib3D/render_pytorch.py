import numpy as np
import neural_renderer as nr

import torch

class Renderer(object):
    def __init__(self, img_size, faces, focal_length, batch_size, use_gpu):
        self.img_size = img_size # with input image size (img_size, img_size)
        faces = torch.from_numpy(faces.astype('int32'))
        self.faces = faces.unsqueeze(0).repeat(batch_size, 1, 1)
        self.renderer = nr.Renderer(camera_mode='projection', image_size=img_size, orig_size=img_size)
        self.use_gpu = use_gpu

        self.K = torch.Tensor([[focal_length, 0, 0],
                               [0, focal_length, 0],
                               [0, 0, 0]]).unsqueeze(0).expand(batch_size, 3, 3).float()

        self.R = torch.eye(3).unsqueeze(0).expand(batch_size, 3, 3).float()
        self.t = torch.zeros((1, 3)).unsqueeze(0).expand(batch_size, 1, 3).float()

        # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
        self.texture_size = 8
        self.textures = torch.ones(1, self.faces.shape[1], self.texture_size, self.texture_size, self.texture_size, 3,
                              dtype=torch.float32).cuda()

        if self.use_gpu:
            self.faces = self.faces.cuda()
            self.renderer = self.renderer.cuda()
            self.K = self.K.cuda()
            self.R = self.R.cuda()
            self.t = self.t.cuda()

    def render_rgb(self, vertices):
        rgb_rendered = self.renderer(vertices, self.faces, self.textures, mode='rgb', K=self.K, R=self.R, t=self.t)

        return rgb_rendered

    def render_silhouette(self, vertices):
        silhouette_proj = self.renderer(vertices, self.faces, mode='silhouettes', K=self.K, R=self.R, t=self.t)

        return silhouette_proj