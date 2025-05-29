# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
from .detr_vae import build_cnnmlp as build_cnnmlp
from .detr_vae import build_diffusion as build_diffusion

from .detr_vae_tac import build as build_vae_tac
from .detr_vae_tac import build_cnnmlp as build_cnnmlp_tac
from .detr_vae_tac import build_diffusion as build_diffusion_tac

def build_ACT_model(args):
    return build_vae(args)

def build_CNNMLP_model(args):
    return build_cnnmlp(args)

def build_diffusion_model(args):
    return build_diffusion(args)

def build_ACT_model_tac(args):
    return build_vae_tac(args)

def build_CNNMLP_model_tac(args):
    return build_cnnmlp_tac(args)

def build_diffusion_model_tac(args):
    return build_diffusion_tac(args)