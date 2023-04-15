"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import glob
import os

import imageio
import torch as th
import torch.distributed as dist
import torchvision as tv
import matplotlib
import numpy as np

from guided_diffusion import dist_util, logger
from guided_diffusion.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.sampler import DPMSolverSampler

import os

os.environ["OPENAI_LOGDIR"] = "test"


def load_data(data_dir, batch_size, image_size):
    label_path = glob.glob(os.path.join(data_dir, '*.png'))
    labels = [imageio.imread(path) for path in label_path]
    # resize labels to image size
    labels = [th.from_numpy(label).unsqueeze(0).float() for label in labels]
    labels = th.stack(labels)
    labels = th.nn.functional.interpolate(labels, size=image_size, mode='nearest')
    return th.utils.data.DataLoader(labels, batch_size=batch_size)


def sample_image(x, diffusion, model, last=True, classifier=None, model_kwargs=None):
    skip = 1

    def model_fn(x, t, **model_kwargs):
        out = model(x, t, **model_kwargs)
        # If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
        # We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
        if hasattr(model, "out_channels"):
            if model.out_channels == 6:
                out = th.split(out, 3, dim=1)[0]
        return out

    def classifier_fn(x, t, y, **classifier_kwargs):
        logits = classifier(x, t)
        log_probs = th.nn.functional.log_softmax(logits, dim=-1)
        return log_probs[range(len(logits)), y.view(-1)]

    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=th.from_numpy(diffusion.betas))
    model_fn_continuous = model_wrapper(
        model_fn,
        noise_schedule,
        model_type="noise",
        model_kwargs=model_kwargs,
        guidance_type="uncond" if classifier is None else "classifier",
        condition=model_kwargs["y"] if "y" in model_kwargs.keys() else None,
        guidance_scale=None,
        classifier_fn=classifier_fn,
        classifier_kwargs={},
    )
    dpm_solver = DPM_Solver(
        model_fn_continuous,
        noise_schedule,
    )
    x = dpm_solver.sample(
        x,
        steps=20,
    )
    return x


def main():
    args = create_argparser().parse_args()

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if os.path.exists(args.model_path):
        model.load_state_dict(
            th.load(args.model_path, map_location="cpu")
        )
    else:
        print(f"model path, {args.model_path} not found, using random init")
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_samples = []
    labels = []
    shape = (args.batch_size, 3, 256, 256)
    noise = th.randn(*(shape[1:])).repeat(shape[0], 1, 1, 1).to(dist_util.dev())
    for i, label in enumerate(data):
        if label.shape[0] != args.batch_size:
            continue
        model_kwargs = preprocess_input({"label": label}, num_classes=args.num_classes)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        # model_kwargs = th.load("cond.pth")
        sample_fn = (
            diffusion.p_sample_loop
        )
        shape = (args.batch_size, 3, label.shape[2], label.shape[3])

        # sample = sampler.sample(S=20, shape=shape, noise_prediction_model=model, conditioning=model_kwargs)

        x = sample_image(noise, diffusion, model, last=True, classifier=None, model_kwargs=model_kwargs)
        imageio.imwrite(f"test_{i}.png", x[0].cpu().numpy().transpose(1, 2, 0))


def preprocess_input(data, num_classes):
    # move to GPU and change data types
    data['label'] = data['label'].long()

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    input_label = th.FloatTensor(bs, num_classes, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

    return {'y': input_semantics}


def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        results_path="",
        is_train=False,
        s=1.0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

# --data_dir ./data/ --dataset_mode echo --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64  --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 151 --class_cond True --no_instance True --batch_size 2 --num_samples 2000 --s 1.5 --model_path OUTPUT/ADE20K-SDM-256CH-FINETUNE/ema_0.9999_best.pt --results_path RESULTS/ADE20K-SDM-256CH
