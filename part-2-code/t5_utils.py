import os
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    if not args.use_wandb:
        return
    wandb.init(
        project="hw4-t5-sql",
        name=args.experiment_name,
        config=vars(args),
    )

def initialize_model(args):
    """
    Initialize T5-small.

    If args.finetune is True, load pretrained weights from
    'google-t5/t5-small'. Otherwise, initialize from config (training from
    scratch).
    """
    ckpt = "google-t5/t5-small"
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(ckpt)
    else:
        config = T5Config.from_pretrained(ckpt)
        model = T5ForConditionalGeneration(config)

    model.to(DEVICE)
    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    """
    Save model checkpoint. We only need state_dict; config comes from
    'google-t5/t5-small'.
    """
    mkdir(checkpoint_dir)
    fname = "best.pt" if best else "last.pt"
    ckpt_path = os.path.join(checkpoint_dir, fname)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved {'best' if best else 'last'} model to {ckpt_path}")

def load_model_from_checkpoint(args, best):
    """
    Load model from checkpoint_dir set in args.checkpoint_dir.
    """
    model = initialize_model(args)
    fname = "best.pt" if best else "last.pt"
    ckpt_path = os.path.join(args.checkpoint_dir, fname)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    print(f"Loaded {'best' if best else 'last'} checkpoint from {ckpt_path}")
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        raise NotImplementedError(f"Unknown optimizer_type {args.optimizer_type}")

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result
