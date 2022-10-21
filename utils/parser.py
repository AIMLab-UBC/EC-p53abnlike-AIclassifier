import argparse
from torchvision import models

from utils.config import _cfg, print_config
from utils.random import fix_random_seeds
from utils.utils import handle_log_folder
from utils.parser_ import ParseKVToDictAction, subtype_kv


# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))
# model_names += ['vit_deit_small_patch16_224']
model_names = ['alexnet', 'vgg16', 'vgg19', 'vgg16_bn','vgg19_bn', 'resnet18',
               'resnet34', 'resnet50', 'resnext50_32x4d', 'resnext101_32x8d',
               'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'mnasnet1_3',
               'shufflenet_v2_x1_5', 'squeezenet1_1', 'efficientnet-b0', 'efficientnet-l2',
               'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
               'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6',
               'efficientnet-b7', 'efficientnet-b8', 'vit_deit_small_patch16_224']

def parse_input():
    parser = argparse.ArgumentParser(description='AttentionMIL')
    parser.add_argument('--experiment_name', type=str,
                help="Experiment's name that is utiziled as the name of a directory in the 'log_dir' location.")
    parser.add_argument('--log_dir', type=str,
                help="Directory in which checkpoints and all information are stored.")
    parser.add_argument('--chunk_file_location', type=str,
                help='Path to the JSON file containing patches address')
    parser.add_argument('--training_chunks', nargs="+", type=int, default=[0],
                help="Space separated IDs specifying chunks included in training.")
    parser.add_argument('--validation_chunks', nargs="+", type=int, default=[1],
                help="Space separated IDs specifying chunks included in validation.")
    parser.add_argument('--test_chunks', nargs="+", type=int, default=[2],
                help="Space separated IDs specifying chunks included in test.")
    parser.add_argument('--patch_pattern',
                help="Patterns of the stored patches, which is used to extracted "
                "information such as slide's ID from the path.")
    parser.add_argument('--subtypes', nargs='+', type=subtype_kv, action=ParseKVToDictAction,
                help="Space separated words describing subtype=groupping pairs for this study.")
    parser.add_argument('--num_classes', default=5, type=int,
                help='Number of output classes, i.e., the number of subtypes.')
    parser.add_argument('--num_patch_workers', default=4, type=int,
                help='Number of data loading workers.')
    parser.add_argument('--batch_size', default=256, type=int,
                help='Batch size for the trianing phase.')
    parser.add_argument('--eval_batch_size', default=256, type=int,
                help='Batch size for the validation and testing phase.')
    parser.add_argument('--seed', default=31, type=int,
                help='Seed for initializing training.')
    parser.add_argument('--backbone', default='resnet18',
                choices=model_names, help="Model architecture inorder to find the "
                "dimension of embedding.")
    parser.add_argument('--external_test_name', type=str,
                help="Usefull when testing on multiple external datasets.")
    parser.add_argument('--external_chunk_file_location', type=str, default=None,
                help='Path to JSON file contains external dataset')
    parser.add_argument('--external_chunks', nargs="+", type=int, default=[0,1,2],
                help="Space separated number IDs specifying chunks to use for testing (default use all the slides).")

    help_method = """Whether to calculate the representation or train the model"""
    subparsers_method = parser.add_subparsers(dest='model_method',
            required=True,
            help=help_method)

    help_repr = """Calculating patchs' embeddings of extracted patches."""
    parser_repr = subparsers_method.add_parser("calculate-representation",
                help=help_repr)
    parser_repr.add_argument('--resize', default=None, type=int,
                help="If the value is true, the extracted patches are resized "
                "prior to being fed to the network. For example, setting this "
                "value to 224 is required for ViTs.")
    parser_repr.add_argument('--method',
                choices=['Vanilla'], default='Vanilla',
                help="The network structure used to calculate embedding; now, "
                "only the Vanilla version is supplied, but this can be expanded "
                "to incorporate other networks such as self-supervised.")
    parser_repr.add_argument('--saved_model_location', type=str,
                help="Path to the saved trained model for extracting embeddings.")
    parser_repr.add_argument('--state', type=str,
                choices=['train', 'validation', 'test', 'external'], default=None,
                help="Specify the data state from train, validation, test, and external."
                "Setting this flag causes the patches to be phase-restitched to the "
                "specific state. It is advantageous while using multithreading. "
                "For instance, we might set this flag to train if we're just "
                "interested in calculating embeddings for training patches.")
    parser_repr.add_argument('--slide_id', type=int,
                help="Identical to the preceding flag, it can be used for "
                "multithreading. This parameter restricts the embedding "
                "calculation to a single slide when set.")


    help_train = """Training AttentionMIL"""
    parser_train = subparsers_method.add_parser("train-attention",
                help=help_train)

    parser_train.add_argument('--epochs', default=5, type=int,
                help='Number of total epochs to run.')
    parser_train.add_argument('--lr', default=4e-5, type=float,
                help='Initial learning rate.')
    parser_train.add_argument('--wd', default=4e-5, type=float,
                help='Weight decay.')
    parser_train.add_argument('--optimizer',
                choices=['Adam', 'AdamW', 'SGD'], default='Adam',
                help='Optimizer for training the model: 1. Adam '
                '2. AdamW 3. SGD')
    parser_train.add_argument('--patience', default=10, type=int,
                help='How long to wait after last time validation loss improved.')
    parser_train.add_argument('--lr_patience', default=5, type=int,
                help="How long to wait after last time validation loss improved "
                "to change the learning rate.")
    parser_train.add_argument('--not_use_weighted_loss', action='store_true',
                help="Setting this flag disables weighted loss in the code. If "
                "the dataset contains a class imbalance, this flag should not be set.")
    parser_train.add_argument('--use_schedular', action='store_true',
                help="Using schedular for decreasig learning rate in a way that if "
                "lr_patience has passed, it will be reduced by 0.8.")
    parser_train.add_argument('--criteria', type=str,
                choices=['overall_auc', 'overall_acc', 'balanced_acc', 'All'], default='All',
                help="Criteria for saving the best model: 1. overall_auc: using AUC "
                "2. overall_acc: uses accuracy 3. balanced_acc: balanced accuracy "
                "for imbalanced data 4. All: uses all the possible criterias "
                "NOTE: For calculating AUC for multiclasses, OVO is used to mitigate "
                "the imbalanced classes.")
    parser_train.add_argument('--only_test', action='store_true',
                help='Only test not train.')
    parser_train.add_argument('--only_external_test', action='store_true',
                help='Only test on the external dataset.')

    help_model = """Structure of the model: Currently, we have just included VarMIL,
                    but others such as DeepMIL can be incorporated."""
    subparsers_model = parser_train.add_subparsers(dest='model',
                required=True,
                help=help_model)

    help_varmil = """VarMIL (variability-aware deep multiple instance learning)"""
    parser_varmil = subparsers_model.add_parser("VarMIL",
                help=help_varmil)

    args = parser.parse_args()
    return args

def parse_arguments():
    args = parse_input()
    cfg  = vars(args)
    handle_log_folder(cfg)
    fix_random_seeds(cfg["seed"])
    cfg = _cfg(cfg)
    print_config(cfg)
    return cfg


if __name__ == "__main__":
    print(parse_arguments())
