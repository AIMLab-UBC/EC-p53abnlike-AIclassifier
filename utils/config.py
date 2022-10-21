import enum
from utils.utils import print_color, print_title, print_mix, print_title_

def _cfg_train(cfg: dict,
               cfg_: dict,
               test_external: bool) -> None:

    cfg_['model'] = {}
    for key in ['model', 'backbone', 'num_classes']:
        cfg_['model'][key] = cfg[key]

    for key in ['lr', 'wd', 'epochs', 'log_dir', 'roc_dir', 'dict_dir',
                'patience', 'optimizer', 'only_test', 'only_test', 'lr_patience',
                'checkpoints', 'use_schedular', 'tensorboard_dir', 'only_external_test',
                'external_test_name']:
        cfg_[key] = cfg[key]

    criteria_ = ['overall_auc', 'overall_acc', 'balanced_acc'] if cfg['criteria'] == 'All' else [cfg['criteria']]
    cfg_['criteria']             = criteria_
    cfg_['test_external']        = test_external
    cfg_['use_weighted_loss']    = not cfg['not_use_weighted_loss']
    cfg_['dataset']['representation_calculation'] = False
    cfg_['dataset']['representation_dir'] = cfg['representation_dir']


def _cfg_repr(cfg: dict,
              cfg_: dict) -> None:

    cfg_['dataset']['resize'] = cfg['resize']
    cfg_['dataset']['method'] = cfg['method']
    cfg_['dataset']['representation_calculation'] = True

    cfg_['representation'] = {}
    for key in ['saved_model_location', 'method', 'backbone', 'representation_dir',
                'state', 'slide_id']:
        cfg_['representation'][key] = cfg[key]
    cfg_['representation']['dataset'] = cfg_['dataset']

def _cfg(cfg: dict) -> dict:

    cfg_ = {}

    enum_ = enum.Enum('SubtypeEnum', cfg["subtypes"])
    test_external = True if cfg['external_chunk_file_location'] is not None else False

    cfg_['model_method'] = cfg['model_method']
    cfg_['CategoryEnum'] = enum_
    cfg_['num_classes']  = cfg['num_classes']

    cfg_['dataset'] = {}
    for key in ['training_chunks', 'validation_chunks', 'test_chunks', 'batch_size',
                'eval_batch_size', 'num_patch_workers', 'subtypes', 'patch_pattern',
                'chunk_file_location', 'external_chunk_file_location', 'external_chunks',
                'external_test_name']:
        cfg_['dataset'][key] = cfg[key]

    cfg_['dataset']['CategoryEnum'] = enum_
    cfg_['dataset']['use_external'] = test_external

    if cfg['model_method'] == 'calculate-representation':
        _cfg_repr(cfg, cfg_)
    elif cfg['model_method'] == 'train-attention':
        _cfg_train(cfg, cfg_, test_external)
    else:
        raise NotImplementedError()

    assert len([None for k in cfg_['CategoryEnum']])==cfg_['num_classes'], \
        f"Number of classes does not match with subtypes!"

    return cfg_


def print_config(cfg: dict) -> None:
    dict__ = {}
    def add_values(dict_: dict) -> None:
        for key, value in dict_.items():
            if key not in dict__:
                dict__[key] = value

    for key, value in cfg.items():
        if type(value) == dict:
            add_values(value)
        else:
            dict__[key] = value

    if dict__['representation_calculation']:
        del dict__['dataset']

    # print_title('Displaying the config settings')
    print_title_('Displaying the config settings')
    for key, value in dict__.items():
        # print_mix(f"__{key}__  {value}", 'CYAN')
        print(f"{key}  {value}")
