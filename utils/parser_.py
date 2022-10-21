import argparse

def make_dict(ll):
    return {k: v for (k, v) in ll}

class ParseKVToDictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, type=None, **kwargs):
        if nargs != '+':
            raise argparse.ArgumentTypeError(f"ParseKVToDictAction can only be used for arguments with nargs='+' but instead we have nargs={nargs}")
        super(ParseKVToDictAction, self).__init__(option_strings, dest,
                nargs=nargs, type=type, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, option_string.lstrip('-'), make_dict(values))

def subtype_kv(kv: str) -> tuple:
    """
    Used to identify and convert key=value arguments into a tuple (key.upper(), int(value)).
    For example: MMRd=0 becomes (MMRD, int(0))
    This is to be passed as the type when calling argparse.ArgumentParser.add_argument()

    Parameters
    ----------
    kv: str
        a key=value argument

    Returns
    -------
    tuple
        (key.upper(), int(value)) from key=value
    """
    try:
        k, v = kv.split("=")
    except:
        raise argparse.ArgumentTypeError(f"value {kv} is not separated by one '='")
    k = k.upper()
    try:
        v = int(v)
    except:
        raise argparse.ArgumentTypeError(f"right side of {kv} should be int")
    return (k, v)
