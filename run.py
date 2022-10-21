from utils.parser import parse_arguments
from AttentionMIL import AttentionMIL
from utils.representation import Representation

def main():
    cfg = parse_arguments()
    if cfg['model_method'] == 'calculate-representation':
        rp = Representation(cfg['representation'])
        rp.run()
    else:
        atm = AttentionMIL(cfg)
        atm.run()

if __name__ == "__main__":
    main()
