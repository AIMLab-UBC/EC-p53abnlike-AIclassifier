from utils.utils import my_collate
from torch.utils.data import DataLoader
from utils.load_json import create_data_set


class Data_Loader(object):
    def __init__(self,
                 cfg: dict,
                 state: str):
        self.cfg = cfg
        self.state = state
        chunk, training_set = self.handle_chunk()

    def handle_chunk(self) -> dict | bool:
        if self.state == 'train':
            chunk = self.cfg["training_chunks"]
            training_set = True
        elif self.state == 'validation':
            chunk = self.cfg["validation_chunks"]
            training_set = False
        elif self.state == 'test':
            chunk = self.cfg["test_chunks"]
            training_set = False
        elif self.state == 'external':
            chunk = self.cfg["external_chunks"]
            training_set = False
        else:
            raise ValueError(f'{state} should be either train, validation, test or external!')
        return chunk, training_set

    def run(self):
        raise NotImplementedError()


class RepresentationDataset(Data_Loader):
    def __init__(self,
                 cfg: dict,
                 state: str,
                 slide_id: int) -> None:
        self.cfg = cfg
        self.state = state
        chunk, _ = self.handle_chunk()
        self.patch_dataset, _ = create_data_set(cfg, chunk, state=state,
                                                slide_id=slide_id)

    def run(self):
        batch_size = self.cfg["eval_batch_size"]
        return DataLoader(self.patch_dataset, batch_size=batch_size,
                      shuffle=False, pin_memory=True,
                      num_workers=self.cfg["num_patch_workers"])

class Dataset(Data_Loader):
    def __init__(self,
                 cfg: dict,
                 state: str = 'train') -> None:
        self.cfg = cfg
        self.state = state
        chunk, training_set = self.handle_chunk()
        self.patch_dataset, self.labels = create_data_set(cfg, chunk, state=state,
                                                          training_set=training_set)

    def run(self):
        batch_size = self.cfg["batch_size"] if self.state=='train' else \
                     self.cfg["eval_batch_size"]

        return DataLoader(self.patch_dataset, batch_size=batch_size,
                      shuffle=True, pin_memory=True, collate_fn=my_collate,
                      num_workers=self.cfg["num_patch_workers"])
