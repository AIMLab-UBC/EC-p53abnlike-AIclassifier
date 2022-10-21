# AttentionMIL

### Development Information ###
```
Date Created: 20 Aug 2022
Developer: Amirali
Version: 0.0
```

### About The Project ###
This repo contains the implementation of *Artificial intelligence-based histopathology image analysis identifies a novel subset of endometrial cancers with unfavourable outcome*. The below GIF illustrates the proposed workflow.

![](gif/workflow.gif)


## Installation

```
mkdir AttentionMIL
cd AttentionMIL
git clone git clone https://svn.bcgsc.ca/bitbucket/scm/~adarbandsari/attentionmil.git .
pip install -r requirements.txt
```


### Usage ###
```

usage: run.py [-h] [--experiment_name EXPERIMENT_NAME] [--log_dir LOG_DIR]
              [--chunk_file_location CHUNK_FILE_LOCATION]
              [--training_chunks TRAINING_CHUNKS [TRAINING_CHUNKS ...]]
              [--validation_chunks VALIDATION_CHUNKS [VALIDATION_CHUNKS ...]]
              [--test_chunks TEST_CHUNKS [TEST_CHUNKS ...]]
              [--patch_pattern PATCH_PATTERN]
              [--subtypes SUBTYPES [SUBTYPES ...]] [--num_classes NUM_CLASSES]
              [--num_patch_workers NUM_PATCH_WORKERS]
              [--batch_size BATCH_SIZE] [--eval_batch_size EVAL_BATCH_SIZE]
              [--seed SEED]
              [--backbone {alexnet,vgg16,vgg19,vgg16_bn,vgg19_bn,resnet18,resnet34,resnet50,resnext50_32x4d,resnext101_32x8d,mobilenet_v2,mobilenet_v3_small,mobilenet_v3_large,mnasnet1_3,shufflenet_v2_x1_5,squeezenet1_1,efficientnet-b0,efficientnet-l2,efficientnet-b1,efficientnet-b2,efficientnet-b3,efficientnet-b4,efficientnet-b5,efficientnet-b6,efficientnet-b7,efficientnet-b8,vit_deit_small_patch16_224}]
              [--external_test_name EXTERNAL_TEST_NAME]
              [--external_chunk_file_location EXTERNAL_CHUNK_FILE_LOCATION]
              [--external_chunks EXTERNAL_CHUNKS [EXTERNAL_CHUNKS ...]]
              {calculate-representation,train-attention} ...

AttentionMIL

positional arguments:
  {calculate-representation,train-attention}
                        Whether to calculate the representation or train the
                        model
    calculate-representation
                        Calculating patchs' embeddings of extracted patches.
    train-attention     Training AttentionMIL

options:
  -h, --help            show this help message and exit
  --experiment_name EXPERIMENT_NAME
                        Experiment's name that is utiziled as the name of a
                        directory in the 'log_dir' location.
  --log_dir LOG_DIR     Directory in which checkpoints and all information are
                        stored.
  --chunk_file_location CHUNK_FILE_LOCATION
                        Path to the JSON file containing patches address
  --training_chunks TRAINING_CHUNKS [TRAINING_CHUNKS ...]
                        Space separated IDs specifying chunks included in
                        training.
  --validation_chunks VALIDATION_CHUNKS [VALIDATION_CHUNKS ...]
                        Space separated IDs specifying chunks included in
                        validation.
  --test_chunks TEST_CHUNKS [TEST_CHUNKS ...]
                        Space separated IDs specifying chunks included in
                        test.
  --patch_pattern PATCH_PATTERN
                        Patterns of the stored patches, which is used to
                        extracted information such as slide's ID from the
                        path.
  --subtypes SUBTYPES [SUBTYPES ...]
                        Space separated words describing subtype=groupping
                        pairs for this study.
  --num_classes NUM_CLASSES
                        Number of output classes, i.e., the number of
                        subtypes.
  --num_patch_workers NUM_PATCH_WORKERS
                        Number of data loading workers.
  --batch_size BATCH_SIZE
                        Batch size for the trianing phase.
  --eval_batch_size EVAL_BATCH_SIZE
                        Batch size for the validation and testing phase.
  --seed SEED           Seed for initializing training.
  --backbone {alexnet,vgg16,vgg19,vgg16_bn,vgg19_bn,resnet18,resnet34,resnet50,resnext50_32x4d,resnext101_32x8d,mobilenet_v2,mobilenet_v3_small,mobilenet_v3_large,mnasnet1_3,shufflenet_v2_x1_5,squeezenet1_1,efficientnet-b0,efficientnet-l2,efficientnet-b1,efficientnet-b2,efficientnet-b3,efficientnet-b4,efficientnet-b5,efficientnet-b6,efficientnet-b7,efficientnet-b8,vit_deit_small_patch16_224}
                        Model architecture inorder to find the dimension of
                        embedding.
  --external_test_name EXTERNAL_TEST_NAME
                        Usefull when testing on multiple external datasets.
  --external_chunk_file_location EXTERNAL_CHUNK_FILE_LOCATION
                        Path to JSON file contains external dataset
  --external_chunks EXTERNAL_CHUNKS [EXTERNAL_CHUNKS ...]
                        Space separated number IDs specifying chunks to use
                        for testing (default use all the slides).

usage: run.py calculate-representation [-h] [--resize RESIZE]
                                       [--method {Vanilla}]
                                       [--saved_model_location SAVED_MODEL_LOCATION]
                                       [--state {train,validation,test,external}]
                                       [--slide_id SLIDE_ID]

options:
  -h, --help            show this help message and exit
  --resize RESIZE       If the value is true, the extracted patches are
                        resized prior to being fed to the network. For
                        example, setting this value to 224 is required for
                        ViTs.
  --method {Vanilla}    The network structure used to calculate embedding;
                        now, only the Vanilla version is supplied, but this
                        can be expanded to incorporate other networks such as
                        self-supervised.
  --saved_model_location SAVED_MODEL_LOCATION
                        Path to the saved trained model for extracting
                        embeddings.
  --state {train,validation,test,external}
                        Specify the data state from train, validation, test,
                        and external.Setting this flag causes the patches to
                        be phase-restitched to the specific state. It is
                        advantageous while using multithreading. For instance,
                        we might set this flag to train if we're just
                        interested in calculating embeddings for training
                        patches.
  --slide_id SLIDE_ID   Identical to the preceding flag, it can be used for
                        multithreading. This parameter restricts the embedding
                        calculation to a single slide when set.

usage: run.py train-attention [-h] [--epochs EPOCHS] [--lr LR] [--wd WD]
                              [--optimizer {Adam,AdamW,SGD}]
                              [--patience PATIENCE]
                              [--lr_patience LR_PATIENCE]
                              [--not_use_weighted_loss] [--use_schedular]
                              [--criteria {overall_auc,overall_acc,balanced_acc,All}]
                              [--only_test] [--only_external_test]
                              {VarMIL} ...

positional arguments:
  {VarMIL}              Structure of the model: Currently, we have just
                        included VarMIL, but others such as DeepMIL can be
                        incorporated.
    VarMIL              VarMIL (variability-aware deep multiple instance
                        learning)

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of total epochs to run.
  --lr LR               Initial learning rate.
  --wd WD               Weight decay.
  --optimizer {Adam,AdamW,SGD}
                        Optimizer for training the model: 1. Adam 2. AdamW 3.
                        SGD
  --patience PATIENCE   How long to wait after last time validation loss
                        improved.
  --lr_patience LR_PATIENCE
                        How long to wait after last time validation loss
                        improved to change the learning rate.
  --not_use_weighted_loss
                        Setting this flag disables weighted loss in the code.
                        If the dataset contains a class imbalance, this flag
                        should not be set.
  --use_schedular       Using schedular for decreasig learning rate in a way
                        that if lr_patience has passed, it will be reduced by
                        0.8.
  --criteria {overall_auc,overall_acc,balanced_acc,All}
                        Criteria for saving the best model: 1. overall_auc:
                        using AUC 2. overall_acc: uses accuracy 3.
                        balanced_acc: balanced accuracy for imbalanced data 4.
                        All: uses all the possible criterias NOTE: For
                        calculating AUC for multiclasses, OVO is used to
                        mitigate the imbalanced classes.
  --only_test           Only test not train.
  --only_external_test  Only test on the external dataset.

usage: run.py train-attention VarMIL [-h]

options:
  -h, --help  show this help message and exit

```

