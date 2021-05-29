class CLASS_DATASET_BASE:
    def __init__(self,name,
                 dataset_root,
                 split, #train, val, test
                 batch_size,
                 input_size,
                 max_len,
                 vocab_size):
        self.name_ = name
        self.dataset_root_ = dataset_root
        self.split_ = split
        self.batch_size_ = batch_size
        self.input_size_ = input_size
        self.max_len_ = max_len
        self.vocab_size_ = vocab_size
        return
    def getStepPerEpoch(self):
        return 0
    def getDataset(self):
        return None