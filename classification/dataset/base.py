class CLASS_DATASET_BASE:
    def __init__(self,name,
                 dataset_root,
                 split, #train, val, test
                 batch_size,
                 input_size = None):
        self.name_ = name
        self.dataset_root_ = dataset_root
        self.split_ = split
        self.batch_size_ = batch_size
        self.input_size_ = input_size
        return
    def getClassList(self):
        return 0
    def getStepPerEpoch(self):
        return 0
    def getDataset(self):
        return None