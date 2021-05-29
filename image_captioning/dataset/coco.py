import os,sys
from tqdm import tqdm
import sklearn
import tensorflow as tf
from .base import CLASS_DATASET_BASE
import time,json


DEBUG_CODE = False

CLASS_DATASET_COCO_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
CLASS_DATASET_COCO_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

MAX_LEN = 34
VOCAB_SIZE = 2500

SEED = 224

STOP_WORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'}

#不做增广加载图片
def loadImage(image_path, input_size):
    data = tf.io.read_file(image_path)
    data = tf.image.decode_jpeg(data, channels=3)
    data = tf.image.resize(data, input_size)
    data = (data/255.0 - CLASS_DATASET_COCO_MEAN) / CLASS_DATASET_COCO_STD
    return data


class CLASS_COCO2014(CLASS_DATASET_BASE):
    def __init__(self,
                 dataset_root,
                 split,  # train, val, test
                 batch_size,
                 input_size, #(h,w)
                 char_level = False,
                 tokenizer = None
                 ):
        super(CLASS_COCO2014, self).__init__("coco2014",dataset_root,split,batch_size,input_size,MAX_LEN,VOCAB_SIZE)
        self.annotation_file = os.path.join(dataset_root,"annotations",f"captions_{split}2014.json")
        self.image_dir = os.path.join(dataset_root,f"{split}2014/")
        self.image_paths, self.image_captions = [],[]
        self.char_level = char_level
        self.tokenizer = tokenizer
        self._loadAnnotation()
    def getStepPerEpoch(self):
        return len(self.image_captions) // self.batch_size_

    def _loadImage(self,image_path):
        data = tf.io.read_file(image_path)
        data = tf.image.decode_jpeg(data, channels=3)
        data = tf.image.resize(data, self.input_size_)
        data = (data/255.0 - CLASS_DATASET_COCO_MEAN ) / CLASS_DATASET_COCO_STD
        return data
    def _loadAnnotation(self):
        with open(self.annotation_file,'r') as f:
            annos_all = json.load(f)
        self.image_paths, self.image_captions = [], []
        for annot in tqdm(annos_all['annotations'],desc="load annotation..."):
            caption = '<start> ' + annot['caption'] + ' <end>'
            caption = caption.lower()
            caption = list(filter(
               lambda x: x  not in STOP_WORDS, caption.lower().split(' ')
            ))
            caption = ' '.join(caption)


            image_id = annot['image_id']
            full_coco_image_path = self.image_dir + f'COCO_{self.split_}2014_' + '%012d.jpg' % (image_id)

            self.image_paths.append(full_coco_image_path)
            self.image_captions.append(caption)
        self.image_paths, self.image_captions = sklearn.utils.shuffle(self.image_paths, self.image_captions,random_state=SEED)
        if DEBUG_CODE:
            self.image_paths, self.image_captions = self.image_paths[0:100000], self.image_captions[0:100000]

        ####################################
        #convert caption from text to index
        if self.tokenizer is None: #share tokenizer between train/test
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
                char_level = self.char_level,
                num_words=self.vocab_size_, #dataset dependent
                oov_token="<unk>",
                filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
            self.tokenizer.fit_on_texts(self.image_captions)
        self.image_captions = self.tokenizer.texts_to_sequences(self.image_captions)
        #n = sorted(list(map(lambda x: len(x), self.image_captions)))[::-1]
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = "<pad>"
        if len(self.tokenizer.index_word.keys()) < self.vocab_size_:
            self.vocab_size_ = len(self.tokenizer.index_word.keys())
        self.image_captions = tf.keras.preprocessing.sequence.pad_sequences(
            self.image_captions,
            padding="post", #add  pad at end of string
            value = 0.0, #pad value
            truncating = "post", #remove ending part if longer than maxlen
            maxlen=self.max_len_
        )
        return

    def getDataset(self):
        @tf.function
        def _convertData(image_path, caption):
            #print(image_path)
            image_data = self._loadImage(image_path)
            return image_data, caption

        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  #to allow order-altering optimizations

        dataset = tf.data.Dataset.from_tensor_slices(
            (self.image_paths, self.image_captions)
        ).shuffle(1024)
        dataset = dataset.with_options(ignore_order)
        #dataset = dataset.repeat()
        dataset = dataset.map(
            _convertData,
            #num_parallel_calls=tf.data.experimental.AUTOTUNE
            num_parallel_calls=3
        )
        dataset = dataset.batch(self.batch_size_,drop_remainder=True).prefetch(self.batch_size_ * 2)
        return dataset

def DatasetTestbed(dataset, epoch_total):
    for epoch_k in range(epoch_total):
        t0_epoch = time.perf_counter()
        for batch_k,(image, caption) in enumerate(dataset.take(100)):
            if epoch_k == 0 and batch_k == 0:
                print(f"image : {image.shape}, {image.dtype}")
                #print(f"caption: {caption.shape},{caption.dtype},{caption}")
                #pass
        t_epoch = round(time.perf_counter() - t0_epoch)
        print(f"epoch {epoch_k} : {t_epoch}sec")
    return

if __name__ == "__main__":
    dataset_root = os.path.join(os.environ["DATASET_ROOT_DIR"],"coco")
    train_dataset_object = CLASS_COCO2014(dataset_root, "train", 8, (224,224))
    train_dataset = train_dataset_object.getDataset()
    DatasetTestbed(train_dataset,10)