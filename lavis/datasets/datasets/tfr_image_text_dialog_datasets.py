import os
from collections import OrderedDict
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
from PIL import Image
import torch.distributed as dist
import json
from .example_pb2 import Example
import dareblopy as db
from io import BytesIO
import numpy as np
import torch
import glob
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.common.dist_utils import get_world_size, get_rank


def read_index_file(index_paths, tfr_paths, global_rank, world_size):
    """ Parse index file, each line contains record_name, record_index, record_offset and label
    index_paths: ann_paths, tfr_paths: vis_root
    """
    samples_offsets = []
    record_files = []
    labels = []
    recordreaders = {}
    record_paths = {}
    dialogs_to_ITpair = {}
    ITpair_to_dialogs = {}
    dialog2type = {}
    type2dialog = {}
    data_type = []
    rank = 0
    dialogs_id = 0
    for data_no, index_path in enumerate(index_paths):
        index_files = glob.glob(index_path) 
        root_dir = tfr_paths[data_no]
        idx = -1
        for index_file in index_files:
            idx += 1
            with open(index_file, 'r') as ifs:
                for line in ifs:
                    if global_rank == rank:
                        dialog_group = line.rstrip().split('\t')
                        if len(dialog_group) % 4 != 0:
                            continue
                        dialog_data_type = dialog_group[0]
                        dialog_group = [dialog_group[dgi:dgi+4] for dgi in range(0,len(dialog_group),4)]
                        
                        dialog_id_mapping = []
                        for dialog_i in dialog_group:
                            record_name, tfrecord_index, tfrecord_offset, label = dialog_i
                            sample_id = len(samples_offsets)
                            ITpair_to_dialogs[sample_id] = dialogs_id
                            dialog_id_mapping.append(sample_id)
                            samples_offsets.append(int(tfrecord_offset))
                            record_file = "%s-%05d.tfrecord" % (record_name, int(tfrecord_index))
                            record_files.append(record_file)
                            if record_file not in record_paths:
                                record_file_path = os.path.join(root_dir, record_file)

                                record_paths[record_file] = record_file_path
                                if record_file not in recordreaders:
                                    recordreaders[record_file] = db.RecordReader(record_file_path)
                            labels.append(label)
                            data_type.append(record_name)
                        dialog2type[dialogs_id] = dialog_data_type
                        type2dialog.setdefault(dialog_data_type, [])
                        type2dialog[dialog_data_type].append(dialogs_id)
                        dialogs_to_ITpair[dialogs_id] = dialog_id_mapping
                        dialogs_id += 1
                    rank = (rank + 1) % world_size
    return record_files, samples_offsets, labels, \
            recordreaders, record_paths, \
            dialogs_to_ITpair, ITpair_to_dialogs, \
            data_type, dialog2type, type2dialog

class IndexTFRDataset(Dataset):
    """ Index TFRecord Dataset
    """
    def __init__(self, index_paths, tfr_paths, global_rank, world_size):
        """ Create a ``IndexTFRDataset`` object
            A ``IndexTFRDataset`` object will read sample proto from *.tfrecord files saved
            in ``tfrecord_dir`` by index_file, the sample proto will convert to image and
            fed into Dataloader.

            Args:
                tfrecord_dir: tfrecord saved dir
                index_file: each line format ``tfr_name\t tfr_record_index \t tfr_record_offset \t label``
                read_label_from_tfr: read label from tfrecord instead of index file for memory saving
        """
        self.records, self.offsets, self.labels, \
            self.recordreaders, self.record_paths, \
                self.dialogs_to_ITpair, self.ITpair_to_dialogs, \
                    self.data_type, self.dialog2type, self.type2dialog = read_index_file(index_paths, tfr_paths, global_rank, world_size)
        self.sample_num = len(self.records)

        assert len(self.labels) > 0

    def __len__(self):
        return self.sample_num

    def _get_record(self, record, offset):
        try:
            pb_data = self.recordreaders[record].read_record(offset)
        except:
            self.recordreaders[record] = db.RecordReader(self.record_paths[record])
            pb_data = self.recordreaders[record].read_record(offset)

        example = Example()
        example.ParseFromString(pb_data)
        feature = example.features.feature

        image = Image.open(BytesIO(feature['image'].bytes_list.value[0])).convert('RGB')
        return image

    def __getitem__(self, index):
        if index == -1 or index == len(self.offsets)-1:
            #* return placeholder img&txt
            image = None
            label = ''
            dialog_id = -1
            data_type = None
        else:
            offset = self.offsets[index]
            record = self.records[index]
            image = self._get_record(record, offset)
            label = self.labels[index]
            dialog_id = self.ITpair_to_dialogs[index]
            data_type = self.data_type[index]
        return image, label, dialog_id, data_type
    


class TfrBaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root
        self.annotation = IndexTFRDataset(ann_paths, vis_root, get_rank(), get_world_size())
        self.dialogs_to_ITpair = self.annotation.dialogs_to_ITpair
        self.dialog2type = self.annotation.dialog2type
        self.type2dialog = self.annotation.type2dialog

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        # self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        raise NotImplementedError



class ImageTextDialogDataset(TfrBaseDataset):
    #* for self-made dialog datasets from image-text pairs
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.image_size = self.vis_processor.image_size
    
    def get_dialogs_to_ITpair(self):
        return self.dialogs_to_ITpair
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        
        if ann[0] is None:
            image = torch.zeros(3,self.image_size,self.image_size)
        else:
            image = self.vis_processor(ann[0])
        caption = self.text_processor(ann[1])
        dialog_id = int(ann[2])
        data_type = ann[3]

        return {
            "image": image,
            "text_input": caption,
            "dialog_id": dialog_id,
            "data_type": data_type,
        }
