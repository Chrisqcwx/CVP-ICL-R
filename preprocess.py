import hyp
from cvpiclr.dataset.preprocess import preprocess

if __name__ == '__main__':
    preprocess(hyp.anno_path, hyp.raw_dataset_dir, hyp.detection_dataset_dir)
