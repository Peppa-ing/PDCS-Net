import argparse

# 超参数配置
parser = argparse.ArgumentParser(description='Hyper-parameters management')


# data in/out and dataset
parser.add_argument('--root_path', default=r'/home/spgou/GYT/PDASNet_plaque_segmentation', help='project root path')
parser.add_argument('--original_datasets_path', default=r'/home/spgou/GYT/PDASNet_plaque_segmentation/plaque datasets', help='plaque datasets path')
parser.add_argument('--precessed_datasets_path', default=r'/home/spgou/GYT/PDASNet_plaque_segmentation/processed_datasets', help='processed plaque datasets path')


args = parser.parse_args()
