import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Tmall', help='dataset name: Tmall/Cosmetics/CIKM19')
    parser.add_argument('--pre-dataset-dir', default='pre_datasets/{dataset}', help='the folder to save the preprocessed dataset')
    parser.add_argument('--embeddings-dir', default='saved_embeddings/{dataset}')
    parser.add_argument('--models-dir', default='saved_models/{dataset}')
    parser.add_argument('--model', default='Model', help='the dataset directory')
    parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=100, help='input batch size')
    parser.add_argument('--item-dim', type=int, default=100, help='embedding size')
    parser.add_argument('--attribute-dim', type=int, default=300)
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--validation', action='store_true', help='validation')
    parser.add_argument('--valid-portion', type=float, default=0.1, help='split the portion')

    parser.add_argument('--num-items', type=int, default=0, help='the number of items')
    parser.add_argument('--num-cats', type=int, default=0, help='the number of cats')
    parser.add_argument('--num-sellers', type=int, default=0, help='the number of sellers')
    parser.add_argument('--num-brands', type=int, default=0, help='the number of brands')

    parser.add_argument('--num-conv-layers', type=int, default=2, help='No. of ItemConv layers')

    parser.add_argument('--num-layers', type=int, default=3, help='No. of self-attention layers')
    parser.add_argument('--num-heads', type=int, default=3, help='No. of attention heads')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability for each deep layer')

    parser.add_argument('--num-coarse-sampling', type=int, default=100, help='No. of coarse-sampling')
    parser.add_argument('--num-expanded-samples', type=int, default=10, help='No. of expanded samples')
    parser.add_argument('--tau', type=float, default=0.2, help='temperature parameter')
    parser.add_argument('--beta', type=float, default=0.02, help='Weight for Sup CL task')
    parser.add_argument('--gamma', type=float, default=0.005, help='Weight for Unsup CL task')  # 0.005
    parser.add_argument('--delta', type=float, default=0.1, help='Weight for Attribute Prediction task')  # 1

    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

    parser.add_argument('--random-seed', type=float, default=2023, help='random seed for the model')

    return parser.parse_args()
