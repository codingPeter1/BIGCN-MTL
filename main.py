import argparse
import random
import time
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from model import I_CRGCN
import os
from os.path import join

from data_set import DataSet
from trainer import Trainer

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--reg_weight', type=float, default=1e-3, help='')
    parser.add_argument('--ssl_weight', type=float, default=0, help='')
    parser.add_argument('--his_weight', type=float, default=0.0, help='')
    parser.add_argument('--kd_weight', type=float, default=0.01, help='')
    parser.add_argument('--kl_weight', type=float, default=0.0, help='')
    parser.add_argument('--ort_weight', type=float, default=0.0, help='')
    parser.add_argument('--ce_weight', type=float, default=0.0, help='')
    parser.add_argument('--bcr_weight', type=float, default=0.0, help='')
    parser.add_argument('--inner_weight', type=float, default=0.0, help='')
    parser.add_argument('--tao', type=float, default=1e-1, help='')
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--node_dropout', type=float, default=0.0)
    parser.add_argument('--message_dropout', type=float, default=0.0)

    parser.add_argument('--data_name', type=str, default='jdata', help='')
    parser.add_argument('--stage', type=int, default=5, help='')
    parser.add_argument('--behaviors', help='', action='append')
    parser.add_argument('--if_load_model', type=bool, default=True, help='')

    # parser.add_argument('--topk', type=list, default=[10, 20, 50, 80], help='')
    parser.add_argument('--topk', type=list, default=[5, 10, 20, 50], help='')
    # parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='')
    parser.add_argument('--metrics', type=list, default=['ndcg', 'recall'], help='')
    parser.add_argument('--lr', type=float, default=0.01, help='')
    parser.add_argument('--decay', type=float, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--test_batch_size', type=int, default=3072, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=str, default=250, help='')
    parser.add_argument('--model_path', type=str, default='./check_point/', help='')
    parser.add_argument('--emb_saved_path', type=str, default='./embeddings_save/', help='')
    parser.add_argument('--degree_path', type=str, default='/degree', help='')
    parser.add_argument('--check_point', type=str, default='', help='')
    parser.add_argument('--model_name', type=str, default='', help='')
    parser.add_argument('--device', type=str, default='cpu', help='')

    args = parser.parse_args()
    if args.data_name == 'JD_2':
        args.data_path = './data/JD/'
        args.behaviors = ['click', 'fav', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'JD_2'
    elif args.data_name == 'JD_3':
        args.data_path = './data/JD/'
        args.behaviors = ['click', 'fav', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'JD_3'
    elif args.data_name == 'JD_4':
        args.data_path = './data/JD/'
        args.behaviors = ['click', 'fav', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'JD_4'
    elif args.data_name == 'UB_2':
        args.data_path = './data/UB/'
        args.behaviors = ['pv', 'fav', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'UB_2'
    elif args.data_name == 'UB_3':
        args.data_path = './data/UB/'
        args.behaviors = ['pv', 'fav', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'UB_3'
    elif args.data_name == 'UB_4':
        args.data_path = './data/UB/'
        args.behaviors = ['pv', 'fav', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'UB_4'
    elif args.data_name == 'Tmall_2':
        args.data_path = './data/Tmall/'
        args.behaviors = ['click', 'fav', 'buy']
        args.layers = [1, 1, 1]
        args.model_name = 'Tmall_2'
    elif args.data_name == 'Tmall_3':
        args.data_path = './data/Tmall/'
        args.behaviors = ['click', 'fav', 'buy']
        args.layers = [1, 1, 1]
        args.model_name = 'Tmall_3'
    elif args.data_name == 'Tmall_4':
        args.data_path = './data/Tmall/'
        args.behaviors = ['click', 'fav', 'buy']
        args.layers = [1, 1, 1]
        args.model_name = 'Tmall_4'
    elif args.data_name == 'Rees46_2':
        args.data_path = './data/Rees46/'
        args.behaviors =['view', 'cart', 'buy']
        args.layers = [1, 1, 1]
        args.model_name = args.data_name
    elif args.data_name == 'Rees46_3':
        args.data_path = './data/Rees46/'
        args.behaviors =['view', 'cart', 'buy']
        args.layers = [1, 1, 1]
        args.model_name = args.data_name
    elif args.data_name == 'Rees46_4':
        args.data_path = './data/Rees46/'
        args.behaviors =['view', 'cart', 'buy']
        args.layers = [1, 1, 1]
        args.model_name = args.data_name
    else:
        raise Exception('data_name cannot be None')


    layer_embeddings_load_path = join(f'./embeddings_save/{args.data_name}',
                                      f"Layer_embeddings_at_stage_{args.stage - 1}_es50.pth")
    embeddings_load_path = join(f'./embeddings_save/{args.data_name}', f"Embeddings_at_stage_{args.stage - 1}_es50.pth")


    LastStage_embeddings = torch.load(layer_embeddings_load_path, map_location=torch.device(args.device))
    current_time = datetime.now() + timedelta(hours=8)
    TIME = current_time.strftime("%Y-%m-%d %H_%M_%S")
    args.TIME = TIME
    logfile = '{}_lr_{}_reg_{}_his_{}_kd_{}_tao_{}_emb_{}_{}'.format(args.model_name, args.lr, args.reg_weight, args.his_weight, args.kd_weight, args.tao, args.embedding_size,TIME)
    args.train_writer = SummaryWriter('./log/train/' + logfile)
    args.test_writer = SummaryWriter('./log/test/' + logfile)
    logger.add('./log/{}/{}.log'.format(args.model_name, logfile), encoding='utf-8')

    start = time.time()
    dataset = DataSet(args)
    logger.info("SEED:"+str(SEED))
    model = I_CRGCN(args, dataset, LastStage_embeddings, embeddings_load_path)
    logger.info(model)
    trainer = Trainer(model, dataset, args)
    saved_dict = trainer.train_model()

    # 选择最好的模型保存每一层的emb
    embeddings_save_path = os.path.join(args.emb_saved_path+args.data_name, f"{args.lr}_{args.reg_weight}_{args.his_weight}_{args.kd_weight}_{args.tao}_Layer_embeddings_at_stage_{args.stage}_es50.pth")
    torch.save(saved_dict, embeddings_save_path)

    logger.info('train end total cost time: {}'.format(time.time() - start))



