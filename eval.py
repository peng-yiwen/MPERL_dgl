import argparse
from dgl.dataloading import NodeDataLoader, MultiLayerFullNeighborSampler
from utils import *
from RGCP import PonderRelationalGraphConvModel


def main(args):
    use_cuda = args['cuda'] and torch.cuda.is_available()
    data_path = os.path.join(args['data_dir'], args['dataset'])

    # graph
    e2id = read_id(os.path.join(data_path, 'entities.tsv'))
    r2id = read_id(os.path.join(data_path, 'relations.tsv'))
    r2id['type'] = len(r2id)
    t2id = read_id(os.path.join(data_path, 'types.tsv'))
    num_entity = len(e2id)
    num_rels = len(r2id)
    num_types = len(t2id)
    num_nodes = num_entity + num_types # if load_ET
    g, _, _, all_true, _, _, test_id = load_graph(data_path, e2id, r2id, t2id, args['load_ET'], args['load_KG'])
    test_sampler = MultiLayerFullNeighborSampler(args['num_layers']) # 2 layers

    # create model
    if args['model'] == 'RGCP':
        model = PonderRelationalGraphConvModel(args['hidden_dim'], num_nodes, num_rels, num_types, num_layers=args['num_layers'], max_steps=args['max_markov_steps'],
                                               num_bases=args['num_bases'], self_loop=args['selfloop'], dropout=args['drop'], seed=args['seed'], cuda=use_cuda)
    else:
        raise ValueError('Unknown model: %s' % args['model'])
    
    if args['dataset'] == 'FB15kET':
            model_name = 'rgcp_fb.pkl'
    elif args['dataset'] == 'YAGO43kET':
        model_name = 'rgcp_yago.pkl'
    else:
        raise ValueError('Unknown dataset: %s' % args['dataset'])

    print('Start evaluation on %d testing data of %s dataset' % (len(test_id), args['dataset']))
    with torch.no_grad():
        if not use_cuda:
            model.load_state_dict(torch.load(model_name), map_location=torch.device('cpu'))
        else:
            model.load_state_dict(torch.load(model_name))
        
        model.eval()
        predict = torch.zeros(num_entity, num_types, dtype=torch.float32)
        test_dataloader = NodeDataLoader(
            g, test_id, test_sampler,
            batch_size=args['test_batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=6
        )
        for input_nodes, output_nodes, blocks in test_dataloader:
            emb_test = model(blocks)[0].cpu().float()
            emb_test = torch.mean(emb_test, 0)
            predict[output_nodes] = emb_test
        evaluate(os.path.join(data_path, 'ET_test.txt'), predict, all_true, e2id, t2id)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RGCP')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='FB15kET')
    parser.add_argument('--load_ET', action='store_true', default=True)
    parser.add_argument('--load_KG', action='store_true', default=True)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_bases', type=int, default=45)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--selfloop', action='store_true', default=True)
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lambda_p', type=float, default=0.2)
    parser.add_argument('--max_markov_steps', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', default=False)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        params = vars(get_params())
        set_logger(params)
        main(params)
    except Exception as e:
        logging.exception(e)
        raise
