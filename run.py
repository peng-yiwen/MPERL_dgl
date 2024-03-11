import argparse
from dgl.dataloading import NodeDataLoader, MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from RGCP import PonderRelationalGraphConvModel
from losses import edl_digamma_loss, RegularizationLoss
from torch.optim.lr_scheduler import StepLR
from utils import *
import time


def main(args):
    use_cuda = args['cuda'] and torch.cuda.is_available()
    data_path = os.path.join(args['data_dir'], args['dataset'])
    save_path = os.path.join(args['save_dir'], args['dataset'])

    # graph
    e2id = read_id(os.path.join(data_path, 'entities.tsv'))
    r2id = read_id(os.path.join(data_path, 'relations.tsv'))
    r2id['type'] = len(r2id)
    t2id = read_id(os.path.join(data_path, 'types.tsv'))
    num_entity = len(e2id)
    num_rels = len(r2id)
    num_types = len(t2id)
    num_nodes = num_entity + num_types
    g, train_label, valid_label, all_true, train_id, valid_id, test_id = load_graph(data_path, e2id, r2id, t2id,
                                                                       args['load_ET'], args['load_KG'])
    if args['neighbor_sampling']:
        train_sampler = MultiLayerNeighborSampler([args['neighbor_num']] * args['num_layers'], replace=True)
    else:
        train_sampler = MultiLayerFullNeighborSampler(args['num_layers'])
    test_sampler = MultiLayerFullNeighborSampler(args['num_layers']) # 2-layers by default
    train_dataloader = NodeDataLoader(
        g, train_id, train_sampler,
        batch_size=args['train_batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=6
    )
    valid_dataloader = NodeDataLoader(
        g, valid_id, test_sampler,
        batch_size=args['test_batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=6
    )

    # model
    if args['model'] == 'RGCP':
        model = PonderRelationalGraphConvModel(args['hidden_dim'], num_nodes, num_rels, num_types, num_layers=args['num_layers'], max_steps=args['max_markov_steps'],
                                               num_bases=args['num_bases'], self_loop=args['selfloop'], dropout=args['drop'], seed=args['seed'], cuda=use_cuda)
    else:
        raise ValueError('No such model')

    if use_cuda:
        model = model.to('cuda')
    for name, param in model.named_parameters():
        logging.debug('Parameter %s: %s, require_grad=%s' % (name, str(param.size()), str(param.requires_grad)))

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args['lr'], weight_decay=args['l2']
    )

    # lr decay
    scheduler = StepLR(optimizer, step_size=args['lr_step'], gamma=0.1)

    # loss
    loss_rec = edl_digamma_loss
    loss_reg = RegularizationLoss(args['lambda_p'], args['max_markov_steps'])
    beta = 0.01 # the weight of loss_reg

    # print all paramter settings for logging
    logging.debug('---------------------------------------------------------------')
    logging.debug('Model: %s' % args['model'])
    logging.debug('Dataset: %s' % args['dataset'])
    logging.debug('Load ET: %s' % args['load_ET'])
    logging.debug('Load KG: %s' % args['load_KG'])
    logging.debug('Neighbor Sampling: %s' % args['neighbor_sampling'])
    logging.debug('Neighbor Number: %d' % args['neighbor_num'])
    logging.debug('Hidden Dimension: %d' % args['hidden_dim'])
    logging.debug('Num Bases: %d' % args['num_bases'])
    logging.debug('Activation: %s' % args['activation'])
    logging.debug('Self Loop: %s' % args['selfloop'])
    logging.debug('Dropout: %.2f' % args['drop'])
    logging.debug('Learning Rate: %.4f' % args['lr'])
    logging.debug('Learning Rate Decay Step: %d' % args['lr_step'])
    logging.debug('L2: %.5f' % args['l2'])
    logging.debug('Train Batch Size: %d' % args['train_batch_size'])
    logging.debug('Test Batch Size: %d' % args['test_batch_size'])
    logging.debug('Max Epoch: %d' % args['max_epoch'])
    logging.debug('Valid Epoch: %d' % args['valid_epoch'])
    # parameters for the Reconstruction Loss
    logging.debug('Seed: %d' % args['seed'])
    logging.debug('Lambda P: %.2f' % args['lambda_p'])
    logging.debug('Max Markov Steps: %d' % args['max_markov_steps'])
    logging.debug('---------------------------------------------------------------')

    # training
    print('Start training on %d training data of %s dataset' % (len(train_id), args['dataset']))
    max_valid_mrr = 0
    early_stop_count = 0 
    n_patience = 5
    for epoch in range(args['max_epoch']):
        t = time.time()
        model.train()
        log = []
        for input_nodes, output_nodes, blocks in train_dataloader:
            label = train_label[output_nodes, :]
            if use_cuda:
                label = label.cuda()
            emb_train, p = model(blocks) # the value of p changed from before
            emb_train = torch.mean(emb_train, 0)

            # Calculate the loss $L = L_{Rec} + \beta L_{Reg}$
            loss_ev = loss_rec(emb_train, label.float(), epoch, all_true.shape[1], 10*label.shape[1], torch.mean(p), 0)
            loss = loss_ev + beta * loss_reg(p, use_cuda)
            log.append({"loss": loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        avg_loss = sum([_['loss'] for _ in log]) / len(log)
        logging.debug('epoch %d: loss: %f time: %.4f' % (epoch, avg_loss, time.time() - t))

        # Validation
        if epoch > 500 and epoch % args['valid_epoch'] == 0:
            # Move the model to CPU
            model = model.cpu()
            model.eval()
            with torch.no_grad():
                predict = torch.zeros(num_entity, num_types, dtype=torch.float32)
                log_val = []
                for input_nodes, output_nodes, blocks in valid_dataloader:
                    val_label = valid_label[output_nodes, :]
                    emb_val, p_val = model(blocks)

                    emb_val = torch.mean(emb_val.cpu().float(), 0)
                    loss_rg = loss_reg(p_val, False).cpu()
                    p_val = p_val.cpu().float()
                    loss_ev = loss_rec(emb_val, val_label.float(), epoch, all_true.shape[1], 10*val_label.shape[1], torch.mean(p_val), 0)
                    loss_val = loss_ev + beta * loss_rg
                    log_val.append(loss_val.item())
                    
                    # predictions
                    predict[output_nodes] = emb_val
                
                loss_valid = sum(log_val) / len(log_val)
                logging.debug('epoch %d: loss_valid: %f' % (epoch, loss_valid))
                valid_mrr = evaluate(os.path.join(data_path, 'ET_valid.txt'), predict, all_true, e2id, t2id)
            
            # Move the model back to GPU
            model = model.cuda()
            model.train()
            if valid_mrr < max_valid_mrr:
                # early stop if the validation MRR is not increasing within n_patience
                early_stop_count += 1
                logging.debug('early stop count: %d' % early_stop_count)
                if early_stop_count >= n_patience:
                    break
            else:
                torch.save(model.state_dict(), os.path.join(save_path, 'R-GCP_model.pkl'))
                max_valid_mrr = valid_mrr
                early_stop_count = 0

    # testing
    print('Start evaluation on %d testing data of %s dataset' % (len(test_id), args['dataset']))
    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(save_path, 'R-GCP_model.pkl')))
        model = model.cpu()
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
    parser.add_argument('--load_ET', action='store_true', default=False)
    parser.add_argument('--load_KG', action='store_true', default=False)
    parser.add_argument('--neighbor_sampling', action='store_true', default=False)
    parser.add_argument('--neighbor_num', type=int, default=35)
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_step', type=int, default=800)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--max_epoch', type=int, default=3000)
    parser.add_argument('--valid_epoch', type=int, default=25)
    # R-GCP
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_bases', type=int, default=45)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--selfloop', action='store_true', default=False)
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lambda_p', type=float, default=0.2)
    parser.add_argument('--max_markov_steps', type=int, default=1)
    parser.add_argument('--l2', type=float, default=5e-4)
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
