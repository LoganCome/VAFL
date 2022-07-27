import inspect

# Dependencies
import math
import sys
import pandas as pd
import asyncio
import logging
import time

logger = logging.getLogger("run_websocket_client")

import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
from syft.frameworks.torch.fl import utils

import torch
import numpy as np

import run_websocket_client as rwc

path = "./result"


async def main():
    # 创建csv文件
    global old_model
    df = pd.DataFrame(columns=['step',
                               'acc1', 'acc2',
                               'acc3', 'accf'])
    t = str(time.time())
    df.to_csv(path + "/" + t + ".csv", index=False)
    iter = 0

    # Hook torch
    hook = sy.TorchHook(torch)

    # Arguments
    args = rwc.define_and_get_arguments(args=[])
    use_cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(args)

    # Configure logging

    if not len(logger.handlers):
        FORMAT = "%(asctime)s - %(message)s"
        DATE_FMT = "%H:%M:%S"
        formatter = logging.Formatter(FORMAT, DATE_FMT)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    LOG_LEVEL = logging.DEBUG
    logger.setLevel(LOG_LEVEL)

    t0 = time.time()

    akwargs_websocket = {"host": "192.168.2.13", "hook": hook, "verbose": args.verbose}
    alice = WebsocketClientWorker(id="alice", port=8777, **akwargs_websocket)

    bkwargs_websocket = {"host": "192.168.2.16", "hook": hook, "verbose": args.verbose}
    bob = WebsocketClientWorker(id="bob", port=8778, **bkwargs_websocket)

    ckwargs_websocket = {"host": "192.168.2.14", "hook": hook, "verbose": args.verbose}
    charlie = WebsocketClientWorker(id="charlie", port=8779, **ckwargs_websocket)

    dkwargs_websocket = {"host": "192.168.2.11", "hook": hook, "verbose": args.verbose}
    #dave = WebsocketClientWorker(id="dave", port=8780, **dkwargs_websocket)

    ekwargs_websocket = {"host": "192.168.2.12", "hook": hook, "verbose": args.verbose}
    #eva = WebsocketClientWorker(id="eva", port=8781, **ekwargs_websocket)

    kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": args.verbose}
    #frank = WebsocketClientWorker(id="frank", port=8782, **kwargs_websocket)
    #frank1 = WebsocketClientWorker(id="frank1", port=8792, **kwargs_websocket)

    testing = WebsocketClientWorker(id="testing", port=8783, **kwargs_websocket)

    worker_instances = [
        alice,
        bob,
        charlie,
        # dave,
        # eva,
        # frank,
        # frank1
    ]

    model = rwc.Net().to(device)
    # print(model)

    print("Federate_after_n_batches: " + str(args.federate_after_n_batches))
    print("Batch size: " + str(args.batch_size))
    print("Initial learning rate: " + str(args.lr))

    learning_rate = args.lr

    traced_model = torch.jit.trace(model, torch.zeros([1, 1, 28, 28], dtype=torch.float))

    for curr_round in range(1, args.training_rounds + 1):

        '''OLD MODEL'''
        Empty_model = utils.scale_model(model, 0)
        old_model = utils.add_model(Empty_model, traced_model)

        # train
        logger.info("Training round %s/%s", curr_round, args.training_rounds)
        results = await asyncio.gather(
            *[
                rwc.fit_model_on_worker(
                    worker=worker,
                    traced_model=traced_model,
                    batch_size=args.batch_size,
                    curr_round=curr_round,
                    max_nr_batches=args.federate_after_n_batches,
                    lr=learning_rate,
                )
                for worker in worker_instances
            ]
        )

        '''
        models, loss, grads, n_server
        
        values: V
        V[worker_id] represents the value of a participant, 
        and the smaller the value, the lower the need for its inclusion in the federation model
        How to calculate V?
        1) grad: the gradient reflects the proximity to the local solution
        2) total number of client: In federation learning, the more the number of participants, 
        the smaller the value of a single participant
        3) accuracy in testing
        '''

        models = {}
        loss_values = {}
        grads = {}
        n_server = 3
        acc = {}

        V = {}

        # test
        test_models = curr_round % 5 == 1 or curr_round == args.training_rounds
        if test_models:

            logger.info("Evaluating models")
            np.set_printoptions(formatter={"float": "{: .0f}".format})
            for worker_id, worker_model, _ in results:
                acc[worker_id] = rwc.evaluate_model_on_worker(
                    model_identifier="Model update " + worker_id,
                    worker=testing,
                    dataset_key="mnist_testing",
                    model=worker_model,
                    nr_bins=10,
                    batch_size=128,
                    print_target_hist=False,
                )
                new_model = worker_model
                grads[worker_id] = utils.add_model(new_model, utils.scale_model(old_model, -1))

                grad = 0
                for p in grads[worker_id].parameters():
                    p2 = torch.dot(p.view(-1), p.view(-1))
                    grad += p2

                print(grad.detach().numpy())
                grad = grad.detach().numpy()

                V[worker_id] = grad * pow(1 + n_server / 1000, acc[worker_id])

            Vlist = list(V.values())

            ave_V = np.mean(Vlist)

            # Federal model (this operation changes the initial model)
            for worker_id, worker_model, worker_loss in results:
                if worker_model is not None:
                    loss_values[worker_id] = worker_loss

                    if V[worker_id] >= ave_V:
                        models[worker_id] = worker_model

            iter += len(models.keys())
            # federated_avg
            traced_model = utils.federated_avg(models)

        if test_models:
            accf = rwc.evaluate_model_on_worker(
                model_identifier="Federated model",
                worker=testing,
                dataset_key="mnist_testing",
                model=traced_model,
                nr_bins=10,
                batch_size=128,
                print_target_hist=False,
            )

            step = curr_round
            acc1 = acc['alice']
            acc2 = acc['bob']
            acc3 = acc['charlie']
            '''
            acc4 = acc['dave']
            acc5 = acc['eva']
            acc6 = acc['frank']
            acc7 = acc['frank1']
            '''

            scv_list = [step, acc1, acc2, acc3, accf]
            scv_data = pd.DataFrame([scv_list])
            scv_data.to_csv(path + "/" + t + ".csv", mode='a', header=False, index=False)

        # decay learning rate
        learning_rate = max(0.98 * learning_rate, args.lr * 0.01)
        if accf is not None and accf >= 94.0:
            break

    torch.save(model.state_dict(), "mnist_cnn.pt")

    # time
    t1 = time.time()
    logger.info('cost:%.4f seconds' % (float(t1 - t0)))
    logger.info('iter:%d' % iter)


if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    # Websockets setup
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())

    # Run main
    asyncio.get_event_loop().run_until_complete(main())
