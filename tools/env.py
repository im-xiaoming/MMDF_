# import os
# import re
# import torch
# import torch.distributed as dist


# def init_dist(args):
#     """Initialize distributed computing environmen
#     t."""
#     args.ngpus_per_node = torch.cuda.device_count()

#     if args.launcher == 'pytorch':
#         _init_dist_pytorch(args)
#     else:
#         raise ValueError('Invalid launcher type: {}'.format(args.launcher))


# def _init_dist_pytorch(args, **kwargs):
#     """Set up environment."""
#     # TODO: use local_rank instead of rank % num_gpus
#     args.rank = args.rank * args.ngpus_per_node + args.gpu
#     args.world_size = args.world_size
#     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
#                             world_size=args.world_size, rank=args.rank)
#     torch.cuda.set_device(args.gpu)
#     print(f"{args.dist_url}, ws:{args.world_size}, rank:{args.rank}")

#     if args.rank % args.ngpus_per_node == 0:
#         args.log = True
#     else:
#         args.log = False
