REGISTRY = {}

from .rnn_agent import RNNAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["ff"] = FFAgent
REGISTRY["central_rnn"] = CentralRNNAgent

from .updet_agent import UPDeT
from .transformer_agg_agent import TransformerAggregationAgent
REGISTRY['updet'] = UPDeT
REGISTRY['transformer_aggregation'] = TransformerAggregationAgent