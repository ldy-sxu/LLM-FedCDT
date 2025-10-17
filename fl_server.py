import torch
from fedlab.contrib.algorithm.fedavg import FedAvgServerHandler
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.utils.aggregator import Aggregators
from model_utils import _serialize_model_parameters_custom, _deserialize_model_parameters_custom
from typing import cast


class FedAvgServer(FedAvgServerHandler):
    def __init__(self, model: torch.nn.Module, global_rounds: int, sample_ratio: float,
                 device: torch.device):
        super().__init__(model=model, global_round=global_rounds, sample_ratio=sample_ratio,
                         cuda=(device.type == 'cuda'), device=str(device))
        self.device = device
        self._model.to(self.device)

    def global_update(self, buffer: list):
        parameters_list_of_dicts = [ele[0] for ele in buffer]
        sample_num_list = [ele[1] for ele in buffer]

        serialized_client_params = [_serialize_model_parameters_custom(p) for p in parameters_list_of_dicts]

        total_samples = sum(sample_num_list)
        weights = [float(num) / total_samples for num in sample_num_list]

        aggregated_serialized_params = Aggregators.fedavg_aggregate(
            serialized_client_params, weights=weights
        )

        if self.device.type == 'cuda':
            aggregated_serialized_params = aggregated_serialized_params.to(self.device)

        aggregated_model_state_dict = _deserialize_model_parameters_custom(
            aggregated_serialized_params, self._model.state_dict()
        )

        self._model.load_state_dict(aggregated_model_state_dict)

        self._buffer = []