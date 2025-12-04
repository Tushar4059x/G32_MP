"""trial-1: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from main.task import Net, get_weights
from typing import List, Tuple, Optional, Dict, Union
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
import json


# Global metrics storage for plotting
METRICS_HISTORY = {
    "rounds": [],
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": [],
    "specificity": [],
    "auc_roc": [],
    "loss": [],
}


def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    """Calculate weighted average of metrics based on number of samples.
    
    Args:
        metrics: List of tuples (num_samples, metrics_dict) from each client
        
    Returns:
        Dictionary containing weighted average of all metrics
    """
    # Calculate total number of samples
    total_samples = sum(num_samples for num_samples, _ in metrics)
    
    # Initialize aggregated metrics
    aggregated = {}
    
    # Get all metric keys from first client
    if metrics:
        metric_keys = metrics[0][1].keys()
        
        # Calculate weighted average for each metric
        for key in metric_keys:
            weighted_sum = sum(num_samples * m[key] for num_samples, m in metrics)
            aggregated[key] = weighted_sum / total_samples
    
    return aggregated


class LoggingFedAvg(FedAvg):
    """Custom FedAvg strategy with round logging and metrics tracking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results with round logging."""
        self.current_round = server_round
        print("\n" + "="*80)
        print(f"🔄 FEDERATED LEARNING - ROUND {server_round}")
        print("="*80)
        print(f"📊 Training Phase: {len(results)} hospitals completed local training")
        
        # Call parent aggregate_fit
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_metrics and "train_loss" in aggregated_metrics:
            print(f"📉 Average Training Loss: {aggregated_metrics['train_loss']:.4f}")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results with metrics tracking."""
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        if metrics_aggregated:
            print(f"\n📈 Evaluation Phase: Aggregated metrics from {len(results)} hospitals")
            print("-" * 80)
            print(f"  Loss:        {loss_aggregated:.4f}")
            print(f"  Accuracy:    {metrics_aggregated.get('accuracy', 0):.4f} ({metrics_aggregated.get('accuracy', 0)*100:.2f}%)")
            print(f"  Precision:   {metrics_aggregated.get('precision', 0):.4f}")
            print(f"  Recall:      {metrics_aggregated.get('recall', 0):.4f} ⭐")
            print(f"  F1-Score:    {metrics_aggregated.get('f1_score', 0):.4f}")
            print(f"  Specificity: {metrics_aggregated.get('specificity', 0):.4f}")
            print(f"  AUC-ROC:     {metrics_aggregated.get('auc_roc', 0):.4f}")
            print("-" * 80)
            
            # Store metrics for plotting
            METRICS_HISTORY["rounds"].append(server_round)
            METRICS_HISTORY["accuracy"].append(metrics_aggregated.get("accuracy", 0))
            METRICS_HISTORY["precision"].append(metrics_aggregated.get("precision", 0))
            METRICS_HISTORY["recall"].append(metrics_aggregated.get("recall", 0))
            METRICS_HISTORY["f1_score"].append(metrics_aggregated.get("f1_score", 0))
            METRICS_HISTORY["specificity"].append(metrics_aggregated.get("specificity", 0))
            METRICS_HISTORY["auc_roc"].append(metrics_aggregated.get("auc_roc", 0))
            METRICS_HISTORY["loss"].append(loss_aggregated)
            
            # Save metrics to file after each round
            with open("metrics_history.json", "w") as f:
                json.dump(METRICS_HISTORY, f, indent=2)
        
        print("="*80 + "\n")
        return loss_aggregated, metrics_aggregated


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy with logging
    strategy = LoggingFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=5,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
