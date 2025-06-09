from src.layers.mlp import EEGMLPClassifier

class EEGMLPEncoder(EEGMLPClassifier):
    """
    MLP Encoder for EEG signal processing.
    This is a specialized version of EEGMLP that focuses on encoding
    node features without a final classification layer.
    """
    def __init__(
        self,
        input_dim=3000,
        hidden_dims=[1024, 512],
        output_dim=128,
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            **kwargs
        )

    def forward(self, x):
        """
        x: [num_nodes, time_steps]  
        returns: [num_nodes, hidden_dim]  
        """
        print("MLP input mean:", x.mean().item(), "std:", x.std().item())
        return self.layers(x)