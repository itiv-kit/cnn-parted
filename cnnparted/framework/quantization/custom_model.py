import torch
from torch import nn

class CustomModel():
    """Base Class for models in which modules can be replaced with explorable modules.
    """

    def __init__(self,
                 base_model: nn.Module,
                 device: torch.device) -> None:
        super().__init__()

        self.base_model = base_model
        self.device = device

        self.explorable_modules = []
        self.explorable_module_names = []

        # Training things
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.base_model.parameters(), lr=0.0001)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=1,
                                                            gamma=0.1)

    def get_explorable_parameter_count(self) -> int:
        return len(self.explorable_modules)

    def load_parameters_file(self, filename: str):
        self.load_parameters(torch.load(filename, map_location=self.device))

    def load_parameters(self, saved_state_dict: dict):
        self.base_model.to(self.device)
        self.base_model.load_state_dict(saved_state_dict,strict=False)

    def save_parameters(self, filename: str):
        torch.save(self.base_model.state_dict(), filename)
