import torch
from torch import nn
from torch.nn import functional as F

class LoraEmbedding(nn.Module):
    def __init__(self, 
                 base_layer :nn.Embedding) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.adapters = set()
        self.active_adapter = "default"
        self.in_features = base_layer.num_embeddings
        self.out_features = base_layer.embedding_dim

    def add_adapter(self, adapter_name: str, r: int, alpha: int) -> None:
        self.adapters.add(adapter_name)
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha/r
        weight_A = torch.randn((r, self.in_features),device=self.base_layer.weight.device)
        weight_B = torch.randn((self.out_features, r),device=self.base_layer.weight.device)
        self.lora_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_B[adapter_name] = nn.Parameter(weight_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_adapter == "default" or self.active_adapter not in self.adapters \
            or self.active_adapter not in self.lora_A \
                or self.active_adapter not in self.lora_B:
            return self.base_layer(x)
        else:
            result = self.base_layer(x)
            embedding_A = self.lora_A[self.active_adapter].T
            embedding_B = self.lora_B[self.active_adapter].T
            scaling = self.scaling[self.active_adapter]
            after_A = F.embedding(x,embedding_A)
            result = result + (after_A @ embedding_B) * scaling
            return result

class LoraLinear(nn.Module):

    def __init__(self, 
                 base_layer: nn.Linear
                 ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.adapters = set()
        self.active_adapter = "default"

    def add_adapter(self, adapter_name: str, r: int, alpha: int) -> None:
        self.adapters.add(adapter_name)
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha/r
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r,bias=False,device=self.base_layer.weight.device)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features,bias=False,device=self.base_layer.weight.device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_adapter == "default" or self.active_adapter not in self.adapters \
            or self.active_adapter not in self.lora_A \
                or self.active_adapter not in self.lora_B:
            return self.base_layer(x)
        else:
            A = self.lora_A[self.active_adapter](x)
            B = self.lora_B[self.active_adapter](A)
            return B * self.scaling[self.active_adapter] + self.base_layer(x)

def get_parent_target_module(model: torch.nn.Module, key: str):
    final_dot_index = key.rfind('.')
    if final_dot_index == -1:
        parent_key = None
        target_key = key
    else:
        parent_key = key[:final_dot_index]
        target_key = key[final_dot_index+1:]
    parent_module = model.get_submodule(parent_key) if parent_key is not None else model
    return parent_module, target_key

def inject_lora_adapter_with_state_dict(model: torch.nn.Module, 
                                        adapter_name: str, 
                                        state_dict: dict, 
                                        r:int,
                                        alpha:int,
                                        targets,
                                        peft_name :str = None,
                                        parent_model_name :str = None,
                                        pissa_dict: dict=None) -> torch.nn.Module:
    sub_modules = model.named_modules()
    def is_target(key):
        for target in targets:
            if target in key:
                return True
        return False
    for key,sub_module in sub_modules:
        if is_target(key):
            parent_module, target_key = get_parent_target_module(model, key)
            if peft_name is None:
                lora_A = f'{key}.lora_A'
                lora_B = f'{key}.lora_B'
            else:
                if parent_model_name is not None:
                    lora_A = f'{parent_model_name}.{key}.lora_A.{peft_name}'
                    lora_B = f'{parent_model_name}.{key}.lora_B.{peft_name}'
                else:
                    lora_A = f'{key}.lora_A.{peft_name}'
                    lora_B = f'{key}.lora_B.{peft_name}'
                if key != 'emb':
                    lora_A = lora_A+'.weight'
                    lora_B = lora_B+'.weight'
            if pissa_dict is not None:
                pissa_A = f'{key}.init_lora_A'
                pissa_B = f'{key}.init_lora_B'
                
            if lora_A in state_dict and lora_B in state_dict:
                if isinstance(sub_module,nn.Linear):

                    lora_linear = LoraLinear(sub_module)
                    lora_linear.add_adapter(adapter_name,r,alpha)
                    #remove the original module 
                    setattr(parent_module,target_key,lora_linear)

                    #find the corresponding lora_A and lora_B
                    if pissa_dict is not None:
                        lora_linear.lora_A[adapter_name].weight.data = torch.cat((state_dict[lora_A],-pissa_dict[pissa_A]), dim=0).to(lora_linear.lora_A[adapter_name].weight.device)
                        lora_linear.lora_B[adapter_name].weight.data = torch.cat((state_dict[lora_B],pissa_dict[pissa_B]), dim=-1).to(lora_linear.lora_B[adapter_name].weight.device)
                    else:
                        lora_linear.lora_A[adapter_name].weight.data = state_dict[lora_A].to(lora_linear.lora_A[adapter_name].weight.device)
                        lora_linear.lora_B[adapter_name].weight.data = state_dict[lora_B].to(lora_linear.lora_B[adapter_name].weight.device)
                elif isinstance(sub_module,LoraLinear):
                    sub_module.add_adapter(adapter_name,r,alpha)
                    if pissa_dict is not None:
                        sub_module.lora_A[adapter_name].weight.data = torch.cat((state_dict[lora_A],-pissa_dict[pissa_A]), dim=0).to(sub_module.lora_A[adapter_name].weight.device)
                        sub_module.lora_B[adapter_name].weight.data = torch.cat((state_dict[lora_B],pissa_dict[pissa_B]), dim=-1).to(sub_module.lora_B[adapter_name].weight.device)
                    else:
                        sub_module.lora_A[adapter_name].weight.data = state_dict[lora_A].to(sub_module.lora_A[adapter_name].weight.device)
                        sub_module.lora_B[adapter_name].weight.data = state_dict[lora_B].to(sub_module.lora_B[adapter_name].weight.device)
                elif isinstance(sub_module,nn.Embedding):
                    lora_embedding = LoraEmbedding(sub_module)
                    lora_embedding.add_adapter(adapter_name,r,alpha)
                    setattr(parent_module,target_key,lora_embedding)
                    if pissa_dict is not None:
                        lora_embedding.lora_A[adapter_name].data = torch.cat((state_dict[lora_A],-pissa_dict[pissa_A]), dim=0).to(lora_embedding.lora_A[adapter_name].data.device)
                        lora_embedding.lora_B[adapter_name].data = torch.cat((state_dict[lora_B],pissa_dict[pissa_B]), dim=-1).to(lora_embedding.lora_A[adapter_name].data.device)
                    else:
                        lora_embedding.lora_A[adapter_name].data = state_dict[lora_A].to(lora_embedding.lora_A[adapter_name].data.device)
                        lora_embedding.lora_B[adapter_name].data = state_dict[lora_B].to(lora_embedding.lora_A[adapter_name].data.device)
                elif isinstance(sub_module,LoraEmbedding):
                    sub_module.add_adapter(adapter_name,r,alpha)
                    if pissa_dict is not None:
                        sub_module.lora_A[adapter_name].data = torch.cat((state_dict[lora_A],-pissa_dict[pissa_A]), dim=0).to(sub_module.lora_B[adapter_name].data.device)
                        sub_module.lora_B[adapter_name].data = torch.cat((state_dict[lora_B],pissa_dict[pissa_B]), dim=-1).to(sub_module.lora_B[adapter_name].data.device)
                    else:
                        sub_module.lora_A[adapter_name].data = state_dict[lora_A].to(sub_module.lora_B[adapter_name].data.device)
                        sub_module.lora_B[adapter_name].data = state_dict[lora_B].to(sub_module.lora_B[adapter_name].data.device)

def set_adapter(model, adapter_name):
    sub_modules = model.named_modules()
    for key,sub_module in sub_modules:
        if isinstance(sub_module,LoraLinear) or isinstance(sub_module,LoraEmbedding):
            sub_module.active_adapter = adapter_name
