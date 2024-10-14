import torch.nn as nn
import torch
import importlib
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time

def instantiate_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_()

class PredModel(nn.Module):
    """
    This class represents a fully connected neural network model with given layer sizes and activation function.
    """
    def __init__(self, pred_model, num_labels, mtype, activation, max_length, max_batch_size, tokenizer_name=None):
        """
        :param sizes: list of layer sizes (excluding the input layer size which is given by n_features parameter)
        :param input_norm: flag indicating whether to perform layer normalization on the input
        :param activation: name of the PyTorch activation function, e.g. Sigmoid or Tanh
        :param dropout: dropout probability
        :param n_features: number of input features
        """
        super(PredModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(pred_model,                                                           num_labels=num_labels)
        self.mtype = mtype
        self.tokenizer = AutoTokenizer.from_pretrained(pred_model if tokenizer_name is None else tokenizer_name)
        self.activation = torch.nn.Identity() if activation is None else instantiate_class(
            "torch.nn.modules.activation", activation)
        self.max_length = max_length
        self.max_batch_size = max_batch_size
        self.num_labels = num_labels
        if self.mtype == "rank":
            assert num_labels == 1

    
    
    @torch.no_grad()
    def score(self, prompts):
        ret = []
        ts = 0
        for i in range(0, len(prompts), self.max_batch_size):
            cur_prompts = prompts[i:min(len(prompts), i + self.max_batch_size)] 
            t1 = time.time()
            inps = self.tokenizer(cur_prompts, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
            input_ids = inps['input_ids'].to("cuda:0")
            attention_mask = inps['attention_mask'].to("cuda:0")
            t2 = time.time()
            ts += t2 - t1
            if self.mtype == "class":
                ret.append( self.model(input_ids, attention_mask).argmax(dim=-1) )
            elif self.mtype == "rank":
                ret.append( self.activation(self.model(input_ids, attention_mask).logits) )
        return ret
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the FCModel.
        :param x: input of shape [batch_size, slate_length, self.layers[0].in_features]
        :return: output of shape [batch_size, slate_length, self.output_size]
        """
        #outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        #seq_lengths = attention_mask.cumsum(dim=1).argmax(dim=1)

        # 从hidden_states中提取每个序列最后一个token的hidden state
        # print(len(outputs.hidden_states))
        #last_token_hidden_states = outputs.hidden_states[-2][torch.arange(outputs.hidden_states[-1].size(0)), seq_lengths]
        # print(seq_lengths)

        # 通过预测头进行预测
        # input_norm =self.layernorm_input(last_token_hidden_states)
        #prediction = self.tanh(self.layer2(self.gelu(self.layer1(last_token_hidden_states)))).reshape(1,-1)
        #prediction = self.activation(self.model.score(last_token_hidden_states)).reshape(1,-1)

        #return prediction
        if self.mtype == "class":
            return self.model(input_ids, attention_mask).logits
        elif self.mtype == "rank":
            return self.activation(self.model(input_ids, attention_mask).logits)
        else:
            assert False, "not support"

def prefill_predictor_model(pred_model, num_labels, mtype, activation, max_length, max_batch_size, tokenizer_name=None):
    """
    Helper function for instantiating LTRModel.
    :param fc_model: FCModel used as input block
    :param transformer: transformer Encoder used as encoder block
    :param post_model: parameters dict for OutputModel output block (excluding d_model)
    :param n_features: number of input features
    :return: LTR model instance
    """
    model = PredModel(pred_model, num_labels, mtype, activation, max_length, max_batch_size, tokenizer_name)
    
    return model
