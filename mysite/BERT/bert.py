# Import tokenizer
import torch

# Import the BERT Model and add few layers for classification downstream
from transformers import BertModel, BertConfig
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class customBERT(nn.Module):
    def __init__(self, hidden_sizes, output_size, bert_dropout=0.1, activation='relu'):
        # hidden_sizes excludes the output, excludes the first layer out of BERT since this is 768
        super(customBERT, self).__init__()

        # BertConfig only works for bert-base-uncased. Furthermore, since we are loading
        # pre-trained, not all config attributes can be changed
        config = BertConfig(output_hidden_states=False, output_attentions=False)
        config.hidden_dropout_prob = bert_dropout
        config.attention_probs_dropout_prob = bert_dropout

        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              config=config)

        # Add hidden layers
        self.hidden_layers = [];
        input_size = 768  # this is default from BERT
        for out_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(input_size, out_size))
            input_size = out_size

        # Add last layer for classification
        self.hidden_layers.append(nn.Linear(input_size, output_size))
        self.batch_norm = torch.nn.BatchNorm1d(hidden_sizes[0])

        # Wrap with nn.ModuleList so everything is moved to CUDA
        self.hidden_layers = torch.nn.ModuleList(self.hidden_layers)

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'lrelu':
            self.activation = F.leaky_relu
        elif activation == 'tanh':
            self.activation = F.tanh

    # Forward takes tokenized ids, and the attention mask

    def forward(self, input, pad_mask):
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel for documentation
        outputs = self.bert(input, attention_mask=pad_mask)
        _, pooled_output = outputs[0], outputs[1]

        # Use only the pooled_output, of shape (batch_size, 768)
        out = pooled_output
        for layer_idx in range(len(self.hidden_layers)):
            layer = self.hidden_layers[layer_idx]
            out = layer(out)
            if layer_idx == 0:
                out = self.batch_norm(out)

            if layer_idx < len(self.hidden_layers) - 1:
                out = self.activation(out)

        return out


# def train_step(model, input_id, att_mask, label, optimizer, weights=None):
#     model.train()
#     # BERT is a big model, might need to use smaller batch sizes to be able to train it without running out of memory
#     # Batching is done outside using dataloader
#
#     # Assume inputs are already tokenized. Hence input is tuple of (embedding idx, attention_mask)
#     # Also, already in tensor format, and moved to device.
#     # Label = list of integers
#     if weights is not None:
#         loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(device))
#     else:
#         loss_fn = nn.CrossEntropyLoss()
#
#     optimizer.zero_grad()
#     y_preds = model.forward(input_id, att_mask)
#     loss_val = loss_fn(y_preds, torch.tensor(label, dtype=torch.long).to(device))
#
#     loss_value = loss_val.item()
#
#     loss_val.backward()
#     nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.8)
#     optimizer.step()
#
#     return None


def create_data_loader(input_embedding_idx, mask_tensor, label, num_batches=100, shuffle=True):
    """
    First 3 arguments must be tensors
    """
    batch_size = round((input_embedding_idx.shape[0] - 3) / num_batches)
    if batch_size < 1:
        batch_size = 1

    dataset_obj = torch.utils.data.TensorDataset(input_embedding_idx, mask_tensor, label)
    dataset_loader = torch.utils.data.DataLoader(dataset_obj, batch_size=batch_size, shuffle=shuffle)
    return dataset_loader


def compute_class_weights(labels):
    # Labels in torch.tensor
    labels_np = torch.clone(labels).cpu().numpy()
    _, class_lengths = np.unique(labels_np, return_counts=True)
    max = np.sum(class_lengths) / class_lengths.shape[0]
    weights = max / class_lengths
    return weights, class_lengths