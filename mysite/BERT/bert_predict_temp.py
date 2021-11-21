# Get the test sets of embeddings with and without corrections and of raw data for Bert
from transformers import BertTokenizer
from mysite.BERT.bert import *

# Define model
def load_model(state_dict_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    hidden_sizes = [64, 32];
    output_size = 4
    model = customBERT(hidden_sizes, output_size, 0.2, 'relu').to(device)
    state_dict = torch.load(state_dict_path, map_location = device)
    model.load_state_dict(state_dict)

    return model


def predict(model, data_path = 'test_data.txt', data_txt = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_txt is None:
        if data_path is None:
            raise Exception('No data is fed')
        else:
            with open(data_path) as f:
                essay = f.read()

    else:
        essay = data_txt

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_dict = tokenizer.encode_plus(essay,
                                           add_special_tokens=True,
                                           padding='max_length',
                                           max_length=512,
                                           return_attention_mask=True,
                                           return_tensors='pt'  # return pytorch tensors
                                           )

    if tokenized_dict['input_ids'].shape[1] > 512:
        # Truncate to 512
        tokenized_essay = tokenized_dict['input_ids'][:, :512]
        tokenized_essay[:, -1] = tokenizer.sep_token_id
        essay_mask = tokenized_dict['attention_mask'][:, :512]

    else:
        tokenized_essay = tokenized_dict['input_ids']
        essay_mask = tokenized_dict['attention_mask']

    test_label = np.asarray([10])
    test_label = torch.tensor(test_label, dtype=torch.float)
    test_idx, test_mask = [tokenized_essay], [essay_mask]
    test_idx = torch.cat(test_idx, dim=0)
    test_mask = torch.cat(test_mask, dim=0)
    test_loader = create_data_loader(test_idx, test_mask, test_label, 20, False)

    model.eval()
    with torch.no_grad():
        y_preds = np.array([0])
        y_true = np.array([0])
        for batch in test_loader:
            input_idx, attention_mask, label = batch
            input_idx = input_idx.to(device)
            attention_mask = attention_mask.to(device)
            logits = model.forward(input_idx, attention_mask)

            logits_np = logits.cpu().detach().numpy()
            y_pred = np.argmax(logits_np, axis=1).reshape(-1)

            # Accumulate
            y_preds = np.concatenate([y_preds, y_pred])
            y_true = np.concatenate([y_true, label.cpu().detach().numpy()])

        y_preds = y_preds[1:]
        y_true = y_true[1:]

        return y_preds, y_true