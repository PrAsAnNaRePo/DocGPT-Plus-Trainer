from torch.utils.data import Dataset
from datasets import load_dataset

def get_prompt(example):
    doc_content = ''
    for sentence in example['context']['sentences']:
        for line in sentence:
            doc_content += line
        doc_content += '\n\n# ---------------------------\n\n'
    return {
        "prompt": f"""<|endoftext|><|user|>
Given the following contents, Answer the question correctly and relevently.
{doc_content}
question: {example['question']}<|end|>
<|assistant|>
{example['answer']}<|endoftext|>"""
    }

class DocDataset(Dataset):
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        
        self.dataset = load_dataset('hotpotqa/hotpot_qa', 'fullwiki', split='train')
        clms = self.dataset.column_names
        self.dataset = self.dataset.map(get_prompt)
        self.dataset = self.dataset.remove_columns(clms)

        print('len of dataset: ', len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        inputs = self.tokenizer(item['prompt'], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
        return inputs