import torch



# 在try.py顶部添加BERT模型定义
class KmerBERT(torch.nn.Module):
    def __init__(self, vocab_size, num_classes=2, hidden_size=768, num_layers=12):
        super().__init__()
        from transformers import BertConfig, BertModel
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=12,
            max_position_embeddings=512,
            pad_token_id=0,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.bert = BertModel(self.config)
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return {
            "logits": self.classifier(pooled_output)
        }
