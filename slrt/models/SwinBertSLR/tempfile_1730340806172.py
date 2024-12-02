import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# 自定义词汇表文件路径
vocab_file = "path/to/custom_vocab.txt"

# 创建自定义的Tokenizer
tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)

# 加载预训练模型
model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 调整嵌入层的大小
model.resize_token_embeddings(len(tokenizer))

# 准备输入句子
sentence = "这是一个测试句子。"
inputs = tokenizer(sentence, return_tensors="pt")

# 获取模型输出
with torch.no_grad():
    outputs = model(**inputs)

# 提取 [CLS] 令牌的隐藏状态
cls_embedding = outputs.logits

# 转换为numpy数组（可选）
cls_embedding_np = cls_embedding.numpy()

print("初始特征向量：")
print(cls_embedding_np)

# 加载数据集
dataset = load_dataset("glue", "mrpc")


# 数据预处理
def preprocess_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding=True)


encoded_dataset = dataset.map(preprocess_function, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
)

# 开始微调
trainer.train()

# 保存微调后的模型
output_dir = "./finetuned_model"
trainer.save_model(output_dir)
# 保存Tokenizer
tokenizer.save_pretrained(output_dir)

# 再次获取模型输出
with torch.no_grad():
    outputs = model(**inputs)

# 提取 [CLS] 令牌的隐藏状态
cls_embedding_finetuned = outputs.logits

# 转换为numpy数组（可选）
cls_embedding_finetuned_np = cls_embedding_finetuned.numpy()

print("微调后的特征向量：")
print(cls_embedding_finetuned_np)
