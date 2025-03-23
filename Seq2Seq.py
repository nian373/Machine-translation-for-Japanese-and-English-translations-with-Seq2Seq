import os
import collections
import math
import timer
from torch import nn
import torch
from d2l import torch as d2l

# 修复数据加载
def read_data_nmt():
    data_path = r'jpn-eng\jpn.txt'
    pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) >= 2:
                src, tgt = parts[0].strip(), parts[1].strip()
                if src and tgt:  # 排除空字符的句子对
                    pairs.append((src, tgt))


    return pairs


# 重新加载数据
pairs = read_data_nmt()
print(f"处理后有效样本数: {len(pairs)}")

def preprocess_nmt(sentence):
    """预处理单个句子"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    sentence = sentence.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = []
    for i, char in enumerate(sentence):
        if i > 0 and no_space(char, sentence[i-1]):
            out.append(' ')
        out.append(char)
    return ''.join(out)

# 预处理所有句子对
preprocessed_pairs = []
for src, tgt in pairs:
    pre_src = preprocess_nmt(src)  # 预处理英语句子
    pre_tgt = preprocess_nmt(tgt)  # 预处理日语句子
    preprocessed_pairs.append((pre_src, pre_tgt))

# 显示预处理后的样本
print("预处理后的样本:")
for i in range(3):
    print(f"英语: {preprocessed_pairs[i][0]}")
    print(f"日语: {preprocessed_pairs[i][1]}")
    print("-" * 50)

# 修改后的分词函数
def tokenize_nmt(text_pairs):
    source, target = [], []
    for src, tgt in text_pairs:
        # 英语按空格分词
        source.append(src.split())
        # 日语按字符分词
        target.append(list(tgt))
    return source, target

# 分词
source, target = tokenize_nmt(preprocessed_pairs)
print("\n分词结果示例:")
print("英语:", source[0])
print("日语:", target[0])

# 词表
src_vocab = d2l.Vocab(source, min_freq=1, reserved_tokens=['<pad>', '<bos>', '<eos>'])
tgt_vocab = d2l.Vocab(target, min_freq=1, reserved_tokens=['<pad>', '<bos>', '<eos>'])
print("源语言词表大小:", len(src_vocab))
print("目标语言词表大小:", len(tgt_vocab))

##加载数据集
def truncate_pad(line, max_len, padding_token):
    if len(line) > max_len:
        return line[:max_len]
    return line + [padding_token] * (max_len - len(line))

print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))


#转换成小批量数据用于训练
def build_array_nmt(lines, vocab, num_steps):
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    cleaned_lines = []
    for l in lines:
        cleaned = [token for token in l if token != vocab['<pad>']]  # 移除所有填充符（确保逻辑正确）
        if len(cleaned) == 0:  # 如果全为填充符，则跳过
            continue
        cleaned_lines.append(cleaned)
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in cleaned_lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


#训练模型
def load_data_nmt(batch_size, num_steps, num_examples=10000):
    """返回翻译数据集的迭代器和词表"""
    raw_pairs = read_data_nmt()[:num_examples]  # 读取前num_examples个例子
    # 预处理每个句子对
    preprocessed_pairs = []
    for src, tgt in raw_pairs:
        pre_src = preprocess_nmt(src)
        pre_tgt = preprocess_nmt(tgt)
        preprocessed_pairs.append((pre_src, pre_tgt))
    # 分词
    source, target = tokenize_nmt(preprocessed_pairs)
    # 构建词表
    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 转换为张量
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
##读出第一个小批量数据
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=10)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break

##实现编码器
class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
print(output.shape)
print(state.shape)

##解码器
class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        # print("After embedding:", X.shape)  # 应为 (num_steps, batch_size, embed_size)

        context = state[-1].repeat(X.shape[0], 1, 1)
        # print("Context shape:", context.shape)  # 应为 (num_steps, batch_size, num_hiddens)

        X_and_context = torch.cat((X, context), 2)
        # print("X_and_context shape:", X_and_context.shape)  # (num_steps, batch_size, embed_size + num_hiddens)

        output, state = self.rnn(X_and_context, state)
        # print("RNN output shape:", output.shape)  # 应为 (num_steps, batch_size, num_hiddens)

        output = self.dense(output)
        # print("After dense layer:", output.shape)  # 应为 (num_steps, batch_size, vocab_size)

        output = output.permute(1, 0, 2)
        # print("Final output shape:", output.shape)  # 应为 (batch_size, num_steps, vocab_size)

        return output, state


decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
print(output.shape, state.shape)

##损失函数
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(sequence_mask(X, torch.tensor([1, 2])))


X = torch.ones(2, 3, 4)
print(sequence_mask(X, torch.tensor([1, 2]), value=-1))

##定义交叉熵损失
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

loss = MaskedSoftmaxCELoss()
loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
     torch.tensor([4, 2, 0]))


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for batch in data_iter:
            l = None  # 初始化 l
            try:
                X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
                if Y_valid_len.sum().item() == 0:  # 跳过无效批次
                    print("跳过无效批次（Y_valid_len全为0）")
                    continue

                bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
                dec_input = torch.cat([bos, Y[:, :-1]], 1)
                outputs = net(X, dec_input, X_valid_len)
                Y_hat = outputs[0]

                l = loss(Y_hat, Y, Y_valid_len)
                l.sum().backward()
                d2l.grad_clipping(net, 1)
                num_tokens = Y_valid_len.sum()
                optimizer.step()

                if l is not None:
                    metric.add(l.sum().item(), num_tokens.item())
            except Exception as e:
                print(f"训练错误: {e}")
                continue

        if metric[1] == 0:
            print(f"Epoch {epoch}: 无有效数据")
            continue

        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))

    print(f'loss {metric[0] / metric[1]:.3f}')


#
embed_size, num_hiddens, num_layers, dropout = 256, 512, 3, 0.2
batch_size, num_steps = 64, 20
lr, num_epochs, device = 0.001, 300, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps,num_examples = 100000)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
#预测
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


#
engs = ['hello .', "i am a man .", 'i have a room .', 'wish you happiness .']
jpn = ['こんにちは .', 'アイ・アム・ア・マン .', '私には部屋がある .', 'ご多幸を祈ります .']
for eng, jpn in zip(engs, jpn):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, jpn, k=2):.3f}')











