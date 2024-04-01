import torch
import torch.nn as nn
import math
class GRU(nn.Module):
    def __init__(self, seq_length, input_dim, output_dim, memory_size = 64):
        super(GRU, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.memory_size = memory_size
        #self.memory = torch.zeros(self.memory_size)
        self.memoryUpdate = nn.Linear(self.memory_size+self.input_dim, self.memory_size)
        self.forgetGate = nn.Linear(self.memory_size+self.input_dim, self.memory_size)
        self.updateGate = nn.Linear(self.memory_size+self.input_dim, self.memory_size)
        self.output = nn.Linear(self.memory_size, output_dim)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d, nn.Parameter}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x):
        memory = torch.zeros(x.shape[0], self.memory_size)
        outputSeq = torch.zeros((x.shape[0], self.seq_length, 2))
        for timestep in range(0, self.seq_length):

            forgetGate = torch.sigmoid(self.forgetGate(torch.cat([memory, x[:, timestep, :]], dim=1)))
            updateGate = torch.sigmoid(self.updateGate(torch.cat([memory, x[:, timestep, :]], dim=1)))
            forgetMemory = forgetGate*memory
            c_candidate = torch.tanh(self.memoryUpdate(torch.cat([forgetMemory, x[:, timestep, :]], dim=1)))
            memory = updateGate*c_candidate+(1-updateGate)*memory
            logit = self.output(memory)
            probability = torch.softmax(logit, dim=1)
            outputSeq[:, timestep, 0] = logit.squeeze()
            outputSeq[:, timestep, 1] = probability.squeeze()
        return outputSeq


class officialGRU(nn.Module):

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv1d, nn.Parameter, nn.GRU}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def __init__(self, input_dim, output_dim, memory_size = 64):
        super(officialGRU, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.memory_size = memory_size
        self.gru = nn.GRU(input_size=self.input_dim*2, hidden_size=self.memory_size, batch_first=True)
        self.linear2 = nn.Linear(self.memory_size, self.output_dim)
        self.linear1 = nn.Linear(self.input_dim, self.input_dim*2)


    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        output, h = self.gru(x)
        last_output = output[:, -1, :]
        logit = self.linear2(last_output)
        probability = torch.softmax(logit, dim=1)
        return logit, probability












