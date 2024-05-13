import torch
import torch.nn as nn
import numpy as np

class EWC(object):
    def __init__(self, model_A, valid_dataloader_A):
        self.model = model_A # model on Task A
        self.dataloader = valid_dataloader_A
        self.mse_loss_fn = nn.MSELoss()
        self.mae_loss_fn = nn.L1Loss()
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}

        # test
        self.model.eval()
        for input, _ in self.dataloader:
            input = input.cuda(non_blocking=True)
            target = target.reshape(-1, 1)
            target = target.cuda(non_blocking=True)
            output = self.model(input)

            self.model.zero_grad()
            mse_loss = self.mse_loss_fn(output, target)
            mae_loss = self.mae_loss_fn(output, target)

            mse_loss.backward()

            # Fihser Information Matrix를 계산하고 업데이트 하는 과정의 일부
                # Fisher Information Matrix  -> 파라미터의 중요도 측정에 사용
            for n, p in self.model.named_parameters(): # model.named_parameters() -> 모델의 모든 파라미터와 그 이름을 반환
                if p.grad is not None: # 해당 파라미터가 그래디언트를 가지고 있는지 확인 (훈련 중에 사용되지 않는 파라미터(예: 학습 과정에서 고정된 파라미터)는 그래디언트가 없을 수 있음)
                    # p.grad.pow(2) -> "Emprical Fisher Information Matrix(loss function의 gradient의 제곱을 사용)"를 사용하여 근사 
                    # self.fisher[n] += ... -> Fisher information의 각 파라미터 항목 업데이트
                    self.fisher[n] += p.grad.pow(2) * len(self.dataloader) 

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.params:
                # pow(2) -> 모델 파라미터 변화에 대한 "패널티" 또는 "cost"를 계산하기 위함
                # self.params[n].data -> Task A에서의 파라미터 값
                # p.data -> Current Task 에서의 파라미터 값
                # self.fisher[n] -> 해당 파라미터의 중요도를 나타내느 Fisher Information
                # 즉, 파라미터 변화의 크기에 그 중요도를 곱하여, 최종적인 패널티 값을 계산 -> 중요도가 높은 파라미터는 더 큰 패널티를 받게 됨
                loss += (self.fisher[n] * (self.params[n].data - p.data).pow(2)).sum()
            
        return loss