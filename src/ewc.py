import torch
import torch.nn as nn
import numpy as np

class EWC(object):
    def __init__(self, model_A, train_dataloader_A):
        self.model = model_A # model on Task A
        self.dataloader = train_dataloader_A
        self.mse_loss_fn = nn.MSELoss()
        self.params = {n: p.clone().detach() for n, p in model_A.named_parameters() if p.requires_grad}
        ''' model.named_parameters() -> 모델의 각 파라미터에 대한 이름과 해당 파라미터 객체를 포함하는 튜플을 제너레이터로 반환
        z.B) named_parameters를 사용하여 파라미터 출력
        for name, param in model.named_parameters():
            print(f"Layer Name: {name}, Parameter Size: {param.size()}")
        Layer Name: conv1.weight, Parameter Size: torch.Size([20, 1, 5, 5])
        Layer Name: conv1.bias, Parameter Size: torch.Size([20])
        Layer Name: conv2.weight, Parameter Size: torch.Size([20, 20, 5, 5])
        Layer Name: conv2.bias, Parameter Size: torch.Size([20])
        '''
        self.fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        
        self.model.eval()
        for _, (input, target) in enumerate(self.dataloader):
            input = input.cuda(non_blocking=True)
            target = target.reshape(-1, 1)
            target = target.cuda(non_blocking=True)
            output = self.model(input)

            self.model.zero_grad()
            mse_loss = self.mse_loss_fn(output, target)
            mse_loss.backward()

            # Fihser Information Matrix를 계산하고 업데이트 하는 과정의 일부
                # Fisher Information Matrix  -> 파라미터의 중요도 측정에 사용, 각 파라미터의 그래디언트 제곱을 누적
            for n, p in self.model.named_parameters(): # model.named_parameters() -> 모델의 모든 파라미터와 그 이름을 반환
                if p.grad is not None: # 해당 파라미터가 그래디언트를 가지고 있는지 확인 (훈련 중에 사용되지 않는 파라미터(예: 학습 과정에서 고정된 파라미터)는 그래디언트가 없을 수 있음)
                    # p.grad.pow(2) -> "Emprical Fisher Information Matrix(loss function의 gradient의 제곱을 사용)"를 사용하여 근사 
                    # input.size(0) -> 각 배치 사이즈
                    # self.fisher[n] += ... -> Fisher information의 각 파라미터 항목 업데이트
                    self.fisher[n] += p.grad.pow(2)
                else:
                    print(f"{n}: p.grad is None")
        # 데이터셋 전체에 대한 평균을 계산하여 Fisher 정보를 최종적으로 업데이트
        for n in self.fisher:
            self.fisher[n] /= len(self.dataloader.dataset)
        
        print("After: ", self.fisher)

    # 1. '.data'를 사용해서 tensor에서 데이터를 직접 추출
    # def penalty(self, model):
    #     loss = 0
    #     for n, p in model.named_parameters():
    #         if n in self.params:
    #             # pow(2) -> 모델 파라미터 변화에 대한 "패널티" 또는 "cost"를 계산하기 위함
    #             # self.params[n].data -> Task A에서의 파라미터 값
    #             # p.data -> Current Task 에서의 파라미터 값
    #             # self.fisher[n] -> 해당 파라미터의 중요도를 나타내느 Fisher Information
    #             # 즉, 파라미터 변화의 크기에 그 중요도를 곱하여, 최종적인 패널티 값을 계산 -> 중요도가 높은 파라미터는 더 큰 패널티를 받게 됨
    #             loss += (self.fisher[n] * (self.params[n].data - p.data).pow(2)).sum()
            
    #     return loss

    # 2. 텐서에서 데이터 추출 안하고 바로 텐서 연산을 수행
    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                diff = (self.params[n] - p)
                fisher_effect = self.fisher[n] * diff.pow(2)
                print(f"Param: {n}, Diff: {diff.norm().item()}, Fisher Effect: {fisher_effect.sum().item()}")
                loss += fisher_effect.sum()
        print(f"Total EWC Loss: {loss.item()}")
        return loss

