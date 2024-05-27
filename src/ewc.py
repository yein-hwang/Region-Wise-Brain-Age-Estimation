import torch
import torch.nn as nn
import numpy as np

class EWC(object):
    def __init__(self, model_A, train_dataloader_A):
        self.model = model_A # model on Task A
        self.dataloader = train_dataloader_A
        self.mse_loss_fn = nn.MSELoss()
        # backprop 시 gradient 계산에 영향을 받지 않도록 기존 계산 그래프로부터 파라미터 값을 분리하고 복사
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
                # Fisher Information Matrix  
                # -> 파라미터의 중요도 측정에 사용, 각 파라미터의 그래디언트 제곱을 누적
                # -> 주어진 파라미터에 대한 데이터셋의 log likelihood 함수의 gradient의 기댓값의 제곱으로 정의
            for n, p in self.model.named_parameters(): # model.named_parameters() -> 모델의 모든 파라미터와 그 이름을 반환
                if p.grad is not None: # 해당 파라미터가 그래디언트를 가지고 있는지 확인 (훈련 중에 사용되지 않는 파라미터(예: 학습 과정에서 고정된 파라미터)는 그래디언트가 없을 수 있음)
                    # p.grad.pow(2) -> "Emprical Fisher Information Matrix(loss function의 gradient의 제곱을 사용)"를 사용하여 근사 
                        # 변화량의 절대 크기 강조: 제곱을 사용하면 작은 차이는 더 작아지고, 큰 차이는 더 커지므로, 파라미터의 변화량이 클수록 패널티가 크게 증가
                        # 방향성 무시: 제곱을 통해 파라미터의 변화가 양의 방향이든 음의 방향이든 그 크기만을 고려 -> 파라미터의 증가 또는 감소 모두 동일하게 패널티를 부과
                    self.fisher[n] += p.grad.pow(2)
                    
        # 데이터셋 전체에 대한 평균을 계산하여 Fisher 정보를 최종적으로 업데이트
            # 각 파라미터에 대한 그래디언트의 제곱이 데이터셋의 모든 샘플에 대해 어떻게 분포하는지를 반영
            # 배치 처리를 통해 계산된 gradient sum of squares는 각 batch에서 계산된 값들의 누적이므로, 모든 데이터를 고르게 고려하기 위해 전체 데이터셋의 크기로 나누어 평균 취하기
        for n in self.fisher:
            self.fisher[n] /= len(self.dataloader.dataset)


    def penalty(self, model):
        ewc_loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                diff = (self.params[n] - p)
                fisher_effect = self.fisher[n] * diff.pow(2)
                ewc_loss += fisher_effect.sum()
        # print(f"Total EWC Loss: {ewc_loss.item()}")
        return ewc_loss



# class EWC(object):
#     def __init__(self, model_A, train_dataloader_A):
#         self.model = model_A # model on Task A
#         self.dataloader = train_dataloader_A
#         self.mse_loss_fn = nn.MSELoss()
#         self.params = {n: p.clone().detach() for n, p in model_A.named_parameters() if p.requires_grad}
#         self.fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        
#         self.model.eval()
#         for _, (input, target) in enumerate(self.dataloader):
#             input = input.cuda(non_blocking=True)
#             target = target.reshape(-1, 1)
#             target = target.cuda(non_blocking=True)
#             output = self.model(input)

#             self.model.zero_grad()
#             mse_loss = self.mse_loss_fn(output, target)
#             mse_loss.backward()

#             for n, p in self.model.named_parameters(): 
#                 if p.grad is not None: 
#                     self.fisher[n] += p.grad.pow(2)

#         for n in self.fisher:
#             self.fisher[n] /= len(self.dataloader.dataset)


#     def penalty(self, model):
#         ewc_loss = 0
#         for n, p in model.named_parameters():
#             if n in self.fisher:
#                 diff = (self.params[n] - p)
#                 fisher_effect = self.fisher[n] * diff.pow(2)
#                 ewc_loss += fisher_effect.sum()
#         return ewc_loss
